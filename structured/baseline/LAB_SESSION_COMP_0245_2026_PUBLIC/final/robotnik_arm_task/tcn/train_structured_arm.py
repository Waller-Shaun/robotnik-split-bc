"""Auxiliary-supervised neural arm policy; original train/val split preserved.

Read recorded q_des_arm/qd_des_arm as TRAINING labels only. They are never
available to the deployed policy. Gravity labels are reconstructed only on
unsaturated torque coordinates. No new expert data or test data is read.
"""
from datetime import datetime,timezone
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from tcn_dataset import DEFAULT_ROOT as ROOT,TASK_DIR
from robotnik_bodyframe_features import build_bodyframe_observations
from structured_arm import StructuredArm,features,decode,FEATURE_VERSION
from train_tcn import compute_stats,write_json
from train_arm_compare import NAV,sha


class Windows(Dataset):
    def __init__(self,split,manifest,limit=None):
        self.filenames=list(manifest['splits'][split])
        if limit:
            heights=manifest['targets'][split]
            chosen=[]
            for low in (True,False):
                chosen.extend([i for i,t in enumerate(heights) if (t[2]<=.5)==low][:2])
            self.filenames=[self.filenames[i] for i in chosen]
        self.x=[];self.labels=[];self.groups=[];self.weights=[]
        self.ends=[0]
        for filename in self.filenames:
            with np.load(ROOT/'RoboEnv'/filename,allow_pickle=False) as d:
                indices=np.arange(0,len(d['observations']),10)
                obs=d['observations'][indices].astype(np.float32)
                x=features(build_bodyframe_observations(obs,d['ee_positions'][indices]),d['base_orientations'][indices])
                v=d['qd_des_arm'][indices].astype(np.float32)
                lead=100*(d['q_des_arm'][indices]-obs[:,5:12])-30*obs[:,12:19]
                torque=d['actions'][indices,4:].astype(np.float32)
                gravity=torque-lead-30*v
                mask=(np.abs(torque)<29.9).astype(np.float32)
                label=np.concatenate((v,gravity,lead,torque,mask),axis=1).astype(np.float32)
                if not np.isfinite(label).all():
                    raise ValueError('Nonfinite auxiliary labels')
                self.x.append(x);self.labels.append(label)
                low=bool(d['target'][2]<=.5+1e-6)
                self.groups.append(np.full(len(x),low,dtype=bool))
                focus=(x[:,24]==0)&(np.linalg.norm(x[:,25:28],axis=1)<.10)
                self.weights.append(np.where(focus,3.,1.))
                self.ends.append(self.ends[-1]+len(x))
        self.groups=np.concatenate(self.groups)
        self.weights=np.concatenate(self.weights)
        if self.groups.all() or not self.groups.any():
            raise ValueError('Both height groups required')
        for low in (True,False):
            m=self.groups==low
            self.weights[m]*=.5/self.weights[m].sum()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self,index):
        episode=int(np.searchsorted(self.ends,index,side='right')-1)
        frame=index-self.ends[episode]
        raw=self.x[episode]
        history=raw[np.maximum(frame-np.arange(19,-1,-1),0)]
        return (torch.from_numpy((history-self.stats['x_mean'])/self.stats['x_std']),
                torch.from_numpy(raw[frame]),torch.from_numpy(self.labels[episode][frame]))


def epoch(model,loader,stats,device,opt=None):
    model.train(opt is not None)
    tensors={k:torch.from_numpy(v).to(device) for k,v in stats.items()}
    sums=np.zeros(4);count=0
    with torch.set_grad_enabled(opt is not None):
        for history,current,label in loader:
            history,current,label=history.to(device),current.to(device),label.to(device)
            v,g=decode(model(history),current,tensors['g_mean'],tensors['g_std'])
            vl=((v-label[:,:7])/tensors['v_std']).square().mean()
            mask=label[:,28:35]
            gl=(((g-label[:,7:14])/tensors['g_std']).square()*mask).sum()/mask.sum().clamp_min(1)
            torque=(label[:,14:21]+30*v+g).clamp(-30,30)
            tl=((torque-label[:,21:28])/tensors['torque_std']).square().mean()
            loss=vl+.2*gl+.2*tl
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite structured loss')
            if opt:
                opt.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
                opt.step()
            sums+=np.array([float(loss.detach()),float(vl.detach()),float(gl.detach()),float(tl.detach())])*len(label)
            count+=len(label)
    return dict(zip(('loss','velocity_mse','gravity_mse','teacher_forced_torque_mse'),(sums/count).tolist()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=('mlp','tcn'),required=True)
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--samples',type=int,default=200000)
    p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--normalization',choices=('robust','empirical'),default='robust')
    args=p.parse_args()
    if min(args.epochs,args.samples)<1 or not np.isfinite(args.lr) or args.lr<=0:
        p.error('Invalid hyperparameters')
    torch.manual_seed(42);np.random.seed(42);torch.set_num_threads(2)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    manifest_path=ROOT/'dataset_info/manifests/merged998/data_split.json'
    manifest=json.loads(manifest_path.read_text())
    data={}
    for split in ('train','val'):
        print(f'Loading {split} auxiliary labels; TEST UNUSED',flush=True)
        data[split]=Windows(split,manifest,4 if args.smoke_test else None)
    xm,xs=compute_stats(data['train'].x)
    if args.normalization=='robust':
        # Nearly constant cos(q7) and gravity-z must not amplify tiny changes
        # by 1e5. Natural physical scales, fixed before validation outcomes.
        xs[:14]=np.maximum(xs[:14],.1)
        xs[14:21]=np.maximum(xs[14:21],.1)
        xs[28:31]=np.maximum(xs[28:31],.05)
    # Leave the phase feature binary; its physical value is also used in decode.
    xm[24]=0;xs[24]=1
    labels=np.concatenate(data['train'].labels)
    gmean=[];gstd=[]
    for j in range(7):
        valid=labels[labels[:,28+j]>0,7+j]
        gmean.append(valid.mean(dtype=np.float64));gstd.append(max(valid.std(dtype=np.float64),.5))
    stats=dict(x_mean=xm,x_std=xs,g_mean=np.array(gmean,dtype=np.float32),g_std=np.array(gstd,dtype=np.float32),
               v_std=np.maximum(labels[:,:7].std(axis=0,dtype=np.float64),.05).astype(np.float32),
               torque_std=np.maximum(labels[:,21:28].std(axis=0,dtype=np.float64),.5).astype(np.float32))
    del labels
    for source in data.values():source.stats=stats
    sampler=WeightedRandomSampler(torch.from_numpy(data['train'].weights),
             min(2048 if args.smoke_test else args.samples,len(data['train'])),True,
             generator=torch.Generator().manual_seed(42))
    loaders={'train':DataLoader(data['train'],batch_size=1024,sampler=sampler),
             'val':DataLoader(data['val'],batch_size=1024)}
    model=StructuredArm(args.model).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    max_epochs=2 if args.smoke_test else args.epochs
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,max_epochs,eta_min=args.lr/10)
    run=ROOT/'RoboEnv'/f'bc_structured_{args.model}'/datetime.now(timezone.utc).strftime(
        'run_%Y%m%d_%H%M%S_%f_'+('smoke' if args.smoke_test else 'full'))
    run.mkdir(parents=True,exist_ok=False)
    sources=[Path(__file__),Path(__file__).with_name('structured_arm.py'),
             TASK_DIR/'robotnik_bodyframe_features.py']
    config=dict(schema_version=3,model_type=args.model,feature_version=FEATURE_VERSION,
                seed=42,raw_dt_s=.001,window_size=20,stride=10,effective_context_samples=20 if args.model=='tcn' else 1,
                mode='smoke' if args.smoke_test else 'full',trained_from_scratch=True,
                architecture='learned gain field and gravity; reference integrator plus fixed PD tracking',
                is_hybrid_controller=True,expert_ik_during_inference=False,expert_gravity_during_inference=False,
                kp=100.,kd=30.,max_lead=.15,max_joint_velocity=1.,torque_limit=30.,
                source_manifest_sha256=sha(manifest_path),nav_run=str(NAV),
                frozen_nav_file_sha256={n:sha(NAV/n) for n in ('best_model.pt','config.json','normalization.npz','data_split.json')},
                source_sha256={s.name:sha(s) for s in sources},parameters=sum(p.numel() for p in model.parameters()),
                auxiliary_labels=['qd_des_arm','unsaturated gravity reconstructed from q_des_arm and actions'],
                inference_internal_state='self integrated learned velocity, never recorded expert q_des',
                sampling='low z<=0.5 / other 50:50; 3x focus for reaching error below 10cm',
                selection_metric='validation auxiliary loss; closed-loop assessment separate',
                max_epochs=max_epochs,batch_size=1024,learning_rate=args.lr,samples_per_epoch=sampler.num_samples,
                normalization_policy=args.normalization,
                normalization_std_floors={'sin_cos_q':.1,'joint_velocity':.1,'unit_gravity_direction':.05} if args.normalization=='robust' else {},
                test_evaluated=False,device=str(device),torch_version=str(torch.__version__))
    write_json(run/'config.json',config);np.savez(run/'normalization.npz',**stats)
    shutil.copy2(manifest_path,run/'data_split.json')
    write_json(run/'used_trajectories.json',{**{s:d.filenames for s,d in data.items()},'test':[]})
    for source in sources:shutil.copy2(source,run/source.name)
    print(f'RUN: {run}; parameters: {config["parameters"]}; device: {device}',flush=True)
    best=float('inf');best_epoch=0;stale=0
    def save(name,number,metrics):
        torch.save(dict(schema_version=3,model_type=args.model,epoch=number,config=config,validation=metrics,
                  normalization={k:torch.from_numpy(v) for k,v in stats.items()},
                  model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),run/name)
    with (run/'history.csv').open('w',newline='') as stream:
        fields=['epoch','seconds']+[f'{s}_{m}' for s in ('train','val') for m in ('loss','velocity_mse','gravity_mse','teacher_forced_torque_mse')]
        writer=csv.DictWriter(stream,fields);writer.writeheader();stream.flush()
        for number in range(1,max_epochs+1):
            start=time.perf_counter()
            tr=epoch(model,loaders['train'],stats,device,optimizer)
            va=epoch(model,loaders['val'],stats,device)
            scheduler.step();seconds=time.perf_counter()-start
            writer.writerow(dict(epoch=number,seconds=seconds,**{f'train_{k}':v for k,v in tr.items()},
                                 **{f'val_{k}':v for k,v in va.items()}));stream.flush()
            print(f'Epoch {number:02d}: train={tr["loss"]:.6f}; val={va["loss"]:.6f}; velocity={va["velocity_mse"]:.6f}; gravity={va["gravity_mse"]:.6f}; {seconds:.1f}s',flush=True)
            if va['loss']<best:
                best,best_epoch,stale=va['loss'],number,0;save('best_model.pt',number,va)
            else:stale+=1
            if number in (5,10,20,30):save(f'epoch_{number:03d}.pt',number,va)
            if stale>=8:break
    save('last_model.pt',number,va)
    write_json(run/'training_finished.json',dict(status='completed',best_epoch=best_epoch,
               epochs_completed=number,best_validation_loss=best,subset_only=args.smoke_test,test_evaluated=False))
    print(f'COMPLETE: {run}; best epoch {best_epoch}',flush=True)


if __name__=='__main__':main()
