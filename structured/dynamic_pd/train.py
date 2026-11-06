"""Expert-torque supervised gain-head learning; no RL and no final test access."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset,WeightedRandomSampler
from .common import ROOT,BASE,sha,write,check_baseline
from .dataset import prepare
from .model import GainNetwork,supervised_torque


class Samples(Dataset):
    def __init__(self,values):
        self.values = values

    def __len__(self):
        return len(self.values['hidden'])

    def __getitem__(self,i):
        a = self.values
        return tuple(torch.from_numpy(a[key][i].copy()) for key in ('hidden','q_error','velocity','qd','gravity','torque')) + (
            torch.from_numpy(a['hidden'][a['previous'][i]].copy()),)


def run_epoch(model,loader,scale,device,optimizer=None):
    model.train(optimizer is not None)
    count = 0
    totals = np.zeros(4)
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            hidden,q_error,v,qd,g,target,previous = [x.to(device) for x in batch]
            gains,delta = model(hidden)
            prev_delta = model(previous)[1]
            predicted = supervised_torque(gains,q_error,v,qd,g)
            torque_loss = ((predicted-target)/scale).square().mean()
            deviation = delta.square().mean()
            smooth = (delta-prev_delta).square().mean()
            loss = torque_loss+.02*deviation+.02*smooth
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite supervised gain loss')
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
            totals += len(hidden)*np.array([float(x.detach()) for x in (loss,torque_loss,deviation,smooth)])
            count += len(hidden)
    return dict(zip(('loss','normalized_torque_mse','gain_deviation','temporal_variation'),(totals/count).tolist()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--smoke-test',action='store_true')
    args = p.parse_args()
    check_baseline()
    run = ROOT/'runs'/('smoke' if args.smoke_test else 'full')
    if run.exists():
        raise FileExistsError('Never overwrite a trained candidate')
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(2)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    arrays,names = {},{}
    for split in ('train','val'):
        arrays[split],names[split] = prepare(split,args.smoke_test,device)
    if set(names['train']) & set(names['val']):
        raise ValueError('Train/validation trajectory overlap')
    h = arrays['train']['hidden']
    mean = h.mean(axis=0,dtype=np.float64).astype(np.float32)
    std = np.maximum(h.std(axis=0,dtype=np.float64),.1).astype(np.float32)
    model = GainNetwork(torch.from_numpy(mean),torch.from_numpy(std)).to(device)
    with np.load(BASE/'models/tcn/normalization.npz') as stored:
        scale = torch.from_numpy(stored['torque_std'].copy()).to(device)
    weight = arrays['train']['weight'].copy()
    low = arrays['train']['low']
    for value in (False,True):
        mask = low==value
        weight[mask] *= .5/weight[mask].sum()
    sampler = WeightedRandomSampler(torch.from_numpy(weight),
        min(2048 if args.smoke_test else 200000,len(h)),True,
        generator=torch.Generator().manual_seed(42))
    loaders = dict(train=DataLoader(Samples(arrays['train']),batch_size=4096,sampler=sampler),
                   val=DataLoader(Samples(arrays['val']),batch_size=4096))
    epochs = 2 if args.smoke_test else 20
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs,eta_min=1e-4)
    run.mkdir(parents=True)
    config = dict(schema_version=1,mode='smoke' if args.smoke_test else 'full',seed=42,
        trained_parameters=sum(v.numel() for v in model.parameters()),
        initialized_p=100.,initialized_d=30.,p_bounds=[80.,120.],d_bounds=[24.,36.],
        maximum_epochs=epochs,batch_size=4096,samples_per_epoch=sampler.num_samples,
        torque_supervision='recorded expert actions; no fabricated variable gain labels',
        offline_qref='recorded expert q_des_arm for offline action loss only',
        online_qref='policy integrated velocity, never recorded reference',
        motion_and_navigation_frozen=True,normalization_fit_on='train only',
        final_test_used=False,device=str(device),
        regularization=dict(deviation=.02,temporal=.02),
        gain_filter_time_constant_s=.010,
        source_sha256={str(q.relative_to(ROOT)):sha(q) for q in (Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('dataset.py'),Path(__file__).with_name('controller.py'))})
    write(run/'config.json',config)
    write(run/'used_trajectories.json',dict(**names,test=[],final_test=[]))
    initial = run_epoch(model,loaders['val'],scale,device)
    write(run/'initial_validation.json',initial)
    print(f'INITIAL val: {initial}; device={device}; trainable={config["trained_parameters"]}',flush=True)
    best = float('inf')
    best_epoch = 0
    stale = 0

    def save(number,metrics):
        torch.save(dict(schema_version=1,epoch=number,config=config,validation=metrics,
                        base_checkpoint_sha256=sha(BASE/'models/tcn/best_model.pt'),
                        gain_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),run/'best_model.pt')

    with (run/'history.csv').open('w',newline='') as stream:
        fields = ['epoch','seconds']+[f'{s}_{k}' for s in ('train','val') for k in initial]
        writer = csv.DictWriter(stream,fields)
        writer.writeheader()
        for epoch in range(1,epochs+1):
            start = time.perf_counter()
            train = run_epoch(model,loaders['train'],scale,device,optimizer)
            val = run_epoch(model,loaders['val'],scale,device)
            scheduler.step()
            writer.writerow(dict(epoch=epoch,seconds=time.perf_counter()-start,
                **{f'train_{k}':v for k,v in train.items()},**{f'val_{k}':v for k,v in val.items()}))
            stream.flush()
            print(f'Epoch {epoch:02d}: train={train["loss"]:.7f}; val={val["loss"]:.7f}; torque={val["normalized_torque_mse"]:.7f}; deviation={val["gain_deviation"]:.6f}',flush=True)
            if val['loss'] < best:
                best,best_epoch,stale = val['loss'],epoch,0
                save(epoch,val)
            else:
                stale += 1
            if stale>=6:
                break
    check_baseline()
    saved = torch.load(run/'best_model.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(saved['gain_state_dict'])
    model.eval()
    with torch.inference_mode():
        gains = model(torch.from_numpy(arrays['val']['hidden'][:min(20000,len(arrays['val']['hidden']))]).to(device))[0].cpu().numpy()
    write(run/'training_finished.json',dict(state='completed',best_epoch=best_epoch,epochs_completed=epoch,
        initial_validation=initial,best_validation=saved['validation'],
        validation_gain_min=gains.min(axis=0).tolist(),validation_gain_max=gains.max(axis=0).tolist(),
        validation_gain_std=gains.std(axis=0).tolist(),checkpoint_sha256=sha(run/'best_model.pt'),
        final_test_loaded=False,baseline_unchanged=True,subset_only=args.smoke_test))
    print('TRAINING COMPLETE; closed-loop validation and independent test still required.',flush=True)


if __name__ == '__main__':
    main()
