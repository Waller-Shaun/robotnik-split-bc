# [Residual120 change] 独立训练副本：冻结120版，只训练新修正分支；不续训旧残差分支。
"""[Residual change] 只训练额外力矩修正网络；冻结原baseline，训练/测试独立。

沿用原恢复示范的26维输入、102条纠偏数据及80/20来源抽样，不重采。
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from finetune_arm_recovery import load_original, draw_indices, update_early_stopping
from train_arm_bc_ring_aux_b2 import check_headers
from recovery_dataset import FROZEN, read_json, write_json, sha256
from recovery_residual120_context import BASELINE, resolve_parent_pair, load_recovery
from robotnik_arm_residual120_policy import (VARIANT,RUN_ROOT,build_from_parent,load_policy_from_checkpoint,
    assert_base_unchanged)


class CachedStore:
    def __init__(self,x,y,low,model,device,batch_size):
        self.x=torch.as_tensor(x,device=device)
        self.y=torch.as_tensor(y,device=device)
        self.low=torch.as_tensor(low,device=device)
        # [Residual speed] 基础策略完全冻结，标准化输入也固定；输出只计算一次，训练不反复跑基础MLP。
        self.base=torch.empty_like(self.y)
        with torch.inference_mode():
            for start in range(0,len(x),batch_size):
                self.base[start:start+batch_size]=model.base(self.x[start:start+batch_size])[:,4:11]


def run_epoch(model,data,indices,batch_size,penalty,optimizer=None):
    model.train(optimizer is not None)
    device=data.x.device
    sums=torch.zeros(2,dtype=torch.float64,device=device)
    counts=torch.zeros(2,dtype=torch.int64,device=device)
    correction_sq=torch.zeros((),dtype=torch.float64,device=device)
    correction_nm_sq=torch.zeros_like(correction_sq);near_bound=torch.zeros_like(correction_sq)
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0,len(indices),batch_size):
            ix=indices[start:start+batch_size]
            delta_nm=model.correction_nm(data.x[ix]);delta=delta_nm/model.arm_std
            # 直接对组合策略匹配专家力矩，等价于学习归一化的expert-base残差。
            errors=(data.base[ix]+delta-data.y[ix]).square().mean(1)
            reg=delta.square().mean(1)
            loss=errors.mean()+penalty*reg.mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite residual loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            masks=torch.stack([data.low[ix],~data.low[ix]])
            sums+=masks.double()@errors.detach().double();counts+=masks.sum(1)
            correction_sq+=reg.detach().double().sum()
            correction_nm_sq+=delta_nm.detach().double().square().mean(1).sum()
            near_bound+=(delta_nm.detach().abs()>=.95*model.limit_nm).double().mean(1).sum()
    s,c=sums.cpu().numpy(),counts.cpu().numpy();n=int(c.sum())
    if n==0:raise ValueError('Empty epoch')
    groups=[float(s[i]/c[i]) if c[i] else None for i in range(2)]
    mse=float(s.sum()/n);reg=float(correction_sq.cpu())/n
    return {'arm_mse':mse,'low_arm_mse':groups[0],'other_arm_mse':groups[1],
        'balanced_arm_mse':float(np.mean(groups)) if all(v is not None for v in groups) else None,
        'total_loss':mse+penalty*reg,'correction_rms_nm':float(torch.sqrt(correction_nm_sq/n).cpu()),
        'correction_near_bound_fraction':float(near_bound.cpu())/n}


def main():
    root=Path(__file__).resolve().parents[3]/'RoboEnv'
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-arm-run-dir',type=Path,default=root/BASELINE)
    p.add_argument('--recovery-manifest',type=Path,default=root/'expert_recovery_data/collection_20261104_112547_770894/episodes/recovery_manifest.json')
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--min-epochs',type=int,default=50)
    p.add_argument('--patience',type=int,default=15)
    p.add_argument('--batch-size',type=int,default=4096)
    p.add_argument('--samples-per-epoch',type=int,default=500_000)
    p.add_argument('--learning-rate',type=float,default=1e-4,help='New residual branch starts from scratch; base frozen')
    p.add_argument('--recovery-fraction',type=float,default=.2)
    p.add_argument('--residual-limit-nm',type=float,default=5.)
    p.add_argument('--residual-penalty',type=float,default=1e-4,help='Penalty on normalized correction magnitude; 0 disables')
    p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto')
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--max-train-files',type=int)
    p.add_argument('--max-val-files',type=int)
    p.add_argument('--check-data-only',action='store_true')
    p.add_argument('--resume-run-dir',type=Path,help='Resume last checkpoint; epochs is cumulative upper bound')
    a=p.parse_args()
    if (min(a.epochs,a.min_epochs,a.patience,a.batch_size,a.samples_per_epoch)<1 or a.min_epochs>a.epochs
            or not 0<=a.recovery_fraction<1 or not np.isfinite(a.learning_rate) or a.learning_rate<=0
            or not np.isfinite(a.residual_limit_nm) or not 0<a.residual_limit_nm<=30
            or not np.isfinite(a.residual_penalty) or a.residual_penalty<0
            or any(v is not None and v<4 for v in (a.max_train_files,a.max_val_files))):p.error('Invalid settings')
    parent,nav,pc,nc=resolve_parent_pair(root,a.parent_arm_run_dir)
    if pc['aux_weight']!=0:raise ValueError('Expected torque-only120/144 recovery parent')
    split=read_json(parent/'data_split.json')
    with np.load(parent/'normalization.npz') as d:stats={k:d[k].copy() for k in ('x_mean','x_std','y_mean','y_std')}
    rx,ry,rl,recovery=load_recovery(a.recovery_manifest,parent,split,stats)
    if a.check_data_only:
        sizes={s:sum(check_headers(split,s,limit)[1]) for s,limit in [('train',a.max_train_files),('val',a.max_val_files)]}
        print('Original frames:',sizes,'; recovery frames:',len(rx),'; training started: False');return
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    random.seed(42);np.random.seed(42);torch.manual_seed(42)
    device=torch.device('cuda' if a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cuda':torch.cuda.manual_seed_all(42)
    else:torch.set_num_threads(min(4,torch.get_num_threads()))
    print('Device:',device,'; GPU:',torch.cuda.get_device_name(0) if device.type=='cuda' else 'none',flush=True)
    parent_cp=torch.load(parent/'best_model.pt',map_location='cpu',weights_only=True)
    if parent_cp['config']!=pc:raise ValueError('Parent checkpoint/config mismatch')
    model=build_from_parent(parent_cp,stats,residual_limit_nm=a.residual_limit_nm)
    smoke=a.max_train_files is not None or a.max_val_files is not None
    resumed=None;start_epoch=0;best=float('inf');best_epoch=0;stale=0
    if a.resume_run_dir:
        source=a.resume_run_dir.resolve();rc=read_json(source/'config.json');done=read_json(source/'training_finished.json')
        if rc.get('experiment_variant')!=VARIANT:raise ValueError('Not a residual run')
        for field,value in [('batch_size',a.batch_size),('samples_per_epoch',a.samples_per_epoch),
                ('learning_rate',a.learning_rate),('recovery_fraction',a.recovery_fraction),
                ('residual_limit_nm',a.residual_limit_nm),('residual_penalty',a.residual_penalty),('smoke_only',smoke)]:
            if rc[field]!=value:raise ValueError(f'Resume must keep {field}={rc[field]}')
        if (Path(rc['source_model_run_dir']).resolve()!=parent
                or rc['recovery_manifest_sha256']!=sha256(a.recovery_manifest)):
            raise ValueError('Resume parent/recovery data changed')
        for path,h in read_json(source/'training_source_hashes.json').items():
            if sha256(path)!=h:raise ValueError(f'Resume source changed: {path}')
        cp=torch.load(source/'last_model.pt',map_location='cpu',weights_only=True)
        previous_best=torch.load(source/'best_model.pt',map_location='cpu',weights_only=True)
        if cp['config']!=rc or previous_best['config']!=rc or cp['epoch']!=done['last_epoch']:
            raise ValueError('Resume checkpoints/config mismatch')
        start_epoch=cp['epoch']
        if a.epochs<=start_epoch:raise ValueError('epochs must exceed saved last epoch')
        model=load_policy_from_checkpoint(cp)
        stop=cp['early_stopping_state'];best,best_epoch,stale=stop['best'],stop['best_epoch'],stop['stale']
        if previous_best['epoch']!=best_epoch:raise ValueError('Resume best epoch mismatch')
        resumed=(source,rc,cp,previous_best)
    assert_base_unchanged(model,parent_cp);model.to(device)
    x,y,low=load_original(split,'train',stats,a.max_train_files)
    original_count=len(x);nrecovery=len(rx)
    low_ix,other_ix=torch.from_numpy(np.flatnonzero(low)),torch.from_numpy(np.flatnonzero(~low))
    train=CachedStore(np.concatenate([x,rx]),np.concatenate([y,ry]),np.concatenate([low,rl]),model,device,a.batch_size)
    del x,y,low,rx,ry,rl
    xv,yv,lv=load_original(split,'val',stats,a.max_val_files)
    val_count=len(xv);val=CachedStore(xv,yv,lv,model,device,a.batch_size);del xv,yv,lv
    if resumed:
        for field,count in [('original_train_frames',original_count),('recovery_train_frames',nrecovery),('val_samples',val_count)]:
            if resumed[1][field]!=count:raise ValueError('Resume frame counts changed')
    generator=torch.Generator().manual_seed(42)
    optimizer=torch.optim.Adam(model.residual.parameters(),lr=a.learning_rate)
    if resumed:
        optimizer.load_state_dict(resumed[2]['optimizer_state_dict'])
        generator.set_state(resumed[2]['sampling_generator_state'].cpu())
        torch.set_rng_state(resumed[2]['torch_rng_state'].cpu())
        if device.type=='cuda':torch.cuda.set_rng_state_all([v.cpu() for v in resumed[2]['cuda_rng_state_all']])
    run=(a.output_dir or root/RUN_ROOT/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')).resolve()
    run.mkdir(parents=True,exist_ok=False)
    for name in ('normalization.npz','data_split.json'):shutil.copyfile(parent/name,run/name)
    shutil.copyfile(a.recovery_manifest,run/'recovery_manifest.json')
    cfg=dict(pc)
    cfg.update({'experiment_variant':VARIANT,'training_mode':'frozen_base_arm_residual_navigation_frozen',
        'policy_architecture':'frozen_MLP26_256_256_128_action11_plus_MLP26_64_64_arm7',
        'source_model_run_dir':str(parent),'source_model_file_sha256':{n:sha256(parent/n) for n in FROZEN},
        'source_model_epoch':int(parent_cp['epoch']),
        # [Residual120 change] 冻结模型来自120版，训练数据生成策略仍是113版，分别保存来源。
        'frozen_base_validation_reference':'120/144_noisy_val_best_epoch138',
        'recovery_collection_policy_run_dir':recovery['parent_arm_run'],
        'recovery_collection_policy_sha256':recovery['parent_sha256'],
        'residual_targets_recomputed_against_current_base':True,
        'previous113_residual_weights_loaded':False,
        'nav_run_dir':str(nav),'frozen_nav_file_sha256':{n:sha256(nav/n) for n in FROZEN},
        'base_parameters_frozen':True,'navigation_weights_updated':False,'aux_weight':0.,
        'auxiliary_output_applied_to_robot':False,'expert_references_used_as_input':False,'gru_used':False,
        'residual_hidden_dims':[64,64],'residual_output_dim':7,'residual_limit_nm':a.residual_limit_nm,
        'residual_penalty':a.residual_penalty,'residual_initialization':'zero_last_layer_exact_parent_output',
        'residual_units':'N*m_before_original_total_torque_clip','residual_gate':'none_always_active',
        'arm_action_std':stats['y_std'][4:11].tolist(),'inference_total_action_clip_nm':30.,
        'recovery_manifest_path':str(a.recovery_manifest.resolve()),'recovery_manifest_sha256':sha256(a.recovery_manifest),
        'original_train_frames':original_count,'recovery_train_frames':nrecovery,'val_samples':val_count,
        'recovery_fraction':a.recovery_fraction,'sampling_method':'original_low_other_equal_plus_fixed_recovery_fraction',
        'learning_rate':a.learning_rate,'batch_size':a.batch_size,'samples_per_epoch':a.samples_per_epoch,
        'max_epochs':a.epochs,'min_epochs':a.min_epochs,'patience':a.patience,'seed':42,'device':str(device),
        'loss_type':'combined_normalized_arm_MSE_plus_small_normalized_residual_penalty',
        'selection_metric':'original_val_balanced_arm_mse_post_training_epochs',
        'resumed':resumed is not None,'resume_from_epoch':start_epoch,
        'resume_from_run_dir':str(resumed[0]) if resumed else None,
        'optimizer_initialization':'restored_Adam' if resumed else 'new_Adam_residual_only',
        'trainable_parameter_count':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'parameter_count':sum(p.numel() for p in model.parameters()),
        'train_samples':original_count+nrecovery,
        'frozen_parameter_count':sum(p.numel() for p in model.base.parameters()),'smoke_only':smoke,
        'inference_module_sha256':{n:sha256(Path(__file__).with_name(n)) for n in [
            'robotnik_arm_residual120_policy.py','recovery_residual120_context.py','robotnik_arm_aux_policy.py','robotnik_bodyframe_features.py',
            'robotnik_navigation_policy.py','robotnik_arm_control.py']}})
    write_json(run/'config.json',cfg);shutil.copyfile(__file__,run/'training_source.py')
    frozen={str(folder/n):sha256(folder/n) for folder in (parent,nav) for n in FROZEN}
    collection_policy=Path(recovery['parent_arm_run'])
    frozen.update({str(collection_policy/n):sha256(collection_policy/n) for n in FROZEN})
    frozen[str(a.recovery_manifest.resolve())]=sha256(a.recovery_manifest)
    frozen.update({str(a.recovery_manifest.resolve().parent/e['file']):e['sha256'] for e in recovery['entries']})
    frozen.update({str(Path(__file__).with_name(n)):h for n,h in cfg['inference_module_sha256'].items()})
    # 复用的数据读取、采样、早停函数同样是训练依赖，记录哈希以便复核和续训。
    for n in ('finetune_arm_recovery.py','recovery_dataset.py','train_arm_bc_ring_aux_b2.py'):
        frozen[str(Path(__file__).with_name(n))]=sha256(Path(__file__).with_name(n))
    if resumed:
        for name in (*FROZEN,'last_model.pt','baseline_model.pt','training_finished.json','history.csv'):
            frozen[str(resumed[0]/name)]=sha256(resumed[0]/name)
        shutil.copyfile(resumed[0]/'history.csv',run/'previous_history.csv')
    write_json(run/'training_source_hashes.json',frozen)
    def verify():
        if any(sha256(p)!=h for p,h in frozen.items()):raise RuntimeError('Frozen source changed')
        assert_base_unchanged(model,parent_cp)
    vi=torch.arange(val_count,device=device);ri=torch.arange(original_count,original_count+nrecovery,device=device)
    initial=run_epoch(model,val,vi,a.batch_size,a.residual_penalty)
    initial_rec=run_epoch(model,train,ri,a.batch_size,a.residual_penalty)
    write_json(run/'initial_validation.json',initial);write_json(run/'initial_recovery_train.json',initial_rec)
    baseline_score=initial['balanced_arm_mse'] if not resumed else read_json(resumed[0]/'training_finished.json')['baseline_balanced_validation_arm_mse']
    def save(name,epoch,metrics):
        torch.save({'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'optimizer_state_dict':optimizer.state_dict(),'sampling_generator_state':generator.get_state(),
            'torch_rng_state':torch.get_rng_state(),'cuda_rng_state_all':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            'early_stopping_state':{'best':best if np.isfinite(best) else None,'best_epoch':best_epoch,'stale':stale},
            'epoch':epoch,'val_mse':metrics['arm_mse'],'val_loss':metrics['balanced_arm_mse'],
            'val_metrics':metrics,'config':cfg},run/name)
    if not resumed:save('baseline_model.pt',0,initial)
    else:
        for name,cp in [('best_model.pt',resumed[3]),('baseline_model.pt',torch.load(resumed[0]/'baseline_model.pt',map_location='cpu',weights_only=True))]:
            inherited=dict(cp);inherited['config']=cfg;torch.save(inherited,run/name)
    print('Output:',run,'; trainable residual parameters:',cfg['trainable_parameter_count'],flush=True)
    print(f'Frozen baseline val={baseline_score:.6f}; initial recovery TRAIN={initial_rec["arm_mse"]:.6f}; max={a.epochs}, min={a.min_epochs}, patience={a.patience}',flush=True)
    # [Residual diagnostic] 原恢复标签所需修正是否超过当前5Nm设计范围，只作可表达性诊断。
    with torch.no_grad():
        needs=(train.y[ri]-train.base[ri])*model.arm_std
        beyond=float((needs.abs()>model.limit_nm).float().mean().cpu())
    write_json(run/'correction_bound_diagnostic.json',{'recovery_target_coordinates_exceeding_bound_fraction':beyond,'limit_nm':a.residual_limit_nm})
    print(f'Recovery target residual coordinates beyond bound: {100*beyond:.2f}%',flush=True)
    keys=('arm_mse','low_arm_mse','other_arm_mse','balanced_arm_mse','total_loss','correction_rms_nm','correction_near_bound_fraction')
    with (run/'history.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f);writer.writerow(['epoch','seconds',*[s+'_'+k for s in ('train','val') for k in keys],
            'recovery_train_mse','best_epoch','best_balanced_val','stale','delta_baseline'])
        for epoch in range(start_epoch+1,a.epochs+1):
            verify();start=time.perf_counter()
            ix=draw_indices(low_ix,other_ix,original_count,nrecovery,a.samples_per_epoch,a.recovery_fraction,generator,device)
            tr=run_epoch(model,train,ix,a.batch_size,a.residual_penalty,optimizer)
            va=run_epoch(model,val,vi,a.batch_size,a.residual_penalty)
            rec=run_epoch(model,train,ri,a.batch_size,a.residual_penalty)
            best,best_epoch,stale,improved,stop=update_early_stopping(va['balanced_arm_mse'],epoch,best,best_epoch,stale,a.min_epochs,a.patience)
            seconds=time.perf_counter()-start;delta=va['balanced_arm_mse']-baseline_score
            writer.writerow([epoch,seconds,*[d[k] for d in (tr,va) for k in keys],rec['arm_mse'],best_epoch,best,stale,delta]);f.flush()
            print(f"Epoch {epoch:03d} | train {tr['arm_mse']:.6f} | val {va['arm_mse']:.6f} | low {va['low_arm_mse']:.6f} | other {va['other_arm_mse']:.6f} | balanced {va['balanced_arm_mse']:.6f} | recovery TRAIN {rec['arm_mse']:.6f} | correction RMS {va['correction_rms_nm']:.3f}Nm | near bound {100*va['correction_near_bound_fraction']:.2f}% | best {best_epoch} | stale {stale}/{a.patience} | delta baseline {delta:+.6f} | stop {'enabled' if epoch>=a.min_epochs else 'locked'} | {seconds:.2f}s",flush=True)
            if improved:save('best_model.pt',epoch,va)
            save('last_model.pt',epoch,va)
            if stop:print('Early stopping: no new post-training balanced validation best after minimum epochs.');break
    verify()
    write_json(run/'training_finished.json',{'best_epoch':best_epoch,'last_epoch':epoch,
        'best_balanced_validation_arm_mse':best,'baseline_balanced_validation_arm_mse':baseline_score,
        'best_minus_baseline_mse':best-baseline_score,'base_parameters_unchanged':True,'navigation_unchanged':True,
        'smoke_only':smoke,'resumed':resumed is not None,'resume_from_epoch':start_epoch})
    print(f'Best residual epoch: {best_epoch}; balanced validation arm MSE: {best:.6f}\nResults: {run}')
    print('Training only. Run batch_test_arm_residual120_parallel.py separately; no expert at inference.')


if __name__=='__main__':main()
