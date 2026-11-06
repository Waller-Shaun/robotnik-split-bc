"""[CNN-T change] 独立监督训练：历史CNN＋Transformer直接预测7维力矩。

原698个训练目标＋已有102条恢复片段；底盘冻结，机械臂网络从零开始。
不采集新数据，不启动仿真或完整测试；训练与测试分开。
"""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from recovery_dataset import FROZEN, sha256, read_json, write_json
from recovery_residual120_context import resolve_parent_pair
from robotnik_bodyframe_features import BODYFRAME_FEATURE_VERSION
from robotnik_arm_cnn_transformer_policy import HistoryArmPolicy, VARIANT, HISTORY_VERSION, RUN_ROOT, CNN_RUN_ROOT
from robotnik_arm_cnn_transformer_data import (original_arrays,recovery_arrays,WindowStore,
                                             recovery_eligible_indices,validation_indices)

METRICS=('arm_mse','low_arm_mse','other_arm_mse','balanced_arm_mse')


def epoch_pass(model,store,indices,batch_size,optimizer=None,amp=False):
    training=optimizer is not None;model.train(training)
    sums=torch.zeros(2,dtype=torch.float64,device=store.x.device)
    counts=torch.zeros(2,dtype=torch.int64,device=store.x.device)
    with torch.set_grad_enabled(training):
        for offset in range(0,len(indices),batch_size):
            x,y,low=store.batch(indices[offset:offset+batch_size])
            # [CNN-T speed] CUDA训练可用bf16；验证与正式推理均为float32。
            with torch.autocast(device_type=store.x.device.type,dtype=torch.bfloat16,enabled=training and amp):
                prediction=model(x)
            error=(prediction.float()-y).square().mean(1);loss=error.mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite torque MSE')
            if training:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
            masks=torch.stack([low,~low])
            sums+=masks.double()@error.detach().double();counts+=masks.sum(1)
    s,c=sums.cpu().numpy(),counts.cpu().numpy()
    if (c==0).any():raise ValueError('Both validation/training height groups are required')
    groups=s/c
    return dict(arm_mse=float(s.sum()/c.sum()),low_arm_mse=float(groups[0]),
                other_arm_mse=float(groups[1]),balanced_arm_mse=float(groups.mean()))


def sample_endpoints(low_indices,other_indices,recovery_indices,count,fraction,generator,device):
    nr=int(round(count*fraction));no=count-nr;nl=no//2;nh=no-nl
    chunks=[low_indices[torch.randint(len(low_indices),(nl,),generator=generator)],
            other_indices[torch.randint(len(other_indices),(nh,),generator=generator)]]
    if nr:chunks.append(recovery_indices[torch.randint(len(recovery_indices),(nr,),generator=generator)])
    ix=torch.cat(chunks)
    return ix[torch.randperm(count,generator=generator)].to(device)


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roboenv',type=Path,default=Path(__file__).resolve().parents[3]/'RoboEnv')
    p.add_argument('--architecture',choices=['cnn_transformer','cnn'],default='cnn_transformer',help='cnn: separate no-attention ablation')
    p.add_argument('--epochs',type=int,default=200)
    p.add_argument('--min-epochs',type=int,default=20)
    p.add_argument('--patience',type=int,default=15)
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--samples-per-epoch',type=int,default=500000)
    p.add_argument('--learning-rate',type=float,default=1e-4)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--history-length',type=int,default=16)
    p.add_argument('--history-stride-steps',type=int,default=10)
    p.add_argument('--validation-stride',type=int,default=10,help='Fixed endpoints every10ms, plus each episode first/last; 1=all frames')
    p.add_argument('--recovery-fraction',type=float,default=.2)
    p.add_argument('--dropout',type=float,default=.1)
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--no-amp',action='store_true',help='Disable CUDA bf16 training')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--max-train-files',type=int,help='Program check only; >=4')
    p.add_argument('--max-val-files',type=int,help='Program check only; >=4')
    p.add_argument('--max-recovery-files',type=int,help='Program check only; >=1')
    p.add_argument('--plan-only',action='store_true',help='Verify context and show architecture, no dataset loading/training/output')
    a=p.parse_args()
    for n in ('epochs','min_epochs','patience','batch_size','samples_per_epoch','history_length','history_stride_steps','validation_stride'):
        if getattr(a,n)<1:p.error(n+' must be positive')
    if a.min_epochs>a.epochs:p.error('min-epochs must not exceed epochs')
    if a.seed<0 or not 0<a.recovery_fraction<1 or not 0<=a.dropout<1:p.error('Invalid seed/fraction/dropout')
    if not math.isfinite(a.learning_rate) or a.learning_rate<=0 or not math.isfinite(a.weight_decay) or a.weight_decay<0:p.error('Invalid optimizer settings')
    if a.samples_per_epoch<10:p.error('samples-per-epoch must be >=10')
    for n in ('max_train_files','max_val_files'):
        if getattr(a,n) is not None and getattr(a,n)<4:p.error(n+' must be >=4')
    if a.max_recovery_files is not None and a.max_recovery_files<1:p.error('max-recovery-files must be >=1')
    if a.history_length>128:p.error('This small experiment supports history length <=128')
    return a


def main():
    a=arguments();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device=torch.device('cuda' if a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available()) else 'cpu')
    torch.set_num_threads(1 if device.type=='cuda' else min(4,torch.get_num_threads()))
    parent,nav,pc,nc=resolve_parent_pair(a.roboenv)
    split=read_json(parent/'data_split.json')
    sets={k:{tuple(np.round(t,6)) for t in split['targets'][k]} for k in ('train','val','test')}
    if any(sets[u]&sets[v] for u,v in (('train','val'),('train','test'),('val','test'))):raise ValueError('Target split leakage')
    if {k:len(v) for k,v in sets.items()}!={'train':698,'val':144,'test':156}:raise ValueError('Wrong 998-target split')
    with np.load(parent/'normalization.npz',allow_pickle=False) as n:stats={k:n[k].copy() for k in ('x_mean','x_std','y_mean','y_std')}
    if (stats['x_mean'].shape!=(26,) or stats['y_mean'].shape!=(11,) or stats['x_std'].shape!=(26,)
            or stats['y_std'].shape!=(11,) or not all(np.isfinite(v).all() for v in stats.values())
            or np.any(stats['x_std']<=0) or np.any(stats['y_std']<=0)
            or stats['x_mean'][22]!=0 or stats['x_std'][22]!=1):raise ValueError('Invalid inherited normalization')
    recovery_path=Path(pc['recovery_data_manifest_path'])
    if sha256(recovery_path)!=pc['recovery_data_manifest_sha256']:raise ValueError('Recovery manifest changed')
    model=HistoryArmPolicy(a.history_length,a.architecture,a.dropout).to(device)
    params=sum(p.numel() for p in model.parameters())
    amp=device.type=='cuda' and torch.cuda.is_bf16_supported() and not a.no_amp
    print(f'Architecture={a.architecture}; device={device}; train bf16 AMP={amp}; parameters={params:,}',flush=True)
    print(f'History={a.history_length}x26; stride={a.history_stride_steps}ms; span={(a.history_length-1)*a.history_stride_steps}ms; output=7 current arm torques.',flush=True)
    print('Fresh arm weights; NO old MLP/residual/PD/IK. Original learned navigation is frozen.',flush=True)
    print(f'Data: same698 train targets +102 existing recovery clips; val144 fixed targets; endpoint stride={a.validation_stride}ms.',flush=True)
    if a.plan_only:return

    print('Loading original TRAIN trajectories...',flush=True)
    original=original_arrays(split,'train',stats,a.max_train_files)
    original_count=len(original[0]);original_lengths=original[3].copy()
    low_ix=torch.from_numpy(np.flatnonzero(original[2]));other_ix=torch.from_numpy(np.flatnonzero(~original[2]))
    rx,ry,rl,rlengths,recovery_manifest=recovery_arrays(recovery_path,parent,split,stats,a.max_recovery_files)
    recovery_count=len(rx)
    span=(a.history_length-1)*a.history_stride_steps
    rec_ix=recovery_eligible_indices(rlengths,original_count,span)
    train=WindowStore([original,(rx,ry,rl,rlengths)],device,a.history_length,a.history_stride_steps)
    del original,rx,ry,rl
    print('Loading original VALIDATION trajectories (no recovery labels)...',flush=True)
    validation=original_arrays(split,'val',stats,a.max_val_files)
    val_lengths=validation[3].copy();val_count=len(validation[0])
    val=WindowStore([validation],device,a.history_length,a.history_stride_steps)
    vi=torch.from_numpy(validation_indices(val_lengths,a.validation_stride)).to(device)
    del validation
    samples_per_epoch=min(a.samples_per_epoch,original_count+len(rec_ix))
    smoke=any(v is not None for v in (a.max_train_files,a.max_val_files,a.max_recovery_files))
    run_root=RUN_ROOT if a.architecture=='cnn_transformer' else CNN_RUN_ROOT
    run=(a.output_dir or a.roboenv/run_root/f'run_{datetime.now():%Y%m%d_%H%M%S_%f}').resolve()
    run.mkdir(parents=True,exist_ok=False)
    modules=('robotnik_arm_cnn_transformer_policy.py','robotnik_bodyframe_features.py','robotnik_navigation_policy.py',
             'robotnik_arm_control.py','recovery_residual120_context.py','robotnik_arm_aux_policy.py')
    c=dict(experiment_variant=VARIANT,architecture=a.architecture,input_dim=26,source_input_dim=23,output_dim=7,
        output_action_columns=list(range(4,11)),input_feature_version=BODYFRAME_FEATURE_VERSION,history_feature_version=HISTORY_VERSION,
        history_length=a.history_length,history_stride_steps=a.history_stride_steps,history_span_seconds=span*.001,
        history_padding='repeat_episode_first',history_hidden_state_carried=False,history_reset='each episode/clip',
        sample_dt_seconds=.001,cnn_channels=[26,64,128],cnn_kernel_size=3,cnn_stride=1,cnn_padding='left_only',
        embedding_dim=128,attention_heads=4,transformer_layers=2 if a.architecture=='cnn_transformer' else 0,
        feedforward_dim=256,dropout=a.dropout,causal_attention=True,current_last_token_readout=a.architecture=='cnn_transformer',
        parameter_count=params,pretrained_weights_loaded=False,baseline_residual_branch=False,arm_finetuned=False,
        expert_references_used_as_input=False,aux_weight=0.,auxiliary_reference_loss=False,
        navigation_weights_updated=False,source_model_run_dir=str(parent),source_model_context_use='normalization_split_recovery_provenance_only_no_weights',
        source_model_file_sha256={n:sha256(parent/n) for n in FROZEN},nav_run_dir=str(nav),
        frozen_nav_file_sha256={n:sha256(nav/n) for n in FROZEN},dataset_variant=pc['dataset_variant'],
        target_counts=split['target_counts'],data_dir=split['data_dir'],base_active_column=22,
        target_position_frame='base_yaw_aligned',ee_error_frame='base_yaw_aligned',
        normalization_distribution='same_frozen_train_statistics_as_recovery120',
        original_train_frames=original_count,recovery_train_frames=recovery_count,recovery_eligible_endpoints=len(rec_ix),
        recovery_clip_initial_context_only_frames=recovery_count-len(rec_ix),recovery_fraction=a.recovery_fraction,
        recovery_data_manifest_path=str(recovery_path),recovery_data_manifest_sha256=sha256(recovery_path),
        original_train_trajectories=len(original_lengths),recovery_trajectories=len(rlengths),
        val_trajectories=len(val_lengths),val_frames=val_count,validation_endpoints=len(vi),validation_stride_steps=a.validation_stride,
        validation_all_frames=a.validation_stride==1,validation_endpoint_rule='per_episode_every_stride_plus_first_and_last',
        validation_metric='0.5*low_z_le_.35_MSE+0.5*other_MSE_on_fixed_validation_endpoints',
        validation_original_only=True,validation_recovery_included=False,sampling_original_low_other_fraction=.5,
        samples_per_epoch=samples_per_epoch,batch_size=a.batch_size,loss_type='MSE_normalized_current_arm_torques',
        optimizer='AdamW',learning_rate=a.learning_rate,weight_decay=a.weight_decay,gradient_clip_norm=1.,
        lr_scheduler='ReduceLROnPlateau_factor0.5_patience5_min1e-6',max_epochs=a.epochs,min_epochs=a.min_epochs,
        patience=a.patience,early_stop_on='post_training_best_balanced_validation_MSE',
        seed=a.seed,device=str(device),torch_version=str(torch.__version__),amp_train_bfloat16=amp,
        validation_dtype='float32',inference_dtype='float32',smoke_only=smoke,
        data_storage='single_frames_on_device_causal_windows_gathered_lazily',
        inference_module_sha256={n:sha256(Path(__file__).with_name(n)) for n in modules},
        training_source_sha256=sha256(Path(__file__)),data_module_sha256=sha256(Path(__file__).with_name('robotnik_arm_cnn_transformer_data.py')))
    write_json(run/'config.json',c)
    for n in ('normalization.npz','data_split.json'):shutil.copy2(parent/n,run/n)
    shutil.copy2(recovery_path,run/'recovery_manifest.json')
    for n,out_name in [('train_arm_bc_cnn_transformer.py','training_source.py'),
                       ('robotnik_arm_cnn_transformer_policy.py','policy_source.py'),
                       ('robotnik_arm_cnn_transformer_data.py','data_source.py')]:shutil.copy2(Path(__file__).with_name(n),run/out_name)
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay,fused=device.type=='cuda')
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5,min_lr=1e-6)
    generator=torch.Generator().manual_seed(a.seed)
    best=math.inf;best_epoch=0;stale=0

    def checkpoint(name,epoch,metrics):
        value=dict(model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},config=c,
            epoch=epoch,val_mse=metrics['arm_mse'],val_loss=metrics['balanced_arm_mse'],val_metrics=metrics,
            optimizer_state_dict=optimizer.state_dict(),scheduler_state_dict=scheduler.state_dict(),
            sampling_generator_state=generator.get_state(),torch_rng_state=torch.get_rng_state(),
            cuda_rng_state_all=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            best_epoch=best_epoch,best_balanced_validation_mse=best,stale=stale)
        temp=run/(name+'.tmp');torch.save(value,temp);temp.replace(run/name)

    print(f'Output: {run}',flush=True)
    print(f'Train frames={train.n:,}; eligible recovery endpoints={len(rec_ix):,}; val endpoints={len(vi):,}/{val_count:,}; batch={a.batch_size}',flush=True)
    initial=epoch_pass(model,val,vi,a.batch_size)
    write_json(run/'initial_validation.json',initial);checkpoint('initial_model.pt',0,initial)
    print('Epoch000 RANDOM initialization (not baseline policy):',initial,flush=True)
    try:
        with (run/'history.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['epoch','seconds','learning_rate']+[s+'_'+m for s in ('train','val') for m in METRICS])
            for epoch in range(1,a.epochs+1):
                started=time.perf_counter();lr=optimizer.param_groups[0]['lr']
                ix=sample_endpoints(low_ix,other_ix,rec_ix,samples_per_epoch,a.recovery_fraction,generator,device)
                tr=epoch_pass(model,train,ix,a.batch_size,optimizer,amp)
                va=epoch_pass(model,val,vi,a.batch_size)
                scheduler.step(va['balanced_arm_mse'])
                if va['balanced_arm_mse']<best:
                    best=va['balanced_arm_mse'];best_epoch=epoch;stale=0;checkpoint('best_model.pt',epoch,va)
                else:stale+=1
                checkpoint('last_model.pt',epoch,va)
                seconds=time.perf_counter()-started
                writer.writerow([epoch,seconds,lr]+[v[m] for v in (tr,va) for m in METRICS]);f.flush()
                print(f'Epoch {epoch:03d} | train {tr["arm_mse"]:.6f} | val {va["arm_mse"]:.6f} | low {va["low_arm_mse"]:.6f} | '
                      f'other {va["other_arm_mse"]:.6f} | balanced {va["balanced_arm_mse"]:.6f} | lr {lr:.2g} | '
                      f'{seconds:.1f}s | stale {stale}/{a.patience}',flush=True)
                if epoch>=a.min_epochs and stale>=a.patience:
                    print('Early stopping: balanced validation torque MSE did not improve.',flush=True);break
    except BaseException:
        write_json(run/'interrupted.json',dict(best_epoch=best_epoch,smoke_only=smoke));raise
    for field,folder in [('source_model_file_sha256',parent),('frozen_nav_file_sha256',nav)]:
        if c[field]!={n:sha256(folder/n) for n in FROZEN}:raise RuntimeError('Frozen context changed during training')
    write_json(run/'training_finished.json',dict(best_epoch=best_epoch,last_epoch=epoch,
        best_balanced_validation_arm_mse=best,navigation_unchanged=True,smoke_only=smoke,
        arm_trained_from_scratch=True,validation_endpoints=len(vi),validation_all_frames=a.validation_stride==1))
    print(f'Best epoch: {best_epoch}; balanced validation arm MSE: {best:.6f}\nResults: {run}',flush=True)
    print('Training only. Run batch_test_bc_split_nav_arm_cnn_transformer_parallel.py separately.',flush=True)


if __name__=='__main__':main()
