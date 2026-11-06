"""[Recovery change] 仅微调机械臂：原专家示范 + 模型偏离后专家恢复示范。

26→256→256→128→11保持不变，只对后7维力矩求损失；不训练辅助头/底盘。
GPU常驻单帧数据，默认每轮50万抽样，训练与仿真测试分开。
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
from robotnik_arm_aux_policy import resolve_finetuned_pair, load_policy_from_checkpoint
from robotnik_bodyframe_features import build_bodyframe_observations
from train_arm_bc_ring_aux_b2 import check_headers
from recovery_dataset import BASELINE, RUN_ROOT, FROZEN, sha256, read_json, write_json, load_recovery


def load_original(split_manifest, part, stats, limit=None):
    files, lengths = check_headers(split_manifest, part, limit)
    n = sum(lengths)
    x, y, low = np.empty((n, 26), np.float32), np.empty((n, 7), np.float32), np.empty(n, bool)
    offset = 0
    for i, ((filename, target), length) in enumerate(zip(files, lengths), 1):
        with np.load(Path(split_manifest['data_dir'])/filename, allow_pickle=False) as d:
            obs, action, times = d['observations'], d['actions'], d['timestamps']
            if (not np.isfinite(action).all() or not np.isin(obs[:, 22], [0, 1]).all()
                    or not np.all(np.diff(times) > 0)):
                raise ValueError('Invalid original trajectory')
            np.testing.assert_allclose(d['target'], target, atol=1e-6, rtol=0)
            sl = slice(offset, offset+length)
            x[sl] = (build_bodyframe_observations(obs, d['ee_positions'])-stats['x_mean'])/stats['x_std']
            y[sl] = (action[:, 4:11]-stats['y_mean'][4:11])/stats['y_std'][4:11]
            low[sl] = target[2] <= .35+1e-6
        offset += length
        if i % 100 == 0 or i == len(files):print(f'{part}: {i}/{len(files)} original trajectories', flush=True)
    if not low.any() or low.all():raise ValueError('Both original height groups required')
    return x, y, low


class Store:
    def __init__(self, x, y, low, device):
        self.x, self.y, self.low = [torch.as_tensor(a).to(device) for a in (x, y, low)]


def epoch_pass(model, store, indices, batch_size, optimizer=None):
    model.train(optimizer is not None)
    sums = torch.zeros(2, dtype=torch.float64, device=store.x.device)
    counts = torch.zeros(2, dtype=torch.int64, device=store.x.device)
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0, len(indices), batch_size):
            ix = indices[start:start+batch_size]
            errors = (model(store.x[ix])[:, 4:11]-store.y[ix]).square().mean(1)
            loss = errors.mean()
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite torque loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
            masks = torch.stack([store.low[ix], ~store.low[ix]])
            sums += masks.double() @ errors.detach().double()
            counts += masks.sum(1)
    s, c = sums.cpu().numpy(), counts.cpu().numpy()
    groups = [float(s[i]/c[i]) if c[i] else None for i in range(2)]
    return {'arm_mse':float(s.sum()/c.sum()), 'low_arm_mse':groups[0], 'other_arm_mse':groups[1],
        'balanced_arm_mse':float(np.mean(groups)) if all(g is not None for g in groups) else None}


def draw_indices(original_low, original_other, recovery_start, recovery_count, count, fraction, generator, device):
    # [Recovery mixing] 20%恢复片段；余下80%原数据内部仍按低/其他高度各半抽样。
    nr = int(round(count*fraction)); no = count-nr
    nl = no//2; nh = no-nl
    parts = [original_low[torch.randint(len(original_low), (nl,), generator=generator)],
        original_other[torch.randint(len(original_other), (nh,), generator=generator)]]
    if nr:parts.append(torch.randint(recovery_count, (nr,), generator=generator)+recovery_start)
    ix = torch.cat(parts)
    return ix[torch.randperm(len(ix), generator=generator)].to(device)


def update_early_stopping(value, epoch, best, best_epoch, stale, min_epochs, patience):
    """[Recovery early-stop fix] 只比较微调后各轮；下降到微调新纪录就清零计数。

    第0轮原基线另存作对照，不阻止新数据分布适应；最低训练轮数只限制停止时机。
    """
    if not np.isfinite(value):raise RuntimeError('Nonfinite balanced validation MSE')
    improved = value < best
    if improved:best, best_epoch, stale = value, epoch, 0
    else:stale += 1
    stop = epoch >= min_epochs and stale >= patience
    return best, best_epoch, stale, improved, stop


def main():
    root = Path(__file__).resolve().parents[3]/'RoboEnv'
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent-arm-run-dir', type=Path, default=root/BASELINE)
    # [Recovery resume] 使用上次last恢复权重、Adam、抽样RNG及早停状态；--epochs为累计上限。
    p.add_argument('--resume-run-dir', type=Path, help='Continue completed recovery run from last_model.pt; epochs is TOTAL limit')
    p.add_argument('--recovery-manifest', type=Path, help='Default: latest completed expert_recovery_data collection')
    p.add_argument('--epochs', type=int, default=100)
    # [Recovery early-stop fix] 至少50轮再允许早停，最多仍由--epochs限制。
    p.add_argument('--min-epochs', type=int, default=50)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=4096)
    p.add_argument('--samples-per-epoch', type=int, default=500_000)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--recovery-fraction', type=float, default=.2, help='0: matched additional-training control; default .2')
    p.add_argument('--device', choices=['auto','cuda','cpu'], default='auto')
    p.add_argument('--output-dir', type=Path)
    p.add_argument('--max-train-files', type=int)
    p.add_argument('--max-val-files', type=int)
    p.add_argument('--check-data-only', action='store_true')
    a = p.parse_args()
    if (min(a.epochs,a.min_epochs,a.patience,a.batch_size,a.samples_per_epoch)<1
            or a.min_epochs>a.epochs or not 0<=a.recovery_fraction<1
            or not np.isfinite(a.learning_rate) or a.learning_rate<=0
            or any(v is not None and v<4 for v in (a.max_train_files,a.max_val_files))):p.error('Invalid settings')
    resume = None
    if a.resume_run_dir is not None:
        source = a.resume_run_dir.resolve()
        rc = read_json(source/'config.json')
        finished = read_json(source/'training_finished.json')
        if (not rc.get('recovery_finetune') or rc.get('smoke_only') != finished.get('smoke_only')
                or not (source/'baseline_model.pt').is_file()):
            raise ValueError('Resume requires a completed recovery run with separate baseline_model.pt')
        if Path(rc['recovery_parent_run_dir']).resolve() != a.parent_arm_run_dir.resolve():
            raise ValueError('Resume must use the same original baseline parent')
        for field, value in [('batch_size',a.batch_size),('samples_per_epoch',a.samples_per_epoch),
                ('learning_rate',a.learning_rate),('recovery_fraction',a.recovery_fraction)]:
            if rc[field] != value:raise ValueError(f'Resume must keep {field}={rc[field]}')
        if bool(rc.get('smoke_only')) != (a.max_train_files is not None or a.max_val_files is not None):
            raise ValueError('Do not mix smoke and formal resume runs')
        resume = {name:torch.load(source/name,map_location='cpu',weights_only=True)
            for name in ('last_model.pt','best_model.pt','baseline_model.pt')}
        last = resume['last_model.pt']
        if (any(cp['config'] != rc for cp in resume.values())
                or last['epoch'] != finished['last_epoch'] or a.epochs <= last['epoch']
                or resume['best_model.pt']['epoch'] != finished['best_epoch']):
            raise ValueError('Resume checkpoints/config differ or total epochs does not exceed last epoch')
        for path, expected in read_json(source/'training_source_hashes.json').items():
            if sha256(path) != expected:raise ValueError(f'Resume source changed: {path}')
        if a.recovery_manifest is None:a.recovery_manifest=Path(rc['recovery_data_manifest_path'])
        if sha256(a.recovery_manifest) != rc['recovery_data_manifest_sha256']:
            raise ValueError('Resume recovery manifest must be unchanged')
        resume.update(source=source,config=rc,finished=finished)
    if a.recovery_manifest is None:
        manifests=sorted((root/'expert_recovery_data').glob('collection_*/episodes/recovery_manifest.json'))
        if not manifests:raise FileNotFoundError('Collect recovery demonstrations first, or pass --recovery-manifest')
        a.recovery_manifest=manifests[-1]
    parent, nav, pc, nc = resolve_finetuned_pair(root, a.parent_arm_run_dir)
    if pc['aux_weight'] != 0:raise ValueError('Start from the torque-only noisy baseline')
    split = read_json(parent/'data_split.json')
    with np.load(parent/'normalization.npz') as d:
        stats = {k:d[k].copy() for k in ('x_mean','x_std','y_mean','y_std')}
    rx, ry, rl, recovery = load_recovery(a.recovery_manifest, parent, split, stats)
    if a.check_data_only:
        counts = {part:sum(check_headers(split,part,limit)[1]) for part,limit in [('train',a.max_train_files),('val',a.max_val_files)]}
        print('Original frames:', counts, '; recovery:', recovery['frames'], '; training started: False'); return
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if a.device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    device = torch.device('cuda' if a.device=='cuda' or (a.device=='auto' and torch.cuda.is_available()) else 'cpu')
    if device.type=='cuda':torch.cuda.manual_seed_all(42)
    else:torch.set_num_threads(min(4,torch.get_num_threads()))
    print('Device:',device, '; GPU:',torch.cuda.get_device_name(0) if device.type=='cuda' else 'none', flush=True)
    print('Recovery manifest:',a.recovery_manifest, '; fraction:',a.recovery_fraction,flush=True)
    checkpoint = torch.load(parent/'best_model.pt',map_location='cpu',weights_only=True)
    if checkpoint['config'] != pc:raise ValueError('Parent checkpoint/config differ')
    if resume is not None:checkpoint=resume['last_model.pt']
    model = load_policy_from_checkpoint(checkpoint)
    # [Recovery training] 辅助头从不参与新损失、不会更新；旧轮速输出行也没有梯度。
    for parameter in model.aux_head.parameters():parameter.requires_grad_(False)
    model.to(device)
    x,y,low = load_original(split,'train',stats,a.max_train_files)
    original_count = len(x)
    low_ix, other_ix = torch.from_numpy(np.flatnonzero(low)), torch.from_numpy(np.flatnonzero(~low))
    train = Store(np.concatenate([x,rx]),np.concatenate([y,ry]),np.concatenate([low,rl]),device)
    recovery_count = len(rx)
    del x,y,low,rx,ry,rl
    xv,yv,lv = load_original(split,'val',stats,a.max_val_files)
    val = Store(xv,yv,lv,device); val_count = len(xv)
    del xv,yv,lv
    if resume is not None:
        for field,count in [('original_train_frames',original_count),('val_samples',val_count),('recovery_train_frames',recovery_count)]:
            if resume['config'][field]!=count:raise ValueError(f'Resume dataset changed: {field}')
    generator = torch.Generator().manual_seed(42)
    vi = torch.arange(val_count,device=device)
    ri = torch.arange(original_count,original_count+recovery_count,device=device)
    # [Recovery optimization] 新Adam，较小学习率；和恢复比例为0的对照采用相同训练设置。
    optimizer = torch.optim.Adam([v for v in model.parameters() if v.requires_grad],lr=a.learning_rate)
    if resume is not None:
        # Adam状态随参数迁移至目标设备；抽样生成器独立恢复，避免从头重复第1轮样本顺序。
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        generator.set_state(checkpoint['sampling_generator_state'].cpu())
        if 'torch_rng_state' in checkpoint:torch.set_rng_state(checkpoint['torch_rng_state'].cpu())
        if device.type=='cuda' and checkpoint.get('cuda_rng_state_all'):
            torch.cuda.set_rng_state_all([v.cpu() for v in checkpoint['cuda_rng_state_all']])
    smoke = a.max_train_files is not None or a.max_val_files is not None
    run_root = RUN_ROOT if a.recovery_fraction else RUN_ROOT+'_control'
    run = (a.output_dir or root/run_root/datetime.now().strftime('run_%Y%m%d_%H%M%S_%f')).resolve()
    run.mkdir(parents=True,exist_ok=False)
    for n in ('normalization.npz','data_split.json','aux_normalization.npz'):shutil.copyfile(parent/n,run/n)
    shutil.copyfile(a.recovery_manifest,run/'recovery_manifest.json')
    # 保留原B2推理兼容字段与祖先绑定；明确记录实际微调起点，不伪装成重新训练。
    config = dict(pc)
    config.update({'recovery_finetune':True,'recovery_parent_run_dir':str(parent),
        'recovery_parent_file_sha256':{n:sha256(parent/n) for n in FROZEN},
        'recovery_data_manifest_path':str(a.recovery_manifest.resolve()),
        'recovery_data_manifest_sha256':sha256(a.recovery_manifest),'recovery_fraction':a.recovery_fraction,
        'recovery_train_frames':recovery_count,'original_train_frames':original_count,'val_samples':val_count,
        'recovery_input_dim':26,'recovery_arm_label_dim':7,'recovery_scope':'train_targets_only',
        'model_prefix_used_as_labels':False,'auxiliary_head_frozen':True,'aux_weight':0.,
        'loss_type':'normalized_arm_torque_MSE_original_plus_recovery',
        'sampling_method':'fixed_recovery_fraction_original_low_other_equal_with_replacement',
        'samples_per_epoch':a.samples_per_epoch,'batch_size':a.batch_size,'max_epochs':a.epochs,
        'patience':a.patience,'min_epochs':a.min_epochs,'learning_rate':a.learning_rate,'seed':42,'device':str(device),
        'early_stopping_reference':'best_post_finetune_epoch_excluding_baseline_epoch0',
        'checkpoint_selection_scope':'post_finetune_epochs_1_to_max_epochs',
        'baseline_checkpoint_file':'baseline_model.pt',
        'smoke_only':smoke,'optimizer_initialization':'new_Adam_recovery_finetune',
        'validation_metrics_distribution':'original_frozen_val_natural_frames_no_recovery',
        'recovery_eval_is_training_diagnostic_only':True,'training_source_sha256':sha256(__file__)})
    start_epoch=checkpoint['epoch'] if resume is not None else 0
    config.update({'resumed':resume is not None,'resume_from_epoch':start_epoch,
        'resume_from_run_dir':str(resume['source']) if resume is not None else None,
        'optimizer_initialization':'restored_Adam_last_checkpoint' if resume is not None else 'new_Adam_recovery_finetune',
        'sampling_rng_restored':resume is not None,'epochs_definition':'cumulative_total_epoch_limit'})
    write_json(run/'config.json',config)
    shutil.copyfile(__file__,run/'training_source.py')
    frozen = {str(folder/n):sha256(folder/n) for folder in (parent,nav) for n in FROZEN}
    frozen[str(a.recovery_manifest.resolve())] = sha256(a.recovery_manifest)
    frozen.update({str(a.recovery_manifest.resolve().parent/e['file']):e['sha256'] for e in recovery['entries']})
    if resume is not None:
        frozen.update({str(resume['source']/n):sha256(resume['source']/n)
            for n in (*FROZEN,'last_model.pt','baseline_model.pt','history.csv','training_finished.json','training_source_hashes.json')})
        shutil.copyfile(resume['source']/'history.csv',run/'previous_history.csv')
    write_json(run/'training_source_hashes.json',frozen)
    def verify():
        if any(sha256(path)!=expected for path,expected in frozen.items()):raise RuntimeError('Frozen source/data changed during training')
    initial = epoch_pass(model,val,vi,a.batch_size)
    write_json(run/'initial_validation.json',initial)
    # [Recovery early-stop fix] 初始值不再作为微调早停门槛，原基线仍明确保存并报告差距。
    baseline_score = initial['balanced_arm_mse']
    best, best_epoch, stale = float('inf'),0,0
    if resume is not None:
        baseline_score=resume['baseline_model.pt']['val_metrics']['balanced_arm_mse']
        best=resume['best_model.pt']['val_metrics']['balanced_arm_mse']
        best_epoch=resume['best_model.pt']['epoch']
        saved_stop=checkpoint.get('early_stopping_state')
        if saved_stop:
            if saved_stop['best_epoch']!=best_epoch or saved_stop['best']!=best:
                raise ValueError('Resume early-stopping state/checkpoint differ')
            stale=saved_stop['stale']
        else:
            # 兼容此次已完成的100轮旧检查点：由已有历史CSV恢复未改善计数。
            with (resume['source']/'history.csv').open(encoding='utf-8',newline='') as h:rows=list(csv.DictReader(h))
            if (not rows or int(rows[-1]['epoch'])!=start_epoch
                    or int(rows[-1]['best_finetune_epoch'])!=best_epoch
                    or float(rows[-1]['best_finetune_val_balanced_arm_mse'])!=best):
                raise ValueError('Resume history/checkpoint differ')
            stale=int(rows[-1]['epochs_without_new_finetune_best'])
    def save(name,epoch,metrics):
        torch.save({'model_state_dict':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            'optimizer_state_dict':optimizer.state_dict(),'sampling_generator_state':generator.get_state(),
            'torch_rng_state':torch.get_rng_state(),
            'cuda_rng_state_all':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
            'early_stopping_state':{'best':best if np.isfinite(best) else None,'best_epoch':best_epoch,'stale':stale},
            'epoch':epoch,'val_mse':metrics['arm_mse'],'val_loss':metrics['balanced_arm_mse'],
            'val_metrics':metrics,'config':config},run/name)
    if resume is None:save('baseline_model.pt',0,initial)
    else:
        # 保留历史最佳（不一定等于last）和最初基线，绝不将第100轮误标成第0轮。
        for name in ('best_model.pt','baseline_model.pt'):
            inherited=dict(resume[name]);inherited['config']=config;torch.save(inherited,run/name)
    print('Output:',run,'; initial val:',initial,flush=True)
    print(f'Early stop: min_epochs={a.min_epochs}, patience={a.patience}, max_epochs={a.epochs}; '
          'best_model selects POST-finetune epochs; epoch0 is baseline_model.pt.',flush=True)
    if resume is not None:
        print(f'Resumed epoch {start_epoch}; Adam and sampling RNG restored; next epoch {start_epoch+1}; '
              f'best {best:.6f}@{best_epoch}; stale={stale}/{a.patience}; original baseline={baseline_score:.6f}',flush=True)
    metrics = ('arm_mse','low_arm_mse','other_arm_mse','balanced_arm_mse')
    with (run/'history.csv').open('w',newline='',encoding='utf-8') as f:
        writer = csv.writer(f);writer.writerow(['epoch','seconds',*[s+'_'+m for s in ('train','val') for m in metrics],
            'recovery_train_mse','best_finetune_val_balanced_arm_mse','best_finetune_epoch',
            'epochs_without_new_finetune_best','early_stop_allowed','val_minus_baseline_balanced_arm_mse'])
        for epoch in range(start_epoch+1,a.epochs+1):
            verify();start=time.perf_counter()
            ix=draw_indices(low_ix,other_ix,original_count,recovery_count,a.samples_per_epoch,a.recovery_fraction,generator,device)
            tr=epoch_pass(model,train,ix,a.batch_size,optimizer)
            va=epoch_pass(model,val,vi,a.batch_size)
            # 纠偏训练误差只作诊断，不能用它挑最佳轮次。
            rec=epoch_pass(model,train,ri,a.batch_size)['arm_mse']
            seconds=time.perf_counter()-start
            best,best_epoch,stale,improved,stop = update_early_stopping(
                va['balanced_arm_mse'],epoch,best,best_epoch,stale,a.min_epochs,a.patience)
            delta = va['balanced_arm_mse']-baseline_score
            writer.writerow([epoch,seconds,*[d[m] for d in (tr,va) for m in metrics],rec,
                best,best_epoch,stale,epoch>=a.min_epochs,delta]);f.flush()
            gate = 'enabled' if epoch>=a.min_epochs else f'locked_until_{a.min_epochs}'
            print(f"Epoch {epoch:03d} | train {tr['arm_mse']:.6f} | val {va['arm_mse']:.6f} | low {va['low_arm_mse']:.6f} | other {va['other_arm_mse']:.6f} | balanced {va['balanced_arm_mse']:.6f} | recovery TRAIN {rec:.6f} | best FT {best:.6f}@{best_epoch} | stale {stale}/{a.patience} | stop {gate} | delta baseline {delta:+.6f} | {seconds:.2f}s",flush=True)
            if improved:save('best_model.pt',epoch,va)
            save('last_model.pt',epoch,va)
            if stop:
                print('Early stopping: minimum epochs completed and no new POST-finetune validation best for patience epochs.');break
    verify()
    write_json(run/'training_finished.json',{'best_epoch':best_epoch,'last_epoch':epoch,
        'best_balanced_validation_arm_mse':best,'smoke_only':smoke,'navigation_unchanged':True,
        'recovery_finetune':True,'recovery_fraction':a.recovery_fraction,'min_epochs':a.min_epochs,
        'baseline_balanced_validation_arm_mse':baseline_score,
        'best_finetune_beats_baseline_mse':bool(best<baseline_score),
        'best_finetune_minus_baseline_mse':best-baseline_score,
        'checkpoint_selection_scope':'post_finetune_epochs_only_baseline_saved_separately',
        'resumed':resume is not None,'resume_from_epoch':start_epoch,'epochs_this_run':epoch-start_epoch})
    # 总轮数沿用原实验编号；原实验目录不写入任何文件。
    print(f'Best recovery epoch: {best_epoch}; balanced validation arm MSE: {best:.6f}\nResults: {run}')
    print(f'Baseline MSE: {baseline_score:.6f}; best finetune minus baseline: {best-baseline_score:+.6f}. '
          'Closed-loop success must be evaluated separately.')
    print('Training only. Run batch_test_arm_recovery_parallel.py separately.')


if __name__=='__main__':main()
