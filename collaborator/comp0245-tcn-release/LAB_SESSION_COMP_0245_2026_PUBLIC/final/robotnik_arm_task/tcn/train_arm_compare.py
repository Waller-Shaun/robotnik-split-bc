"""Paired arm-only transfer experiment; navigation is frozen, test unused.

Both candidates inherit exactly the same released arm function and statistics.
The train/val split equals its parent split. Samples every 10 ms are pooled;
training resamples low/other heights 50/50. Validation keeps natural frames,
and model selection uses the mean of low/other normalized arm MSEs.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tcn_dataset import DEFAULT_ROOT as ROOT, TASK_DIR, TCNTrajectoryDataset
from arm_compare_models import CLASSES, initialize_from_released, verify_transfer
from train_tcn import write_json

PARENT = ROOT / "artifacts/RoboEnv/bc_training_arm_ring_r08_lowz_supplement/run_20261103_141232_983306"
NAV = ROOT / "artifacts/RoboEnv/bc_training_nav_ring_r08/run_20261103_022339_570160"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ArmWindows(Dataset):
    def __init__(self, raw, stats):
        self.raw = raw
        self.x = [np.ascontiguousarray((x-stats['x_mean'])/stats['x_std']) for x in raw.features]
        self.y = [np.ascontiguousarray((y[:, 4:]-stats['y_mean'])/stats['y_std']) for y in raw.actions]
        manifest = json.loads(raw.manifest_path.read_text())
        height = dict(zip(manifest['splits'][raw.split], manifest['targets'][raw.split]))
        self.low = np.concatenate([np.full(len(x), height[f][2] <= .35+1e-6, dtype=bool)
                                   for x, f in zip(self.x, raw.filenames)])
        self.offsets = np.arange(raw.window_size-1, -1, -1)
        if not self.low.any() or self.low.all():
            raise ValueError('Both height groups required')

    def __len__(self):
        return len(self.low)

    def __getitem__(self, index):
        episode, t = self.raw.locate(index)
        window = self.x[episode][np.maximum(t-self.offsets, 0)]
        return torch.from_numpy(window), torch.from_numpy(self.y[episode][t]), bool(self.low[index])


def epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    sums, counts = np.zeros(2), np.zeros(2, dtype=np.int64)
    with torch.set_grad_enabled(optimizer is not None):
        for x, y, low in loader:
            x, y, low = x.to(device), y.to(device), low.to(device)
            error = (model(x)-y).square()
            loss = error.mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite arm loss')
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                optimizer.step()
            for group, mask in enumerate((low, ~low)):
                sums[group] += error[mask].detach().sum().item()
                counts[group] += int(mask.sum())
    if (counts == 0).any():
        raise ValueError('Missing height group')
    per_group = sums/(7*counts)
    return dict(mse=float(sums.sum()/(7*counts.sum())), low_mse=float(per_group[0]),
                other_mse=float(per_group[1]), balanced_mse=float(per_group.mean()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke-test', action='store_true')
    mode.add_argument('--full', action='store_true')
    p.add_argument('--model', choices=CLASSES, required=True)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--samples-per-epoch', type=int, default=200000)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-5)
    p.add_argument('--patience', type=int, default=6)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if min(args.epochs, args.samples_per_epoch, args.batch_size, args.patience) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        p.error('Invalid training hyperparameters')
    torch.set_num_threads(2)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    parent = torch.load(PARENT/'best_model.pt', map_location='cpu', weights_only=True)
    check = verify_transfer(parent['model_state_dict'])
    print(f'Transfer checks: {check}', flush=True)
    # Reset seed after diagnostics; identical data sampler for both policies.
    torch.manual_seed(args.seed)
    manifest_path = ROOT/'dataset_info/manifests/merged998/data_split.json'
    manifest = json.loads(manifest_path.read_text())
    inherited = json.loads((PARENT/'data_split.json').read_text())
    for split in ('train', 'val', 'test'):
        if set(manifest['splits'][split]) != set(inherited['splits'][split]):
            raise ValueError('Parent and new split differ')
    for name, expected in parent['config']['frozen_nav_file_sha256'].items():
        if sha(NAV/name) != expected:
            raise ValueError('Frozen navigation differs from released pair')
    with np.load(PARENT/'normalization.npz', allow_pickle=False) as data:
        stats = {k:data[k].copy() for k in ('x_mean','x_std','y_mean','y_std')}
    stats['y_mean'], stats['y_std'] = stats['y_mean'][4:], stats['y_std'][4:]
    if any(not np.isfinite(a).all() for a in stats.values()) or any(np.any(stats[k] <= 0) for k in ('x_std', 'y_std')):
        raise ValueError('Invalid inherited normalization')
    raw = {}
    for split in ('train', 'val'):
        print(f'Loading {split} trajectories; no test trajectories loaded', flush=True)
        raw[split] = TCNTrajectoryDataset(split=split, window_size=20, stride=10)
        if args.smoke_test:
            # Choose subset after reading metadata, ensuring both height strata.
            heights = dict(zip(manifest['splits'][split], manifest['targets'][split]))
            selected = []
            for group in (True, False):
                selected.extend([i for i, f in enumerate(raw[split].filenames)
                                 if (heights[f][2] <= .35+1e-6) == group][:2])
            source = raw[split]
            source.features = [source.features[i] for i in selected]
            source.actions = [source.actions[i] for i in selected]
            source.filenames = [source.filenames[i] for i in selected]
            source.cumulative_lengths = np.r_[0, np.cumsum([len(x) for x in source.features])].tolist()
    sets = {key:ArmWindows(value, stats) for key,value in raw.items()}
    low = sets['train'].low
    weights = np.where(low, .5/low.sum(), .5/(~low).sum())
    draws = min(2048 if args.smoke_test else args.samples_per_epoch, len(low))
    sampler = WeightedRandomSampler(torch.from_numpy(weights), draws, replacement=True,
                                     generator=torch.Generator().manual_seed(args.seed))
    loaders = {"train": DataLoader(sets['train'], batch_size=args.batch_size, sampler=sampler),
               "val": DataLoader(sets['val'], batch_size=args.batch_size)}
    model = CLASSES[args.model]()
    initialize_from_released(model, parent['model_state_dict'])
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    run = ROOT/'RoboEnv'/f'bc_arm_transfer_{args.model}'/datetime.now(timezone.utc).strftime(
        'run_%Y%m%d_%H%M%S_%f_'+('smoke' if args.smoke_test else 'full'))
    run.mkdir(parents=True, exist_ok=False)
    config = dict(schema_version=2, model_type=args.model, mode='smoke' if args.smoke_test else 'full',
                  seed=args.seed, raw_dt_s=.001, window_size=20, stride=10,
                  effective_context_samples=20 if args.model=='tcn' else 1,
                  feature_version=raw['train'].feature_version, start_padding='repeat_first_observation',
                  initialization='same released arm weights; TCN current-tap inflation, historical taps zero',
                  parent_run=str(PARENT), parent_sha256=sha(PARENT/'best_model.pt'),
                  parent_normalization_sha256=sha(PARENT/'normalization.npz'),
                  nav_run=str(NAV), frozen_nav_file_sha256=parent['config']['frozen_nav_file_sha256'],
                  source_manifest_sha256=sha(manifest_path), trained_outputs='7 arm torques only',
                  sampling='every tenth frame; low z<=0.35 and other heights 50/50 with replacement',
                  normalization='inherited training-only parent statistics; binary phase unchanged',
                  selection_metric='balanced validation arm MSE; epoch 0 retained as eligible fallback',
                  validation_distribution='all validation trajectories, every tenth frame, no resampling',
                  samples_per_epoch=draws, batch_size=args.batch_size, learning_rate=args.lr,
                  max_epochs=2 if args.smoke_test else args.epochs, patience=args.patience,
                  parameters=sum(p.numel() for p in model.parameters()), device=str(device),
                  torch_version=str(torch.__version__), test_evaluated=False,
                  transfer_checks=check, trained_from_scratch=False)
    sources = [Path(__file__), Path(__file__).with_name('arm_compare_models.py'),
               Path(__file__).with_name('tcn_dataset.py'), Path(__file__).with_name('tcn_model.py'),
               TASK_DIR/'robotnik_bodyframe_features.py']
    config['source_sha256'] = {s.name:sha(s) for s in sources}
    write_json(run/'config.json', config)
    np.savez(run/'normalization.npz', **stats)
    shutil.copy2(manifest_path, run/'data_split.json')
    write_json(run/'used_trajectories.json', {**{k:raw[k].filenames for k in ('train','val')}, 'test':[]})
    for source in sources:
        shutil.copy2(source, run/source.name)
    print(f'RUN: {run}', flush=True)
    print(f'Device: {device}; train: {len(sets["train"])} frames; val: {len(sets["val"])} frames', flush=True)
    initial = epoch(model, loaders['val'], device)
    write_json(run/'initial_validation.json', initial)
    def save(name, number, metrics):
        torch.save(dict(schema_version=2, model_type=args.model, config=config, epoch=number,
                        validation=metrics, normalization={k:torch.from_numpy(v) for k,v in stats.items()},
                        model_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}), run/name)
    best, best_epoch, stale = initial['balanced_mse'], 0, 0
    save('initial_model.pt', 0, initial); save('best_model.pt', 0, initial)
    print(f'Initial balanced arm MSE: {best:.8f}', flush=True)
    with (run/'history.csv').open('w', newline='') as stream:
        fields = ['epoch','seconds']+[f'{s}_{m}' for s in ('train','val') for m in initial]
        writer = csv.DictWriter(stream, fields); writer.writeheader(); stream.flush()
        for number in range(1, config['max_epochs']+1):
            started = time.perf_counter()
            train_metrics = epoch(model, loaders['train'], device, optimizer)
            metrics = epoch(model, loaders['val'], device)
            seconds = time.perf_counter()-started
            writer.writerow(dict(epoch=number, seconds=seconds,
                                 **{f'train_{k}':v for k,v in train_metrics.items()},
                                 **{f'val_{k}':v for k,v in metrics.items()})); stream.flush()
            print(f'Epoch {number:03d}: train={train_metrics["mse"]:.8f}; val balanced={metrics["balanced_mse"]:.8f}; {seconds:.1f}s', flush=True)
            if metrics['balanced_mse'] < best:
                best, best_epoch, stale = metrics['balanced_mse'], number, 0
                save('best_model.pt', number, metrics)
            else:
                stale += 1
            if stale >= args.patience:
                break
    save('last_model.pt', number, metrics)
    restored = torch.load(run/'best_model.pt', weights_only=True, map_location='cpu')
    copy = CLASSES[args.model]().eval(); copy.load_state_dict(restored['model_state_dict'])
    with torch.no_grad():
        prediction = copy(sets['val'][0][0][None])
    assert prediction.shape == (1,7) and torch.isfinite(prediction).all()
    if sha(PARENT/'best_model.pt') != config['parent_sha256'] or any(sha(NAV/k) != v for k,v in config['frozen_nav_file_sha256'].items()):
        raise RuntimeError('Frozen source changed')
    write_json(run/'training_finished.json', dict(status='completed', best_epoch=best_epoch,
               best_balanced_validation_arm_mse=best, epochs_completed=number,
               checkpoint_reload_passed=True, test_evaluated=False, subset_only=args.smoke_test,
               initial_balanced_arm_mse=initial['balanced_mse'], navigation_unchanged=True))
    print(f'TRAINING COMPLETE; best epoch {best_epoch}; balanced arm MSE {best:.8f}; frozen navigation unchanged', flush=True)


if __name__=='__main__':
    main()
