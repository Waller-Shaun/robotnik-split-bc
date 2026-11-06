"""Portable NEW gain-training run; never replace the audited model/results."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_experiment import verify
from dynamic_pd.common import BASE, sha, write
from dynamic_pd.model import GainNetwork
from dynamic_pd.train import Samples, run_epoch
import dynamic_pd.dataset as dataset


def make_model(hidden, device):
    """Cache-hit-independent initialization, using training features only."""
    torch.manual_seed(42)
    np.random.seed(42)
    mean = hidden.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.maximum(hidden.std(axis=0, dtype=np.float64), .1).astype(np.float32)
    return GainNetwork(torch.from_numpy(mean), torch.from_numpy(std)).to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Directory containing RoboEnv/ and the original expert data subdirectories')
    parser.add_argument('--output', type=Path, required=True, help='New non-existing output directory')
    parser.add_argument('--smoke-test', action='store_true')
    args = parser.parse_args()
    data_root, output = args.data_root.resolve(), args.output.resolve()
    if not (data_root/'RoboEnv').is_dir():
        parser.error('--data-root must contain RoboEnv/ with the original expert dataset')
    if output.exists():
        parser.error('Output exists: never overwrite a previous experiment')
    verify(runtime=True)
    output.mkdir(parents=True)
    write(output/'local_paths.json', dict(dataset_root=str(data_root), training_data_read_only=True))
    torch.set_num_threads(2)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    arrays, names = {}, {}
    # Only the data/cache location changes. Frozen source, models, assets and
    # their integrity checks stay untouched. No archived audit is bypassed.
    old_cache_root = dataset.ROOT
    try:
        dataset.ROOT = output
        for split in ('train', 'val'):
            arrays[split], names[split] = dataset.prepare(split, args.smoke_test, device)
    finally:
        dataset.ROOT = old_cache_root
    assert not set(names['train']) & set(names['val'])
    # Seed AFTER feature extraction. Otherwise construction of frozen networks
    # on a cache miss consumes RNG while a cache hit does not.
    hidden = arrays['train']['hidden']
    model = make_model(hidden, device)
    with np.load(BASE/'models/tcn/normalization.npz') as stored:
        scale = torch.from_numpy(stored['torque_std'].copy()).to(device)
    weight = arrays['train']['weight'].copy()
    for value in (False, True):
        mask = arrays['train']['low'] == value
        weight[mask] *= .5/weight[mask].sum()
    draws = min(2048 if args.smoke_test else 200000, len(hidden))
    sampler = WeightedRandomSampler(torch.from_numpy(weight), draws, True,
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(Samples(arrays['train']), batch_size=4096, sampler=sampler)
    val_loader = DataLoader(Samples(arrays['val']), batch_size=4096)
    epochs = 2 if args.smoke_test else 20
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=1e-4)
    config = dict(schema_version=1, mode='portable_smoke' if args.smoke_test else 'portable_full',
        seed=42, seed_reset_after_feature_preparation=True,
        trained_parameters=sum(p.numel() for p in model.parameters()),
        maximum_epochs=epochs, samples_per_epoch=draws, batch_size=4096,
        final_test_used=False, original_recorded_model_not_overwritten=True,
        not_the_model_in_recorded_final_results=True,
        offline_qref='recorded expert references; training loss only',
        online_qref='integral of learned velocity; no recorded references',
        p_bounds=[80.,120.], d_bounds=[24.,36.], gain_filter_time_constant_s=.010,
        regularization=dict(deviation=.02, temporal=.02), device=str(device),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            (ROOT/'dynamic_pd/model.py', ROOT/'dynamic_pd/controller.py', ROOT/'dynamic_pd/dataset.py', Path(__file__))})
    write(output/'config.json', config)
    write(output/'used_trajectories.json', dict(**names, test=[], final_test=[]))
    initial = run_epoch(model, val_loader, scale, device)
    best, best_epoch, stale = float('inf'), 0, 0
    with (output/'history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, ['epoch','seconds','train_loss','val_loss','val_normalized_torque_mse'])
        writer.writeheader()
        for epoch in range(1, epochs+1):
            start = time.perf_counter()
            train = run_epoch(model, train_loader, scale, device, optimizer)
            val = run_epoch(model, val_loader, scale, device)
            scheduler.step()
            writer.writerow(dict(epoch=epoch, seconds=time.perf_counter()-start,
                train_loss=train['loss'], val_loss=val['loss'],
                val_normalized_torque_mse=val['normalized_torque_mse']))
            stream.flush()
            print(f"Epoch {epoch}: val={val['loss']:.7f}", flush=True)
            if val['loss'] < best:
                best, best_epoch, stale = val['loss'], epoch, 0
                torch.save(dict(schema_version=1, epoch=epoch, config=config, validation=val,
                    base_checkpoint_sha256=sha(BASE/'models/tcn/best_model.pt'),
                    gain_state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}),
                    output/'best_model.pt')
            else:
                stale += 1
            if stale >= 6:
                break
    from dynamic_pd.controller import DynamicPDController
    controller = DynamicPDController(checkpoint=output/'best_model.pt')
    assert controller.gain_tick == 0
    verify(runtime=False)
    write(output/'training_finished.json', dict(state='completed', best_epoch=best_epoch,
        initial_validation=initial, selected_loss=best, final_test_used=False,
        recorded_final_success_rate_not_applicable=True, checkpoint_sha256=sha(output/'best_model.pt'),
        checkpoint_reload_passed=True, smoke_only=args.smoke_test))
    print('Saved a NEW training run: '+str(output))
    print('No independent success rate yet. Do not substitute the archived final score for these new weights.')


if __name__ == '__main__':
    main()
