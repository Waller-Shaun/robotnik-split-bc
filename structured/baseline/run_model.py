"""Portable frozen-model entry point. No training or historical path mutation."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/tcn'
sys.path.insert(0, str(SOURCE))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_controller(model):
    from test_structured_arm import StructuredController
    from test_arm_compare import ArmController
    manifest = json.loads((ROOT / 'release_manifest.json').read_text())
    record = manifest['models'][model]
    run = ROOT / record['directory']
    if sha(run / record['checkpoint']) != record['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    cls = ArmController if model == 'baseline' else StructuredController
    return cls(run, record['checkpoint'])


def run_episode(args, controller=None):
    import numpy as np
    import torch
    from rollout_noisy import rollout as noisy_rollout, NOISY_CONFIG
    from test_tcn import rollout as clean_rollout
    torch.set_num_threads(1)
    controller = controller or load_controller(args.model)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    controller.reset()
    args.target = [float(v) for v in args.target]
    args.roboenv_dir = ROOT / 'external/RoboEnv'
    args.sensor_noise = 'config' if args.noise_profile == 'friend' else 'off'
    args.sim_config = NOISY_CONFIG
    args.result_file = Path(args.result_file)
    controller.provenance.update(release_portable_entry=True, evaluation_split='custom_or_shared_target',
                                 release_checkpoint_unchanged=True, no_training=True)
    rollout = noisy_rollout if args.noise_profile == 'friend' else clean_rollout
    if args.headless:
        rollout(args, controller)
    else:
        from fast_gui_demo import live_gui
        args.headless = True
        with live_gui(args.target, fps=args.display_fps, realtime=not args.unpaced) as mirrors:
            rollout(args, controller)
            args.result_file.with_suffix('.timing.json').write_text(json.dumps(mirrors[0].timings(), indent=2))
    return json.loads(args.result_file.with_suffix('.json').read_text())


def main():
    import numpy as np
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=('tcn', 'mlp', 'baseline'), default='tcn')
    p.add_argument('--target', nargs=3, type=float, default=[-.9, 1.2, 1.1])
    p.add_argument('--seed', type=int, default=137)
    p.add_argument('--duration', type=float, default=20.)
    p.add_argument('--noise-profile', choices=('friend', 'off'), default='friend')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--display-fps', type=int, default=30)
    p.add_argument('--unpaced', action='store_true')
    p.add_argument('--result-file', type=Path)
    args = p.parse_args()
    if not np.isfinite(args.target).all() or not np.isfinite(args.duration) or not .001 <= args.duration <= 20:
        p.error('Finite target and duration in [0.001,20] seconds required')
    if not 0 <= args.seed < 2**32 or not 1 <= args.display_fps <= 120:
        p.error('Seed must be uint32 and display FPS must be 1..120')
    if args.result_file is None:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        args.result_file = ROOT / 'outputs' / f'{args.model}_{args.noise_profile}_{stamp}.npz'
    print(f'FROZEN {args.model}; noise={args.noise_profile}; target={args.target}; no training', flush=True)
    run_episode(args)


if __name__ == '__main__':
    main()
