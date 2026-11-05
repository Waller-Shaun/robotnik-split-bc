"""Frozen learned navigation + transferred arm MLP/TCN; shared physics loop."""
from collections import deque
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from test_tcn import ROOT, rollout
from arm_compare_models import CLASSES
from check_reference_mlp import NAV_RUN, load_stats, sha256
from robotnik_navigation_policy import load_navigation_policy, build_navigation_observation
from robotnik_bodyframe_features import BODYFRAME_FEATURE_VERSION, build_bodyframe_observations


class ArmController:
    def __init__(self, run_dir, checkpoint='best_model.pt'):
        self.run_dir = Path(run_dir).resolve()
        self.checkpoint_path = self.run_dir/checkpoint
        ckpt = torch.load(self.checkpoint_path, weights_only=True, map_location='cpu')
        if ckpt.get('schema_version') != 2 or ckpt['model_type'] not in CLASSES:
            raise ValueError('Expected arm transfer checkpoint')
        self.config = ckpt['config']
        if self.config['feature_version'] != BODYFRAME_FEATURE_VERSION or self.config['start_padding'] != 'repeat_first_observation':
            raise ValueError('Feature/history definition differs')
        self.kind = ckpt['model_type']
        self.policy_type = f'frozen_nav8_transferred_Arm{self.kind.upper()}'
        self.model_source_path = Path(__file__).with_name('arm_compare_models.py')
        if sha256(self.model_source_path) != self.config['source_sha256'][self.model_source_path.name]:
            raise ValueError('Model source changed since training')
        self.model = CLASSES[self.kind]().eval()
        self.model.load_state_dict(ckpt['model_state_dict'])
        if not all(torch.isfinite(v).all() for v in self.model.parameters()):
            raise ValueError('Invalid checkpoint')
        self.stats = {k:v.numpy().copy() for k,v in ckpt['normalization'].items()}
        stored = load_stats(self.run_dir, 26, 7)
        for key in stored:
            np.testing.assert_array_equal(stored[key], self.stats[key])
        self.nav_stats = load_stats(NAV_RUN, 8, 4)
        for name, expected in self.config['frozen_nav_file_sha256'].items():
            if sha256(NAV_RUN/name) != expected:
                raise ValueError('Frozen navigation changed')
        nav = torch.load(NAV_RUN/'best_model.pt', weights_only=True, map_location='cpu')
        self.nav = load_navigation_policy(nav).eval()
        self.manifest = json.loads((self.run_dir/'data_split.json').read_text())
        if sha256(self.run_dir/'data_split.json') != self.config['source_manifest_sha256']:
            raise ValueError('Manifest changed')
        self.window_size, self.stride = self.config['window_size'], self.config['stride']
        self.offsets = np.arange(self.window_size-1,-1,-1)*self.stride
        self.history = deque(maxlen=1+(self.window_size-1)*self.stride)
        self.epoch, self.val_mse = ckpt['epoch'], ckpt['validation']['mse']
        self.provenance = dict(experiment='fixed_learned_navigation_arm_only_transfer',
                               checkpoint=str(self.checkpoint_path), training_epoch=self.epoch,
                               initialization=self.config['initialization'],
                               pretrained_parent=self.config['parent_run'],
                               trained_from_scratch=False, navigation_unchanged=True,
                               nav_sha256=self.config['frozen_nav_file_sha256'],
                               evaluator_sha256=sha256(Path(__file__)))

    def reset(self):
        self.history.clear()

    def push(self, observation):
        value = np.asarray(observation, dtype=np.float32)
        if value.shape != (26,) or not np.isfinite(value).all():
            raise ValueError('Invalid observation')
        self.history.append(value.copy())
        frames = np.asarray(self.history)
        return frames[np.maximum(len(frames)-1-self.offsets, 0)]

    def predict(self, observation, **context):
        window = self.push(observation)
        x = np.ascontiguousarray((window-self.stats['x_mean'])/self.stats['x_std'])
        nav_x = build_navigation_observation(context['base_position'], context['bearing'],
                                             context['goal_position'], context['goal_bearing'], context['phase'])
        with torch.inference_mode():
            arm = self.model(torch.from_numpy(x)[None])[0].numpy()
            wheels = self.nav(torch.from_numpy((nav_x-self.nav_stats['x_mean'])/self.nav_stats['x_std'])[None])[0].numpy()
        action = np.r_[wheels*self.nav_stats['y_std']+self.nav_stats['y_mean'],
                       arm*self.stats['y_std']+self.stats['y_mean']]
        if not np.isfinite(action).all():
            raise FloatingPointError('Invalid policy action')
        return action


def check_history(controller):
    path = ROOT/'RoboEnv'/controller.manifest['splits']['train'][0]
    with np.load(path, allow_pickle=False) as data:
        features = build_bodyframe_observations(data['observations'][:250], data['ee_positions'][:250])
    for frame, observation in enumerate(features):
        np.testing.assert_array_equal(controller.push(observation), features[np.maximum(frame-controller.offsets, 0)])
    controller.reset()
    print('PASS: checkpoint, normalization, frozen navigation, online/offline causal history', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--checkpoint', choices=('best_model.pt','initial_model.pt','last_model.pt'), default='best_model.pt')
    p.add_argument('--target', nargs=3, type=float)
    p.add_argument('--val-index', type=int, default=79)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--duration', type=float, default=20.)
    p.add_argument('--sensor-noise', choices=('off','config'), default='off')
    p.add_argument('--headless', action='store_true')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--result-file', type=Path)
    p.add_argument('--roboenv-dir', type=Path, default=ROOT/'external/RoboEnv')
    args = p.parse_args()
    if not 0 <= args.val_index < 144 or not np.isfinite(args.duration) or args.duration < .001:
        p.error('Invalid target index/duration')
    torch.set_num_threads(1)
    controller = ArmController(args.run_dir, args.checkpoint)
    check_history(controller)
    if args.check_only:
        return
    if args.target is None:
        args.target = controller.manifest['targets']['val'][args.val_index]
    if args.seed is None:
        args.seed = int(controller.manifest['evaluation_seeds']['val'][args.val_index])
    np.random.seed(args.seed); torch.manual_seed(args.seed)
    rollout(args, controller)


if __name__=='__main__':
    main()
