"""Verify packaged bytes and (optionally) controller loading/inference/reset."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime', action='store_true')
    args = p.parse_args()
    manifest = json.loads((ROOT / 'release_manifest.json').read_text())
    for name, expected in manifest['file_sha256'].items():
        if sha(ROOT / name) != expected:
            raise ValueError('Packaged file differs: ' + name)
    print(f"PASS: {len(manifest['file_sha256'])} packaged file fingerprints")
    if args.runtime:
        import numpy as np
        import torch
        from run_model import load_controller, SOURCE
        sys.path.insert(0, str(SOURCE.parent))
        from robotnik_arm_control import BaseNavState
        for name, expected in manifest['dependencies']['file_sha256'].items():
            if sha(ROOT / name) != expected:
                raise ValueError('Simulator/asset differs: ' + name)
        torch.set_num_threads(1)
        from test_tcn import ROOT as resolved_root
        assert resolved_root == ROOT
        # Synthetic finite state: verifies dimensions/reset, not task success.
        observation = np.zeros(26, dtype=np.float32)
        observation[4] = 1
        observation[19:22] = [-.9, 1.2, 1.1]
        observation[22] = 1
        context = dict(base_position=np.zeros(3), base_orientation=np.array([0.,0.,0.,1.]),
                       bearing=0., goal_position=np.array([-.51,.68]), goal_bearing=2.2143,
                       phase=BaseNavState(phase='align'))
        for model in manifest['models']:
            controller = load_controller(model)
            first = controller.predict(observation, **context)
            assert first.shape == (11,) and np.isfinite(first).all()
            controller.predict(observation, **context)
            controller.reset()
            np.testing.assert_array_equal(first, controller.predict(observation, **context))
            print(f'PASS: {model} checkpoint, normalization, navigation, finite inference, exact episode reset')
        print('PASS: portable ROOT and pinned simulator fingerprints; no train/test trajectories loaded')


if __name__ == '__main__':
    main()
