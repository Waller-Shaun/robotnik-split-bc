"""Portable integrity checks, without the author's original data/workspace."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dynamic_pd.common import BASE, sha


def verify(runtime=False):
    manifest = json.loads((BASE/'release_manifest.json').read_text())
    checked = 0
    for name, expected in manifest['file_sha256'].items():
        if sha(BASE/name) != expected:
            raise ValueError('Frozen baseline differs: '+name)
        checked += 1
    selection = json.loads((ROOT/'selection.json').read_text())
    for name, expected in selection['frozen_files'].items():
        if sha(ROOT/name) != expected:
            raise ValueError('Selected experiment differs: '+name)
        checked += 1
    bundle_manifest = ROOT/'experiment_manifest.json'
    if bundle_manifest.exists():
        bundle = json.loads(bundle_manifest.read_text())
        for name, expected in bundle['file_sha256'].items():
            if sha(ROOT/name) != expected:
                raise ValueError('Packaged experiment file differs: '+name)
            checked += 1
    if runtime:
        for name, expected in manifest['dependencies']['file_sha256'].items():
            if sha(BASE/name) != expected:
                raise ValueError('Simulator assets differ: '+name)
            checked += 1
        import torch
        from dynamic_pd.controller import DynamicPDController
        controller = DynamicPDController()
        assert not controller.history and controller.gain_tick == 0
        assert controller.gain_model.training is False
        print('Runtime: Python '+sys.version.split()[0]+', Torch '+torch.__version__)
    report = ROOT/'FINAL_RESULTS.json'
    if report.exists():
        result = json.loads(report.read_text())
        assert result['final_test_complete']
        for model in ('fixed', 'dynamic'):
            row = result['metrics'][model]
            assert row['episodes'] == 1500 and row['distinct_targets'] == 500
            assert abs(row['success_rate_percent']-100*row['successes']/1500) < 1e-10
        print('Recorded final result counts are internally consistent.')
    print(f'PASS: {checked} frozen file hashes; original dataset/ZIP not required for inference or replay.')
    return selection


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', action='store_true')
    verify(parser.parse_args().runtime)
