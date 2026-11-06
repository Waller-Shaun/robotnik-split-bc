"""Prepare a byte-identical baseline copy and a predeclared final target list."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    original = WORKSPACE / 'robotnik-split-bc-main'
    release = WORKSPACE / 'comp0245-tcn-release'
    target = ROOT / 'baseline'
    if target.exists():
        raise FileExistsError('Never overwrite an experiment baseline')
    manifest = json.loads((release / 'release_manifest.json').read_text())
    target.mkdir()
    for name, expected in manifest['file_sha256'].items():
        source = release / name
        if sha(source) != expected:
            raise ValueError('Frozen release file differs: ' + name)
        dest = target / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        assert sha(dest) == expected
    shutil.copy2(release / 'release_manifest.json', target / 'release_manifest.json')
    plan_path = original / 'RoboEnv/prospective_noise_20261104/plan.json'
    old = json.loads(plan_path.read_text())
    old_folder = plan_path.parent
    if any(p.is_dir() and p.name.startswith('reserve_') for p in old_folder.iterdir()):
        raise ValueError('Original reserve may already have been evaluated; stop and audit')
    initial = dict(original_plan_sha256=sha(plan_path),
        development=old['partitions']['development'], final=old['partitions']['reserve'],
        final_targets_previously_unused=True, final_target_count=500,
        repeats_per_final_target=3, paired_conditions=['fixed', 'dynamic'],
        final_episodes_per_model=1500, duration_s=20., dt_s=.001,
        true_ee_threshold_m=.01, required_hold_s=0,
        noise='eight native Gaussian getter standard deviations 0.001',
        selected_on='original offline validation torque loss plus fixed regularizers; no final outcomes',
        developer_target_results_not_independent_final=True,
        original_frozen_file_sha256=old['file_sha256'],
        original_submission_zip_sha256=sha(WORKSPACE / 'comp0245-tcn-release.zip'),
        storage_profile='True EE positions/errors and actions full 1ms; diagnostics and gains 10ms; no control/scoring downsampling',
        gain_design=dict(p_center=100.,d_center=30.,relative_span=.2,
                         filter_time_constant_s=.010,head_hidden_units=32,
                         deviation_regularizer=.02,temporal_regularizer=.02))
    (ROOT / 'protocol.json').write_text(json.dumps(initial, indent=2))
    (ROOT / 'local_paths.json').write_text(json.dumps(dict(dataset_root=str(original),
        training_data_read_only=True,original_release=str(release)),indent=2))
    print('Prepared independent baseline; original files untouched. Final goals not evaluated.')


if __name__ == '__main__':
    main()
