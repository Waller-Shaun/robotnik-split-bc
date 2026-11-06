"""Bundle exact selected code/models/results, excluding caches/raw rollouts/envs."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from verify_experiment import verify
from dynamic_pd.common import sha, write


def main():
    selection = verify(runtime=False)
    final = json.loads((ROOT/'FINAL_RESULTS.json').read_text())
    if not final['final_test_complete'] or not final['raw_trajectory_audit_passed']:
        raise ValueError('Cannot package incomplete or unaudited results')
    target = ROOT.parent/'comp0245-dynamic-pd-model.zip'
    if target.exists():
        raise FileExistsError('Never overwrite an existing package')
    baseline = json.loads((ROOT/'baseline/release_manifest.json').read_text())
    names = {'baseline/'+n for n in baseline['file_sha256']}
    names.add('baseline/release_manifest.json')
    for directory, pattern in [('dynamic_pd', '*.py'), ('tools', '*.py'),
                               ('runs/full', '*.json'), ('runs/full', '*.csv'),
                               ('figures', '*.png'), ('outputs/development', '*.json'),
                               ('outputs/final', '*.json')]:
        names.update(str(p.relative_to(ROOT)) for p in (ROOT/directory).glob(pattern))
    names.add('runs/full/best_model.pt')
    for pattern in ('*.md', '*AUDIT.json'):
        names.update(p.name for p in ROOT.glob(pattern))
    names.update(('.gitignore', 'protocol.json', 'selection.json', 'local_paths.json', 'FINAL_RESULTS.json'))
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        selected_checkpoint_sha256=selection['checkpoint_sha256'],
        original_release_not_overwritten=True,
        included='Complete selected inference/training/evaluation source, frozen parent/gain models, split definitions, result records and docs',
        excluded='Raw expert data, feature caches, raw rollout NPZ/log archives, Python environments, upstream Git checkout/meshes, smoke checkpoints',
        dependencies_fetched_by='python baseline/tools/setup_dependencies.py',
        result_source='FINAL_RESULTS.json; raw local archive was audited before packaging',
        file_sha256={name:sha(ROOT/name) for name in sorted(names)})
    write(ROOT/'experiment_manifest.json', manifest)
    names.add('experiment_manifest.json')
    with zipfile.ZipFile(target, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted(names):
            archive.write(ROOT/name, arcname='comp0245-dynamic-pd-model/'+name)
    with zipfile.ZipFile(target) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC check failed')
        for name, expected in manifest['file_sha256'].items():
            import hashlib
            if hashlib.sha256(archive.read('comp0245-dynamic-pd-model/'+name)).hexdigest()!=expected:
                raise ValueError('Archived bytes differ: '+name)
    receipt = dict(path=str(target), bytes=target.stat().st_size, sha256=sha(target),
                   files=len(names), archived_bytes_verified=True)
    write(ROOT/'PACKAGE_RECEIPT.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
