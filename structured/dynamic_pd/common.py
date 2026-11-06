import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'baseline'
SOURCE = BASE / 'LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task/tcn'
for path in (BASE, SOURCE, SOURCE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temp.replace(path)


def check_baseline():
    manifest = json.loads((BASE / 'release_manifest.json').read_text())
    for name, expected in manifest['file_sha256'].items():
        if sha(BASE / name) != expected:
            raise ValueError('Experiment baseline changed: ' + name)
    for name, expected in manifest['dependencies']['file_sha256'].items():
        if sha(BASE / name) != expected:
            raise ValueError('Experiment simulator assets changed: ' + name)
    protocol = json.loads((ROOT / 'protocol.json').read_text())
    for name, expected in protocol['original_frozen_file_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Original frozen model/source changed: ' + name)
    if sha(ROOT.parent / 'comp0245-tcn-release.zip') != protocol['original_submission_zip_sha256']:
        raise ValueError('Original submission ZIP changed')


def load_fixed():
    from run_model import load_controller
    return load_controller('tcn')
