"""Fetch the pinned upstream simulator, never overwrite an existing checkout."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'release_manifest.json').read_text())
    pin = manifest['dependencies']
    target = ROOT / 'external/RoboEnv'
    if target.exists():
        actual = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != pin['roboenv_commit']:
            raise ValueError('Existing simulator commit differs; keep it intact and use a clean release folder')
    else:
        target.parent.mkdir(exist_ok=True)
        subprocess.run(['git', 'clone', '--no-checkout', pin['roboenv_url'], str(target)], check=True)
        subprocess.run(['git', '-C', str(target), 'checkout', pin['roboenv_commit']], check=True)
    subprocess.run(['git', '-C', str(target), 'submodule', 'update', '--init', '--recursive'], check=True)
    actual = subprocess.check_output(['git', '-C', str(target / 'simulation_and_control'), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != pin['simulation_and_control_commit']:
        raise ValueError('Simulation submodule pin differs')
    print('Pinned simulator ready. Next:')
    print('pixi install --manifest-path external/RoboEnv/pixi.toml -e ml --locked')


if __name__ == '__main__':
    main()
