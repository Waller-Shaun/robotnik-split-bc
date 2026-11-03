"""Create a local runtime copy from immutable portable artifacts."""
import argparse
from pathlib import Path
import shutil

from relocate import relocate, read, write

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roboenv", type=Path,
                        help="Optional existing RoboEnv; reuse its robot assets (not its trained runs)")
    args = parser.parse_args()
    target = ROOT / "RoboEnv"
    marker = target / ".robotnik_release_prepared.json"
    if marker.exists():
        raise FileExistsError("This release is already prepared here. Use the existing runtime; do not overwrite training runs.")
    source = (args.source_roboenv or target).resolve()
    robot_assets = source / "models/summit_xl_arm_description"
    if not robot_assets.is_dir():
        raise FileNotFoundError("Clone RoboEnv with submodules into ./RoboEnv first, or pass --source-roboenv.")
    snapshot = ROOT / "artifacts/RoboEnv"
    for path in snapshot.rglob("*"):
        if path.is_file() and path.relative_to(snapshot).parts[0] != "configs":
            dest = target / path.relative_to(snapshot)
            if dest.exists():
                raise FileExistsError(f"Refusing to overwrite {dest}")
    # The release's evaluation configuration is deliberate. Never silently replace a different one.
    config = target / "configs/robotnik_arm.json"
    expected = snapshot / "configs/robotnik_arm.json"
    if config.exists() and read(config) != read(expected):
        raise ValueError("Existing RoboEnv robotnik_arm.json differs. Use a clean pinned clone; keep your original environment unchanged.")
    for path in snapshot.rglob("*"):
        if path.is_file():
            dest = target / path.relative_to(snapshot)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
    # Only the curated metadata directories are traversed for relocation, not all files in upstream RoboEnv.
    # tree has no user-created experiments at this point (checked above).
    relocate_runtime(target)
    project = ROOT / "LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task"
    project_models = project / "models/summit_xl_arm_description"
    if project_models.exists():
        raise FileExistsError(project_models)
    shutil.copytree(robot_assets, project_models)
    # Existing local installs can run without cloning a second environment. Preserve evaluation model paths.
    if source != target:
        target_models = target / "models/summit_xl_arm_description"
        if not target_models.exists():
            shutil.copytree(robot_assets, target_models)
    write(marker, {"workspace": str(ROOT), "source_roboenv": str(source), "tensors_unchanged": True})
    print("Prepared:", target)
    print("Next: python tools/verify_release.py --runtime")


def relocate_runtime(target):
    # Limit rewrites to the frozen artifact snapshot using a temporary staging copy.
    import tempfile
    with tempfile.TemporaryDirectory(prefix="robotnik_metadata_") as temp:
        staging = Path(temp) / "RoboEnv"
        shutil.copytree(ROOT / "artifacts/RoboEnv", staging)
        relocate(staging, ROOT)
        for path in staging.rglob("*"):
            if path.is_file():
                dest = target / path.relative_to(staging)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, dest)


if __name__ == "__main__":
    main()
