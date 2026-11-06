"""Export exactly the frozen split files; keep raw data outside ordinary Git."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from relocate import read, write

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-roboenv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--include-pretraining-near", action="store_true", help="Export all 931+200 trajectories; otherwise export current 998 only")
    args = parser.parse_args()
    source = args.source_roboenv.resolve()
    manifest = read(ROOT / "artifacts/RoboEnv/bc_training_arm_ring_r08_lowz_supplement/data_split.json")
    files = set(f for names in manifest["splits"].values() for f in names)
    if args.include_pretraining_near:
        full = read(ROOT / "artifacts/RoboEnv/bc_training_26d_noise_ring/data_split.json")
        files.update("expert_data_23d_noise_ring/" + f for names in full["splits"].values() for f in names)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "dataset_index.json").exists() or list(output.glob("dataset-part-*.zip")):
        raise FileExistsError("Choose an empty export directory")
    part, size, archive = 0, 0, None
    records, archives = [], []
    try:
        for name in sorted(files):
            relative = Path(name)
            path = (source / relative).resolve()
            if not path.is_relative_to(source) or not path.is_file():
                raise FileNotFoundError(path)
            length = path.stat().st_size
            if archive is None or size + length > 400 * 1024**2:
                if archive:
                    archive.close()
                part += 1
                archive_path = output / f"dataset-part-{part:02d}.zip"
                archive = zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_STORED, allowZip64=True)
                archives.append(archive_path.name)
                size = 0
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, arcname="RoboEnv/" + relative.as_posix())
            records.append({"path": "RoboEnv/" + relative.as_posix(), "bytes": length, "sha256": digest, "archive": archives[-1]})
            size += length
    finally:
        if archive:
            archive.close()
    # Collection plan is needed when rebuilding the supplementary merge.
    # Use the already published, portable plan rather than copying local absolute paths.
    with zipfile.ZipFile(output / archives[-1], "a") as z:
        z.write(ROOT / "artifacts/RoboEnv/expert_data_23d_noise_ring_lowz_supplement/collection_plan.json",
                "portable_collection_plan.json")
    def file_sha(path):
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    write(output / "dataset_index.json", {"trajectories": len(records), "include_pretraining_near": args.include_pretraining_near,
        "extract_to": "repository root; preserves RoboEnv subdirectories", "files": records,
        "archives": [{"name": name, "sha256": file_sha(output / name)} for name in archives]})
    print(f"Exported {len(records)} trajectories in {len(archives)} ZIP files to {output}")


if __name__ == "__main__":
    main()
