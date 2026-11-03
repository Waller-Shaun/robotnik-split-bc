"""CLI path/epoch adapter; the original model and training loss stay unchanged."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task"
sys.path.insert(0, str(CODE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-path", type=Path, default=ROOT / "RoboEnv/bc_training_26d_noise_ring/data_split.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "RoboEnv/bc_training_26d_noise_ring_bodyframe")
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error("epochs must be positive")
    import train_bc_26d_noise_ring_bodyframe as original
    original.SPLIT_PATH = args.split_path.resolve()
    original.SOURCE_SPLIT_DIR = original.SPLIT_PATH.parent
    original.TRAINING_DIR = args.output_root.resolve()
    original.MAX_EPOCHS = args.epochs
    original.main()


if __name__ == "__main__":
    main()
