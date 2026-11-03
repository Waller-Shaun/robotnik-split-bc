"""CLI path adapter for the unchanged original low-height collector."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "LAB_SESSION_COMP_0245_2026_PUBLIC/final/robotnik_arm_task"
sys.path.insert(0, str(CODE))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-split", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    import collect_expert_ring_lowz_supplement as original
    original.BASE_SPLIT = args.base_split.resolve()
    sys.argv = [str(CODE / "collect_expert_ring_lowz_supplement.py"), *remaining]
    original.main()


if __name__ == "__main__":
    main()
