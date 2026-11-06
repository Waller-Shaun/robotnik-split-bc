"""One-command gain, reset, display and cache-RNG regression tests."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dynamic_pd.common import BASE


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        ['dynamic_pd.test_dynamic_pd', 'test_fast_gui_demo', 'test_retrain_gain'])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
