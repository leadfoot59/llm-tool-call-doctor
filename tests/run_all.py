#!/usr/bin/env python3
"""Run the whole test suite with the current interpreter.

    python3 tests/run_all.py [-v]

Equivalent to ``python3 -m toolcalldoctor selftest``.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for extra in (str(ROOT), str(ROOT / "tests")):
    if extra not in sys.path:
        sys.path.insert(0, extra)


def main() -> int:
    verbosity = 2 if "-v" in sys.argv or "--verbose" in sys.argv else 1
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
