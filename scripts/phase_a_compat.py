"""Import-friendly wrapper around `phase-a/run_eval.py`."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


MODULE_PATH = Path(__file__).resolve().parent.parent / "phase-a" / "run_eval.py"
SPEC = importlib.util.spec_from_file_location("phase_a_run_eval", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
sys.modules["phase_a_run_eval"] = MODULE
SPEC.loader.exec_module(MODULE)
main = MODULE.main
