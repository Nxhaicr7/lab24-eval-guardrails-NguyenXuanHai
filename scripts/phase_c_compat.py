"""Import wrappers for phase-c modules in a hyphenated directory."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent.parent


def _load(module_name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


input_guard_module = _load("phase_c_input_guard", "phase-c/input_guard.py")
output_guard_module = _load("phase_c_output_guard", "phase-c/output_guard.py")

InputGuard = input_guard_module.InputGuard
OutputGuard = output_guard_module.OutputGuard
graceful_refusal = input_guard_module.graceful_refusal
