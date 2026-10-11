"""Compatibility entry point for tools/build_portable_cpp.py."""

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools/build_portable_cpp.py"), run_name="__main__"
)
