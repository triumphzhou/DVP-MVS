#!/usr/bin/env python3
"""Compatibility entry point; the maintained implementation is in mvs_process."""

import os
import runpy
from pathlib import Path


HERE = Path(__file__).resolve().parent
os.environ.setdefault("DVP_RUN_ROOT", str(HERE))
runpy.run_path(str(HERE.parent / "mvs_process" / "prepare_dvp_scene.py"), run_name="__main__")
