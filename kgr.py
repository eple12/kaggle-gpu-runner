#!/usr/bin/env python3
"""Run kgr without installing it, e.g. from a git submodule:

    python vendor/kaggle-gpu-runner/kgr.py push my_run
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from kgr.cli import main  # noqa: E402

sys.exit(main())
