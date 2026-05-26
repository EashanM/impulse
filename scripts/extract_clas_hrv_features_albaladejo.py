#!/usr/bin/env python3
"""Backward-compatible wrapper: CLAS ECG HRV extraction (20 s / 5 s stride).

Prefer:
  uv run python scripts/extract_clas_features.py --modality ecg
  uv run python scripts/clas_preprocess_and_benchmark.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import _repo_root  # noqa: F401

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from extract_clas_features import main  # noqa: E402

if __name__ == "__main__":
    # Force ECG-only with legacy output path unless user passes flags
    if "--modality" not in sys.argv:
        sys.argv[1:1] = ["--modality", "ecg"]
    if "--processed-root" not in sys.argv and "-h" not in sys.argv and "--help" not in sys.argv:
        sys.argv.extend(
            ["--processed-root", "data/processed_clas_features", "--ecg-window-sec", "20", "--ecg-stride-sec", "5"]
        )
    main()
