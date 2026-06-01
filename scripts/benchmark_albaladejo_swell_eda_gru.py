#!/usr/bin/env python3
"""LOSO GRU on Albaladejo-style SWELL EDA feature NPZs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.benchmark_albaladejo_swell_gru import main


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend(
            [
                "--data-root",
                "data/processed_swell_eda_albaladejo_w20_s5",
                "--out-csv",
                "runs/albaladejo_swell_eda_gru_loso_w20_s5.csv",
            ]
        )
    main()
