#!/usr/bin/env python3
"""
Build per-subject `.pt` files from SWELL-KW minute physiology features.

Reads the dataset-provided CSV (HR, RMSSD, SCL per minute), applies the binary
label rule in `src/data/swell_labels.py`, and writes `S{subject_id}.pt` files so
existing LOSO scripts that glob `data_root/S*.pt` can be pointed at `--data-root`.

Use the official feature CSV (has ``Condition`` labels). The MatLabTable minute export under
``2 - Minute data/`` is physiology-only (no stress labels).

Example:
    uv run python scripts/preprocess_swell.py \\
        --csv \"data/raw/SWELL/3 - Feature dataset/per sensor/D - Physiology features (HR_HRV_SCL - final).csv\" \\
        --out-dir data/processed_swell

Drop rest minutes (negative class = neutral ``N`` only vs stress ``T``/``I``)::

    uv run python scripts/preprocess_swell.py \\
        --csv \"data/raw/SWELL/3 - Feature dataset/per sensor/D - Physiology features (HR_HRV_SCL - final).csv\" \\
        --exclude-rest \\
        --out-dir data/processed_swell_no_rest
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.swell_labels import (
    clean_physiology_values,
    condition_to_binary_label,
    condition_to_code,
    impute_rowwise_ffill_zero,
)


def pp_to_subject_id(pp: str) -> int:
    """Map `PP12` -> 12 for filenames like `S12.pt` (matches `S*.pt` benchmarks)."""
    s = str(pp).strip()
    if not s.upper().startswith("PP"):
        raise ValueError(f"Expected PP-prefixed id, got {pp!r}")
    return int(s[2:])


def parse_swell_timestamp_to_epoch_sec(ts: str) -> float:
    """
    Parse SWELL timestamp strings like `20120918T131600000`.

    Interpreted as local naive datetime: YYYYMMDD + THHMMSS + milliseconds (3 digits).
    """
    ts = str(ts).strip()
    if len(ts) < 15:
        return float("nan")
    head = ts[:15]  # YYYYMMDDTHHMMSS
    tail = ts[15:18] if len(ts) >= 18 else "000"
    base = datetime.strptime(head, "%Y%m%dT%H%M%S")
    ms = int(tail.ljust(3, "0")[:3])
    return base.timestamp() + ms / 1000.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess SWELL physiology CSV to S*.pt tensors")
    parser.add_argument(
        "--csv",
        default="data/raw/SWELL/3 - Feature dataset/per sensor/D - Physiology features (HR_HRV_SCL - final).csv",
        help="Path to D - Physiology features (HR_HRV_SCL - final).csv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed_swell",
        help="Output directory for S{n}.pt and subject_id_map.json",
    )
    parser.add_argument(
        "--exclude-rest",
        action="store_true",
        help="Drop minute rows where Condition is rest (R). Negative class is neutral (N) only vs stress (T,I).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]

    required = {"PP", "C", "Condition", "timestamp", "HR", "RMSSD", "SCL"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.exclude_rest:
        print("preprocess_swell: --exclude-rest (dropping Condition=R minute rows)")

    id_map: dict[str, str] = {}
    n_written = 0

    for pp in sorted(df["PP"].dropna().unique(), key=lambda x: pp_to_subject_id(str(x))):
        g = df[df["PP"] == pp].copy()
        g = g.sort_values("timestamp")
        if args.exclude_rest:
            g = g[g["Condition"].astype(str).str.strip().str.upper() != "R"].copy()
        if g.empty:
            print(f"  skip {pp}: no rows after --exclude-rest")
            continue

        sid = pp_to_subject_id(str(pp))
        id_map[str(sid)] = str(pp)

        hr = clean_physiology_values(g["HR"].to_numpy())
        rmssd = clean_physiology_values(g["RMSSD"].to_numpy())
        scl = clean_physiology_values(g["SCL"].to_numpy())

        X = np.column_stack([hr, rmssd, scl]).astype(np.float64)
        X = impute_rowwise_ffill_zero(X)

        conditions = g["Condition"].astype(str).str.strip().str.upper()
        y_list: list[int] = []
        code_list: list[int] = []
        for c in conditions:
            y_list.append(condition_to_binary_label(c))
            code_list.append(condition_to_code(c))

        y = np.asarray(y_list, dtype=np.int64)
        condition_code = np.asarray(code_list, dtype=np.int32)
        block_c = g["C"].to_numpy(dtype=np.int32)

        ts_raw = g["timestamp"].astype(str)
        epoch = np.array([parse_swell_timestamp_to_epoch_sec(t) for t in ts_raw], dtype=np.float64)
        if np.isfinite(epoch).any():
            t0 = float(np.nanmin(epoch))
            timestamps_sec = np.where(np.isfinite(epoch), epoch - t0, np.arange(len(epoch), dtype=np.float64))
        else:
            timestamps_sec = np.arange(len(g), dtype=np.float64) * 60.0

        out_path = out_dir / f"S{sid}.pt"
        torch.save(
            {
                "subject_id": sid,
                "swell_pp": str(pp),
                "exclude_rest": bool(args.exclude_rest),
                "cardiac_features": X.astype(np.float32),
                "somatic_features": np.zeros((len(X), 0), dtype=np.float32),
                "cardiac_feature_names": ["HR", "RMSSD", "SCL"],
                "somatic_feature_names": [],
                "labels": y,
                "condition_code": condition_code,
                "block_c": block_c,
                "timestamps_sec": timestamps_sec.astype(np.float64),
            },
            out_path,
        )
        n_written += 1
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        print(f"  {pp} -> {out_path.name} | n={len(y)} binary1={n_pos} binary0={n_neg}")

    map_path = out_dir / "subject_id_map.json"
    if args.exclude_rest:
        legend = (
            "labels: 0=non-stress (N only; R rows dropped), 1=stress (T,I); "
            "condition_code R=0,N=1,T=2,I=3"
        )
    else:
        legend = "labels: 0=non-stress (N,R), 1=stress (T,I); condition_code R=0,N=1,T=2,I=3"
    map_path.write_text(
        json.dumps(
            {"swell_pp_by_S_id": id_map, "label_legend": legend, "exclude_rest": bool(args.exclude_rest)},
            indent=2,
        )
    )
    print(f"\nWrote {n_written} subjects to {out_dir}")
    print(f"Wrote id map: {map_path}")


if __name__ == "__main__":
    main()
