#!/usr/bin/env python3
"""Align raw BVP windows to ECG CardioMind windows by timestamp.

Creates a BVP output folder where each subject has exactly the same window timeline
as the ECG processed data, resolving any window count mismatches and ensuring
strict multimodal parity for fusion experiments.

Mirrors scripts/align_eda_to_ecg_windows.py but handles the BVP raw-window format.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _load(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def _binary_from_raw_labels(labels: np.ndarray, stress_label: int = 2) -> np.ndarray:
    return (labels == stress_label).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align BVP windows to ECG windows by timestamp")
    parser.add_argument("--bvp-root", default="data/processed_bvp_raw")
    parser.add_argument("--ecg-root", default="data/processed_cardiomind_strict_ratio")
    parser.add_argument("--out-root", default="data/processed_bvp_raw_aligned_to_ecg")
    parser.add_argument("--stress-label", type=int, default=2)
    args = parser.parse_args()

    bvp_root = Path(args.bvp_root)
    ecg_root = Path(args.ecg_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    bvp_files = {p.stem: p for p in bvp_root.glob("S*.pt")}
    ecg_files = {p.stem: p for p in ecg_root.glob("S*.pt")}
    subjects = sorted(
        set(bvp_files.keys()) & set(ecg_files.keys()),
        key=lambda s: int(s[1:]),
    )

    if not subjects:
        raise FileNotFoundError("No overlapping S*.pt subjects between BVP and ECG roots")

    print(f"Aligning {len(subjects)} subjects")
    total_dropped = 0

    for sid in subjects:
        db = _load(bvp_files[sid])
        dc = _load(ecg_files[sid])

        t_b = np.asarray(db["timestamps_sec"], dtype=np.float64)
        t_c = np.asarray(dc["timestamps_sec"], dtype=np.float64)

        # Build mapping: for each ECG timestamp, find the closest BVP timestamp
        ecg_ts_set = set(t_c.tolist())
        keep_mask = np.asarray([t in ecg_ts_set for t in t_b], dtype=bool)

        bvp_windows = np.asarray(db["bvp_windows"], dtype=np.float32)[keep_mask]
        y_bvp = np.asarray(db["labels"], dtype=np.int64)[keep_mask]
        ts_bvp = t_b[keep_mask]

        # Re-order to match ECG timestamp order
        order = {float(t): i for i, t in enumerate(ts_bvp.tolist())}
        idx = [order[float(t)] for t in t_c.tolist() if float(t) in order]

        bvp_aligned = bvp_windows[idx]
        y_aligned = y_bvp[idx]
        ts_aligned = ts_bvp[idx]

        dropped = int(len(t_b) - len(ts_aligned))
        total_dropped += dropped

        if len(ts_aligned) != len(t_c):
            raise RuntimeError(
                f"{sid}: alignment failed; aligned_len={len(ts_aligned)} ecg_len={len(t_c)}"
            )

        if not np.array_equal(ts_aligned, t_c):
            raise RuntimeError(f"{sid}: timestamps not perfectly aligned after mapping")

        # Sanity: binary labels should match ECG labels after mapping
        y_bin = _binary_from_raw_labels(y_aligned, stress_label=args.stress_label)
        y_ecg = np.asarray(dc["labels"], dtype=np.int64)
        mismatch = int(np.sum(y_bin != y_ecg))

        out_path = out_root / f"{sid}.pt"
        torch.save(
            {
                "subject_id": db.get("subject_id", int(sid.lstrip("S"))),
                "bvp_windows": bvp_aligned,
                "labels": y_aligned,
                "timestamps_sec": ts_aligned,
            },
            out_path,
        )

        print(
            f"{sid}: bvp_in={len(t_b)} ecg={len(t_c)} aligned={len(ts_aligned)} "
            f"dropped={dropped} binary_mismatch_vs_ecg={mismatch}"
        )

    print(f"\nDone. Total dropped BVP windows: {total_dropped}")
    print(f"Saved aligned BVP files to: {out_root}")


if __name__ == "__main__":
    main()
