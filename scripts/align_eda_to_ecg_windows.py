#!/usr/bin/env python3
"""Align strict EDA windows to ECG CardioMind windows by timestamp.

Creates an EDA output folder where each subject has exactly the same window timeline
as the ECG processed data. This resolves the known S13 5-window mismatch and ensures
strict multimodal parity for fusion experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _load(path: Path) -> dict:
    return torch.load(path, weights_only=False)


def _binary_from_eda_labels(labels: np.ndarray, stress_label: int = 2) -> np.ndarray:
    return (labels == stress_label).astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Align EDA windows to ECG windows by timestamp")
    parser.add_argument("--eda-root", default="data/processed_eda_strict_ratio")
    parser.add_argument("--ecg-root", default="data/processed_cardiomind_strict_ratio")
    parser.add_argument("--out-root", default="data/processed_eda_strict_ratio_aligned_to_ecg")
    parser.add_argument("--stress-label", type=int, default=2)
    args = parser.parse_args()

    eda_root = Path(args.eda_root)
    ecg_root = Path(args.ecg_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    eda_files = {p.stem: p for p in eda_root.glob("S*.pt")}
    ecg_files = {p.stem: p for p in ecg_root.glob("S*.pt")}
    subjects = sorted(set(eda_files.keys()) & set(ecg_files.keys()), key=lambda s: int(s[1:]))

    if not subjects:
        raise FileNotFoundError("No overlapping S*.pt subjects between EDA and ECG roots")

    print(f"Aligning {len(subjects)} subjects")
    total_dropped = 0

    for sid in subjects:
        de = _load(eda_files[sid])
        dc = _load(ecg_files[sid])

        t_e = np.asarray(de["timestamps_sec"], dtype=np.float64)
        t_c = np.asarray(dc["timestamps_sec"], dtype=np.float64)

        ecg_ts_set = set(t_c.tolist())
        keep_mask_eda = np.asarray([t in ecg_ts_set for t in t_e], dtype=bool)

        X_eda = np.asarray(de["cardiac_features"], dtype=np.float32)[keep_mask_eda]
        y_eda = np.asarray(de["labels"], dtype=np.int64)[keep_mask_eda]
        ts_eda = t_e[keep_mask_eda]

        order = {float(t): i for i, t in enumerate(ts_eda.tolist())}
        idx_in_eda = [order[float(t)] for t in t_c.tolist() if float(t) in order]

        X_aligned = X_eda[idx_in_eda]
        y_aligned = y_eda[idx_in_eda]
        ts_aligned = ts_eda[idx_in_eda]

        dropped = int(len(t_e) - len(ts_aligned))
        total_dropped += dropped

        if len(ts_aligned) != len(t_c):
            raise RuntimeError(
                f"{sid}: alignment failed; aligned_len={len(ts_aligned)} ecg_len={len(t_c)}"
            )

        if not np.array_equal(ts_aligned, t_c):
            raise RuntimeError(f"{sid}: timestamps not perfectly aligned after mapping")

        # Sanity: binary labels should match ECG labels after mapping
        y_bin = _binary_from_eda_labels(y_aligned, stress_label=args.stress_label)
        y_ecg = np.asarray(dc["labels"], dtype=np.int64)
        mismatch = int(np.sum(y_bin != y_ecg))

        out_path = out_root / f"{sid}.pt"
        torch.save(
            {
                "subject_id": de.get("subject_id", int(sid.lstrip("S"))),
                "cardiac_features": X_aligned,
                "somatic_features": np.zeros((X_aligned.shape[0], 0), dtype=np.float32),
                "cardiac_feature_names": de.get("cardiac_feature_names", []),
                "somatic_feature_names": [],
                "labels": y_aligned,
                "timestamps_sec": ts_aligned,
            },
            out_path,
        )

        print(
            f"{sid}: eda_in={len(t_e)} ecg={len(t_c)} aligned={len(ts_aligned)} dropped={dropped} "
            f"binary_mismatch_vs_ecg={mismatch}"
        )

    print(f"\nDone. Total dropped EDA windows: {total_dropped}")
    print(f"Saved aligned EDA files to: {out_root}")


if __name__ == "__main__":
    main()
