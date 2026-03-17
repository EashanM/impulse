#!/usr/bin/env python3
"""Hybrid linear LOSO baseline: CardioMind-12 + DSP summaries.

Fuses per-timestamp CardioMind ECG-HRV features with aligned DSP window
summary features (from IHR/EDA phasic windows), then evaluates LOSO with
logistic regression.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _dsp_window_summaries(Xw: np.ndarray) -> np.ndarray:
    """Compute compact DSP summaries from windows (N, L, 2)."""
    ihr = Xw[:, :, 0]
    eda = Xw[:, :, 1]
    out = np.column_stack(
        [
            ihr.mean(axis=1),
            ihr.std(axis=1),
            np.diff(ihr, axis=1).mean(axis=1),
            eda.mean(axis=1),
            np.abs(eda).mean(axis=1),
            (eda * eda).mean(axis=1),
            eda.std(axis=1),
        ]
    )
    return out.astype(np.float32)


def _align_nearest(ref_ts: np.ndarray, src_ts: np.ndarray, max_delta: float = 1.1) -> tuple[np.ndarray, np.ndarray]:
    """Map each ref timestamp to nearest src index within tolerance."""
    pos = np.searchsorted(src_ts, ref_ts)
    pos = np.clip(pos, 0, len(src_ts) - 1)
    left = np.clip(pos - 1, 0, len(src_ts) - 1)

    d_pos = np.abs(src_ts[pos] - ref_ts)
    d_left = np.abs(src_ts[left] - ref_ts)
    choose_left = d_left < d_pos
    idx = pos.copy()
    idx[choose_left] = left[choose_left]

    ok = np.abs(src_ts[idx] - ref_ts) <= max_delta
    return idx, ok


def _load_subject_hybrid(cardiomind_path: Path, dsp_path: Path, use_train_mask: bool) -> tuple[np.ndarray, np.ndarray]:
    cm = torch.load(cardiomind_path, weights_only=False)
    X_cm = np.asarray(cm["cardiac_features"], dtype=np.float32)  # (Tcm, 12)
    y_cm = np.asarray(cm["labels"], dtype=np.int64)
    t_cm = np.asarray(cm["timestamps_sec"], dtype=np.float64)

    d = np.load(dsp_path)
    Xw = d["windows"].astype(np.float32)
    tw = d["window_start_sec"].astype(np.float64)
    if use_train_mask and "window_train_mask" in d:
        m = d["window_train_mask"].astype(bool)
        Xw = Xw[m]
        tw = tw[m]

    X_dsp = _dsp_window_summaries(Xw)  # (Nd, 7)
    idx, ok = _align_nearest(t_cm, tw, max_delta=1.1)

    X = np.concatenate([X_cm[ok], X_dsp[idx[ok]]], axis=1)
    y = y_cm[ok]
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid linear LOSO: CardioMind + DSP")
    parser.add_argument("--cardiomind-root", default="data/processed_cardiomind")
    parser.add_argument("--dsp-root", default="data/processed_wesad_dsp")
    parser.add_argument("--use-train-mask", action="store_true")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--out-csv", default="runs/linear_hybrid_loso.csv")
    args = parser.parse_args()

    cm_root = Path(args.cardiomind_root)
    dsp_root = Path(args.dsp_root)

    cm_files = {p.stem: p for p in cm_root.glob("S*.pt")}
    dsp_files = {p.stem.replace("_dsp", ""): p for p in dsp_root.glob("S*_dsp.npz")}
    subjects = sorted(set(cm_files.keys()) & set(dsp_files.keys()), key=lambda s: int(s.lstrip("S")))
    if not subjects:
        raise FileNotFoundError("No overlapping subjects between cardiomind and dsp roots")

    print(f"Hybrid Linear LOSO | subjects={[int(s.lstrip('S')) for s in subjects]} | class_weight={args.class_weight}")

    rows = []
    cw = None if args.class_weight == "none" else "balanced"

    subj_data = {}
    for sid in subjects:
        X, y = _load_subject_hybrid(cm_files[sid], dsp_files[sid], use_train_mask=args.use_train_mask)
        subj_data[sid] = (X, y)

    for held_out in subjects:
        X_tr_list, y_tr_list = [], []
        X_te, y_te = subj_data[held_out]
        for sid in subjects:
            if sid == held_out:
                continue
            X, y = subj_data[sid]
            X_tr_list.append(X)
            y_tr_list.append(y)

        X_tr = np.concatenate(X_tr_list, axis=0)
        y_tr = np.concatenate(y_tr_list, axis=0)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=2000, random_state=42, class_weight=cw)),
            ]
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        r = {
            "subject": int(held_out.lstrip("S")),
            "accuracy": float(accuracy_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
        }
        rows.append(r)
        print(f"  {held_out}: acc={r['accuracy']:.3f} f1={r['f1']:.3f} prec={r['precision']:.3f} rec={r['recall']:.3f}")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subject", "accuracy", "f1", "precision", "recall"])
        writer.writeheader()
        writer.writerows(rows)

        acc = np.array([r["accuracy"] for r in rows], dtype=np.float64)
        f1 = np.array([r["f1"] for r in rows], dtype=np.float64)
        pr = np.array([r["precision"] for r in rows], dtype=np.float64)
        rc = np.array([r["recall"] for r in rows], dtype=np.float64)
        writer.writerow(
            {
                "subject": "MEAN±STD",
                "accuracy": f"{acc.mean():.6f}±{acc.std():.6f}",
                "f1": f"{f1.mean():.6f}±{f1.std():.6f}",
                "precision": f"{pr.mean():.6f}±{pr.std():.6f}",
                "recall": f"{rc.mean():.6f}±{rc.std():.6f}",
            }
        )

    print(f"mean_accuracy: {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"mean_f1: {f1.mean():.3f} ± {f1.std():.3f}")
    print(f"mean_precision: {pr.mean():.3f} ± {pr.std():.3f}")
    print(f"mean_recall: {rc.mean():.3f} ± {rc.std():.3f}")
    print(f"saved_csv: {out_csv}")


if __name__ == "__main__":
    main()
