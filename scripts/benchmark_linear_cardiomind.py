#!/usr/bin/env python3
"""CardioMind linear replication benchmark (LOSO).

Replicates a linear baseline on the CardioMind-style 12 ECG-HRV features:
- Input: `data/processed_cardiomind/S*.pt` cardiac_features only (12 dims)
- Labels: binary stress (1) vs non-stress (0)
- Split: strict leave-one-subject-out (LOSO)
- Preprocessing: StandardScaler fit on train folds only
- Model: LogisticRegression (linear decision boundary)
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


def _load_subject_pt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = torch.load(path, weights_only=False)
    X = np.asarray(d["cardiac_features"], dtype=np.float32)
    y = np.asarray(d["labels"], dtype=np.int64)

    # Guard against any NaN/Inf windows
    valid = np.isfinite(X).all(axis=1)
    X = X[valid]
    y = y[valid]
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Replicate CardioMind linear baseline on 12 ECG features")
    parser.add_argument("--data-root", default="data/processed_cardiomind")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--out-csv", default="runs/linear_cardiomind_loso.csv")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    files = sorted(data_root.glob("S*.pt"), key=lambda p: int(p.stem.lstrip("S")))
    if not files:
        raise FileNotFoundError(f"No CardioMind processed files found in {data_root}")

    subjects = [int(p.stem.lstrip("S")) for p in files]
    print(
        f"CardioMind Linear LOSO | subjects={subjects} | "
        f"class_weight={args.class_weight} | C={args.C}"
    )

    subj_data = {sid: _load_subject_pt(p) for sid, p in zip(subjects, files)}
    rows = []
    cw = None if args.class_weight == "none" else "balanced"

    for held_out in subjects:
        X_te, y_te = subj_data[held_out]

        X_tr_list, y_tr_list = [], []
        for sid in subjects:
            if sid == held_out:
                continue
            X, y = subj_data[sid]
            X_tr_list.append(X)
            y_tr_list.append(y)

        X_tr = np.concatenate(X_tr_list, axis=0)
        y_tr = np.concatenate(y_tr_list, axis=0)

        clf = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        C=args.C,
                        max_iter=args.max_iter,
                        solver="lbfgs",
                        random_state=42,
                        class_weight=cw,
                    ),
                ),
            ]
        )
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_te)

        row = {
            "subject": held_out,
            "accuracy": float(accuracy_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
            "n_samples": int(len(y_te)),
            "stress_pct": float(np.mean(y_te == 1) * 100.0),
        }
        rows.append(row)
        print(
            f"  S{held_out}: acc={row['accuracy']:.3f} f1={row['f1']:.3f} "
            f"prec={row['precision']:.3f} rec={row['recall']:.3f} "
            f"(n={row['n_samples']}, stress={row['stress_pct']:.1f}%)"
        )

    acc = np.array([r["accuracy"] for r in rows], dtype=np.float64)
    f1 = np.array([r["f1"] for r in rows], dtype=np.float64)
    pr = np.array([r["precision"] for r in rows], dtype=np.float64)
    rc = np.array([r["recall"] for r in rows], dtype=np.float64)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        fieldnames = ["subject", "accuracy", "f1", "precision", "recall", "n_samples", "stress_pct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "subject": "MEAN±STD",
                "accuracy": f"{acc.mean():.6f}±{acc.std():.6f}",
                "f1": f"{f1.mean():.6f}±{f1.std():.6f}",
                "precision": f"{pr.mean():.6f}±{pr.std():.6f}",
                "recall": f"{rc.mean():.6f}±{rc.std():.6f}",
                "n_samples": int(np.mean([r['n_samples'] for r in rows])),
                "stress_pct": f"{np.mean([r['stress_pct'] for r in rows]):.3f}",
            }
        )

    print(f"mean_accuracy: {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"mean_f1: {f1.mean():.3f} ± {f1.std():.3f}")
    print(f"mean_precision: {pr.mean():.3f} ± {pr.std():.3f}")
    print(f"mean_recall: {rc.mean():.3f} ± {rc.std():.3f}")
    print(f"saved_csv: {out_csv}")


if __name__ == "__main__":
    main()
