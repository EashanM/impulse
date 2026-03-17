#!/usr/bin/env python3
"""EDA strict linear benchmark (LOSO), modeled after CardioMind linear benchmark.

Input
-----
Processed strict EDA files from scripts/preprocess_eda_strict.py:
  data/processed_eda_strict_ratio/S*.pt

Supports two label modes:
- binary: stress (label=2) vs non-stress (labels 1,3,4)
- multiclass: retain labels {1,2,3,4}
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


def _load_subject_pt(path: Path, label_mode: str) -> tuple[np.ndarray, np.ndarray]:
    d = torch.load(path, weights_only=False)
    X = np.asarray(d["cardiac_features"], dtype=np.float32)
    y_raw = np.asarray(d["labels"], dtype=np.int64)

    valid = np.isfinite(X).all(axis=1)
    X = X[valid]
    y_raw = y_raw[valid]

    if label_mode == "binary":
        y = (y_raw == 2).astype(np.int64)
    else:
        keep = np.isin(y_raw, [1, 2, 3, 4])
        X = X[keep]
        y = y_raw[keep]

    return X, y


def _safe_pct(y: np.ndarray, cls: int) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.mean(y == cls) * 100.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="LOSO linear benchmark on strict EDA features")
    parser.add_argument("--data-root", default="data/processed_eda_strict_ratio")
    parser.add_argument("--label-mode", choices=["binary", "multiclass"], default="binary")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=2000)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--out-csv", default="runs/linear_eda_strict_loso.csv")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    files = sorted(data_root.glob("S*.pt"), key=lambda p: int(p.stem.lstrip("S")))
    if not files:
        raise FileNotFoundError(f"No processed EDA files found in {data_root}")

    subjects = [int(p.stem.lstrip("S")) for p in files]
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
        files = [data_root / f"S{sid}.pt" for sid in subjects]

    print(
        f"EDA Linear LOSO | subjects={subjects} | label_mode={args.label_mode} | "
        f"class_weight={args.class_weight} | C={args.C}"
    )

    subj_data = {sid: _load_subject_pt(p, label_mode=args.label_mode) for sid, p in zip(subjects, files)}
    rows = []
    cw = None if args.class_weight == "none" else "balanced"

    multi = args.label_mode == "multiclass"
    avg = "macro" if multi else "binary"

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

        if len(np.unique(y_tr)) < (2 if not multi else 4):
            print(f"  S{held_out}: skip (insufficient classes in train)")
            continue

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
            "f1": float(f1_score(y_te, pred, average=avg, zero_division=0)),
            "precision": float(precision_score(y_te, pred, average=avg, zero_division=0)),
            "recall": float(recall_score(y_te, pred, average=avg, zero_division=0)),
            "n_samples": int(len(y_te)),
        }

        if multi:
            row.update(
                {
                    "lbl1_pct": _safe_pct(y_te, 1),
                    "lbl2_pct": _safe_pct(y_te, 2),
                    "lbl3_pct": _safe_pct(y_te, 3),
                    "lbl4_pct": _safe_pct(y_te, 4),
                }
            )
            print(
                f"  S{held_out}: acc={row['accuracy']:.3f} f1_macro={row['f1']:.3f} "
                f"prec_macro={row['precision']:.3f} rec_macro={row['recall']:.3f} (n={row['n_samples']})"
            )
        else:
            row["stress_pct"] = _safe_pct(y_te, 1)
            print(
                f"  S{held_out}: acc={row['accuracy']:.3f} f1={row['f1']:.3f} "
                f"prec={row['precision']:.3f} rec={row['recall']:.3f} "
                f"(n={row['n_samples']}, stress={row['stress_pct']:.1f}%)"
            )

        rows.append(row)

    if not rows:
        print("No valid LOSO rows produced.")
        return

    acc = np.asarray([r["accuracy"] for r in rows], dtype=np.float64)
    f1 = np.asarray([r["f1"] for r in rows], dtype=np.float64)
    pr = np.asarray([r["precision"] for r in rows], dtype=np.float64)
    rc = np.asarray([r["recall"] for r in rows], dtype=np.float64)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    if multi:
        fieldnames = [
            "subject",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "n_samples",
            "lbl1_pct",
            "lbl2_pct",
            "lbl3_pct",
            "lbl4_pct",
        ]
    else:
        fieldnames = [
            "subject",
            "accuracy",
            "f1",
            "precision",
            "recall",
            "n_samples",
            "stress_pct",
        ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        summary = {
            "subject": "MEAN±STD",
            "accuracy": f"{acc.mean():.6f}±{acc.std():.6f}",
            "f1": f"{f1.mean():.6f}±{f1.std():.6f}",
            "precision": f"{pr.mean():.6f}±{pr.std():.6f}",
            "recall": f"{rc.mean():.6f}±{rc.std():.6f}",
            "n_samples": int(np.mean([r["n_samples"] for r in rows])),
        }
        if multi:
            summary.update(
                {
                    "lbl1_pct": f"{np.mean([r['lbl1_pct'] for r in rows]):.3f}",
                    "lbl2_pct": f"{np.mean([r['lbl2_pct'] for r in rows]):.3f}",
                    "lbl3_pct": f"{np.mean([r['lbl3_pct'] for r in rows]):.3f}",
                    "lbl4_pct": f"{np.mean([r['lbl4_pct'] for r in rows]):.3f}",
                }
            )
        else:
            summary["stress_pct"] = f"{np.mean([r['stress_pct'] for r in rows]):.3f}"

        writer.writerow(summary)

    print(f"mean_accuracy: {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"mean_f1: {f1.mean():.3f} ± {f1.std():.3f}")
    print(f"mean_precision: {pr.mean():.3f} ± {pr.std():.3f}")
    print(f"mean_recall: {rc.mean():.3f} ± {rc.std():.3f}")
    print(f"saved_csv: {out_csv}")


if __name__ == "__main__":
    main()
