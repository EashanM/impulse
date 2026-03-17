#!/usr/bin/env python3
"""Linear baseline (LOSO) on WESAD DSP windows.

Uses logistic regression on flattened window tensors from
data/processed_wesad_dsp/S*_dsp.npz.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _load_npz_windows(npz_path: Path, use_train_mask: bool) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path)
    X = d["windows"].astype(np.float32)
    y = d["window_labels"].astype(np.int64)
    if use_train_mask and "window_train_mask" in d:
        m = d["window_train_mask"].astype(bool)
        X = X[m]
        y = y[m]
    X = X.reshape(len(X), -1)
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="Linear LOSO baseline on DSP windows")
    parser.add_argument("--data-root", default="data/processed_wesad_dsp")
    parser.add_argument("--use-train-mask", action="store_true", help="Use window_train_mask filtering if available")
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--out-csv", default="runs/linear_dsp_loso.csv")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    files = sorted(data_root.glob("S*_dsp.npz"))
    if not files:
        raise FileNotFoundError(f"No DSP files found in {data_root}")
    if args.max_subjects is not None:
        files = files[: args.max_subjects]

    subjects = [int(p.stem.split("_")[0].lstrip("S")) for p in files]
    print(f"Linear DSP LOSO | subjects={subjects} | class_weight={args.class_weight}")

    rows: list[dict] = []
    cw = None if args.class_weight == "none" else "balanced"

    for held_out in subjects:
        X_tr_list, y_tr_list = [], []
        X_te = None
        y_te = None

        for sid, p in zip(subjects, files):
            X, y = _load_npz_windows(p, use_train_mask=args.use_train_mask)
            if sid == held_out:
                X_te, y_te = X, y
            else:
                X_tr_list.append(X)
                y_tr_list.append(y)

        X_tr = np.concatenate(X_tr_list, axis=0)
        y_tr = np.concatenate(y_tr_list, axis=0)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=42,
                        class_weight=cw,
                    ),
                ),
            ]
        )
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)

        r = {
            "subject": held_out,
            "accuracy": float(accuracy_score(y_te, pred)),
            "f1": float(f1_score(y_te, pred, zero_division=0)),
            "precision": float(precision_score(y_te, pred, zero_division=0)),
            "recall": float(recall_score(y_te, pred, zero_division=0)),
        }
        rows.append(r)
        print(
            f"  S{held_out}: acc={r['accuracy']:.3f} f1={r['f1']:.3f} "
            f"prec={r['precision']:.3f} rec={r['recall']:.3f}"
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        fieldnames = ["subject", "accuracy", "f1", "precision", "recall"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
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
