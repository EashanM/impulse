#!/usr/bin/env python3
"""LOSO: train encoder (linear, GRU on raw, or CNN+GRU), then logistic regression on embeddings.

Uses CLAS by_block segments under data/CLAS_Database/CLAS. Default binary labels:
high-vs-low load (see src.data.clas_dataset).

Example (quick smoke):
  python scripts/clas_loso_encoder_lr.py --encoder linear --max-subjects 8 \\
      --epochs 5 --device cpu
"""

from __future__ import annotations

import _repo_root  # noqa: F401

import argparse
import csv
from pathlib import Path

import numpy as np

from src.data.clas_dataset import DEFAULT_CLAS_ROOT
from src.training.clas_loso_core import resolve_device, run_loso_encoder_lr, seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description="CLAS LOSO encoder + logistic regression")
    parser.add_argument("--clas-root", type=Path, default=DEFAULT_CLAS_ROOT)
    parser.add_argument(
        "--encoder",
        choices=["linear", "gru", "cnn_gru", "lstm", "cnn_lstm"],
        default="linear",
    )
    parser.add_argument("--modality", choices=["ecg2", "ppg", "gsr", "accel3"], default="ppg")
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--stride-sec", type=float, default=8.0)
    parser.add_argument("--target-len", type=int, default=1024, help="Resampled length per window")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--min-quality", type=float, default=None)
    parser.add_argument("--quality-modality", choices=["ecg", "eda", "ppg"], default="ecg")
    parser.add_argument("--scale-z", action="store_true", help="StandardScaler on embeddings before LR")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--out-csv", type=Path, default=Path("runs/clas_loso_encoder_lr.csv"))
    args = parser.parse_args()

    seed_all(args.seed)
    device = resolve_device(args.device)

    fold_rows = run_loso_encoder_lr(
        clas_root=args.clas_root,
        modality=args.modality,
        encoder=args.encoder,
        window_sec=args.window_sec,
        stride_sec=args.stride_sec,
        target_len=args.target_len,
        embedding_dim=args.embedding_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        gru_hidden=args.gru_hidden,
        seed=args.seed,
        device=device,
        min_quality=args.min_quality,
        quality_modality=args.quality_modality,
        scale_z=args.scale_z,
        max_subjects=args.max_subjects,
        scheme="high_vs_low",
        data=None,
    )

    if not fold_rows:
        raise SystemExit("No valid LOSO folds (need both classes in train and test). Check CLAS paths.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "held_out",
        "n_train_windows",
        "n_test_windows",
        "acc",
        "macro_f1",
        "f1_class_0",
        "f1_class_1",
        "precision_macro",
        "recall_macro",
    ]
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in fold_rows:
            w.writerow({k: row[k] for k in fieldnames})

    accs = [float(r["acc"]) for r in fold_rows]
    f1s = [float(r["macro_f1"]) for r in fold_rows]
    print(
        f"LOSO folds: {len(fold_rows)} | acc mean={np.mean(accs):.4f} std={np.std(accs):.4f} | "
        f"macro_f1 mean={np.mean(f1s):.4f} std={np.std(f1s):.4f} | wrote {args.out_csv}"
    )


if __name__ == "__main__":
    main()
