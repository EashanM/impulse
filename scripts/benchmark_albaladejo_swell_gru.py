#!/usr/bin/env python3
"""LOSO GRU on Albaladejo-style SWELL HRV NPZs (sequence of HRV windows)."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.albaladejo_swell_sequences import (
    DEFAULT_EXCLUDE,
    load_subjects_data,
    seed_all,
    train_gru_one_loso_fold,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Albaladejo SWELL HRV sequence GRU LOSO")
    parser.add_argument(
        "--data-root",
        default="data/processed_swell_hrv_albaladejo_w20_s5",
        help="Directory of PP*.npz from extract_swell_hrv_features_albaladejo.py",
    )
    parser.add_argument("--exclude", nargs="*", default=DEFAULT_EXCLUDE)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=14)
    parser.add_argument("--rnn-type", choices=["gru", "lstm"], default="gru")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--decision-threshold-tune",
        choices=["none", "f1_val"],
        default="none",
        help="f1_val: choose probability threshold on inner validation sequences to maximize F1.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--no-baseline-norm", action="store_true")
    parser.add_argument(
        "--out-csv",
        default="runs/albaladejo_swell_gru_loso_w20_s5.csv",
    )
    parser.add_argument(
        "--out-predictions-csv",
        default=None,
        help="Optional long CSV of per-sequence y_true/y_pred/p_stress. Defaults to <out-csv stem>_predictions.csv.",
    )
    args = parser.parse_args()

    seed_all(args.seed)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    data_root = Path(args.data_root)
    npz_files = sorted(data_root.glob("PP*.npz"))
    if not npz_files:
        raise FileNotFoundError(
            f"No PP*.npz in {data_root}. Run scripts/extract_swell_hrv_features_albaladejo.py first."
        )

    excluded = set(args.exclude)
    subjects = [p.stem for p in npz_files if p.stem not in excluded]
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
    npz_by_subject = {p.stem: p for p in npz_files if p.stem in subjects}
    if len(subjects) < 2:
        raise ValueError("Need at least 2 subjects for LOSO")

    use_baseline = not args.no_baseline_norm
    subj_data = load_subjects_data(subjects, npz_by_subject, use_baseline_norm=use_baseline)

    input_dim = next(iter(subj_data.values()))[0].shape[1]
    norm_tag = "baseline[0,1]" if use_baseline else "none"
    print(
        f"Albaladejo SWELL GRU LOSO | subjects={len(subjects)} | features={input_dim} | "
        f"rnn={args.rnn_type} | seq_len={args.seq_len} | hidden={args.hidden_dim} | class_weight={args.class_weight} | "
        f"norm={norm_tag} | device={device} | data={data_root}"
    )

    rows: list[dict[str, float | int | str]] = []
    prediction_rows: list[dict[str, float | int | str]] = []
    metric_keys = [
        "accuracy",
        "macro_f1",
        "macro_precision",
        "macro_recall",
        "f1_nostress",
        "f1_stress",
    ]

    for held_out in subjects:
        held_out_num = int(held_out.removeprefix("PP"))
        row, n_te, predictions = train_gru_one_loso_fold(
            subj_data,
            subjects,
            held_out,
            seq_len=args.seq_len,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            rnn_type=args.rnn_type,
            epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            class_weight=args.class_weight,
            decision_threshold=args.decision_threshold,
            decision_threshold_tune=args.decision_threshold_tune,
            device=device,
            split_seed=args.seed + held_out_num,
        )
        rows.append(row)
        for seq_idx, y_true, y_pred, p_stress, threshold_used in zip(
            predictions["sequence_index"],
            predictions["y_true"],
            predictions["y_pred"],
            predictions["p_stress"],
            predictions["decision_threshold_used"],
            strict=True,
        ):
            prediction_rows.append(
                {
                    "subject": held_out,
                    "sequence_index": int(seq_idx),
                    "y_true": int(y_true),
                    "y_pred": int(y_pred),
                    "p_stress": float(p_stress),
                    "decision_threshold": float(threshold_used),
                }
            )
        print(
            f"  {held_out}: acc={row['accuracy']:.3f} macro_f1={row['macro_f1']:.3f} | "
            f"f1=[nostress={row['f1_nostress']:.3f}, stress={row['f1_stress']:.3f}] "
            f"thr={row['decision_threshold_used']:.3f} "
            f"cm=[[{row['tn']}, {row['fp']}], [{row['fn']}, {row['tp']}]] (n={n_te})"
        )

    print()
    for key in metric_keys:
        vals = np.asarray([float(r[key]) for r in rows], dtype=np.float64)
        print(f"mean_{key}: {vals.mean():.3f} +- {vals.std(ddof=0):.3f}")

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "n_test", "tn", "fp", "fn", "tp", "decision_threshold_used"] + metric_keys,
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nsaved_csv: {out_path}")

    pred_path = (
        Path(args.out_predictions_csv)
        if args.out_predictions_csv is not None
        else out_path.with_name(f"{out_path.stem}_predictions{out_path.suffix}")
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    with pred_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "sequence_index", "y_true", "y_pred", "p_stress", "decision_threshold"],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(f"saved_predictions_csv: {pred_path}")


if __name__ == "__main__":
    main()
