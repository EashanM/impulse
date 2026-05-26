#!/usr/bin/env python3
"""CLAS stress proxy (high vs low load): LOSO grid over ECG, PPG, EDA × linear, GRU, CNN+GRU.

Writes mean accuracy, macro F1, macro precision, macro recall (averaged over LOSO folds) to
CSV and Markdown under ``--out-dir``.

Label scheme: high cognitive/affective load = stress (1); baseline/neutral = not stress (0).
See ``src.data.clas_dataset`` HIGH_LOAD_BLOCK_TYPES / LOW_LOAD_BLOCK_TYPES.

Example:
  uv run python scripts/clas_benchmark_stress_table.py --out-dir runs/clas_stress_benchmark \\
      --epochs 25 --device cpu
"""

from __future__ import annotations

import _repo_root  # noqa: F401

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.clas_dataset import DEFAULT_CLAS_ROOT, discover_participant_ids
from src.training.clas_loso_core import (
    load_all_participants,
    resolve_device,
    run_loso_encoder_lr,
    seed_all,
)


STREAM_NAMES = {
    "ecg2": "ECG",
    "ppg": "PPG",
    "gsr": "EDA",
}

def _md_escape_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLAS LOSO summary table (ECG, PPG, EDA × 3 encoders)")
    parser.add_argument("--clas-root", type=Path, default=DEFAULT_CLAS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/clas_stress_benchmark"))
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--stride-sec", type=float, default=8.0)
    parser.add_argument("--target-len", type=int, default=1024)
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
    parser.add_argument("--scale-z", action="store_true")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--write-fold-rows",
        action="store_true",
        help="Also write fold_details.csv (one row per held-out subject per run)",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (e.g. for log capture)",
    )
    args = parser.parse_args()

    seed_all(args.seed)
    device = resolve_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pids_all = discover_participant_ids(args.clas_root)
    if args.max_subjects is not None:
        pids_all = pids_all[: args.max_subjects]

    modalities = ["ecg2", "ppg", "gsr"]
    encoders = ["linear", "gru", "cnn_gru"]

    summary_rows: list[dict[str, object]] = []
    fold_all: list[dict[str, object]] = []

    for mi, modality in enumerate(modalities):
        data = load_all_participants(
            args.clas_root,
            pids_all,
            modality,
            args.window_sec,
            args.stride_sec,
            args.target_len,
            "high_vs_low",
            args.min_quality,
            args.quality_modality,
        )
        subjects = sorted(data.keys())
        print(
            f"[{STREAM_NAMES[modality]}] loaded {len(subjects)} participants with windows "
            f"(from {len(pids_all)} requested ids)."
        )
        if len(subjects) < 2:
            print(f"  Skip {modality}: need >=2 participants with data.")
            continue

        if args.no_progress:
            enc_iter = enumerate(encoders)
        else:
            enc_iter = tqdm(
                list(enumerate(encoders)),
                desc=f"{STREAM_NAMES[modality]} encoders",
                unit="encoder",
                leave=True,
            )

        for ei, encoder in enc_iter:
            run_seed = args.seed + mi * 100 + ei
            seed_all(run_seed)
            t0 = time.perf_counter()
            fold_rows = run_loso_encoder_lr(
                clas_root=args.clas_root,
                modality=modality,
                encoder=encoder,
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
                seed=run_seed,
                device=device,
                min_quality=args.min_quality,
                quality_modality=args.quality_modality,
                scale_z=args.scale_z,
                max_subjects=None,
                scheme="high_vs_low",
                data=data,
            )
            elapsed = time.perf_counter() - t0
            if not args.no_progress and isinstance(enc_iter, tqdm):
                enc_iter.set_postfix(last=encoder, sec=f"{elapsed:.1f}s")
            else:
                print(f"  encoder={encoder} done in {elapsed:.1f}s")

            if not fold_rows:
                msg = f"No valid folds for {modality}/{encoder}."
                if args.no_progress:
                    print(f"    {msg}")
                else:
                    tqdm.write(f"    {msg}")
                continue

            accs = [float(r["acc"]) for r in fold_rows]
            f1s = [float(r["macro_f1"]) for r in fold_rows]
            precs = [float(r["precision_macro"]) for r in fold_rows]
            recalls = [float(r["recall_macro"]) for r in fold_rows]

            summary_rows.append(
                {
                    "data_stream": STREAM_NAMES[modality],
                    "modality_key": modality,
                    "model": encoder,
                    "n_folds": len(fold_rows),
                    "accuracy_mean": float(np.mean(accs)),
                    "f1_macro_mean": float(np.mean(f1s)),
                    "precision_macro_mean": float(np.mean(precs)),
                    "recall_macro_mean": float(np.mean(recalls)),
                }
            )

            if args.write_fold_rows:
                for r in fold_rows:
                    fold_all.append(
                        {
                            "data_stream": STREAM_NAMES[modality],
                            "modality_key": modality,
                            "model": encoder,
                            "held_out": r["held_out"],
                            "n_train_windows": r["n_train_windows"],
                            "n_test_windows": r["n_test_windows"],
                            "acc": r["acc"],
                            "macro_f1": r["macro_f1"],
                            "precision_macro": r["precision_macro"],
                            "recall_macro": r["recall_macro"],
                        }
                    )

    if not summary_rows:
        raise SystemExit("No summary rows produced. Check CLAS layout and filters.")

    summary_csv = args.out_dir / "clas_stress_summary.csv"
    fields = [
        "data_stream",
        "modality_key",
        "model",
        "n_folds",
        "accuracy_mean",
        "f1_macro_mean",
        "precision_macro_mean",
        "recall_macro_mean",
    ]
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row[k] for k in fields})

    md_path = args.out_dir / "clas_stress_summary.md"
    lines = [
        "# CLAS stress proxy (high load vs low load)",
        "",
        "Metrics are **macro** precision/recall/F1 and **accuracy**, each **averaged over LOSO folds**.",
        "Stress = high cognitive/affective load blocks; not stress = baseline/neutral.",
        "",
        "| data_stream | model | n_folds | accuracy | f1_macro | precision_macro | recall_macro |",
        "|-------------|-------|---------|----------|----------|-----------------|--------------|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _md_escape_cell(str(row["data_stream"])),
                    _md_escape_cell(str(row["model"])),
                    str(int(row["n_folds"])),
                    f"{row['accuracy_mean']:.4f}",
                    f"{row['f1_macro_mean']:.4f}",
                    f"{row['precision_macro_mean']:.4f}",
                    f"{row['recall_macro_mean']:.4f}",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append(f"Written: `{summary_csv}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if args.write_fold_rows and fold_all:
        fold_csv = args.out_dir / "fold_details.csv"
        fn = list(fold_all[0].keys())
        with fold_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            for row in fold_all:
                w.writerow(row)

    print(f"Wrote {summary_csv} and {md_path} ({len(summary_rows)} rows).")


if __name__ == "__main__":
    main()
