#!/usr/bin/env python3
"""CLAS stress proxy (high vs low load): LOSO grid over ECG, PPG, EDA × linear, GRU, CNN+GRU.

PPG and EDA use NeuroKit2 feature maps (``ppg_nk``, ``eda_nk``) from preprocessed caches.
ECG uses raw resampled waveforms (``ecg2``).

Preprocess NK features first:
  uv run python scripts/preprocess_clas_nk.py --out-root data/processed_clas_nk

Example benchmark:
  uv run python scripts/clas_benchmark_stress_table.py \\
      --processed-root data/processed_clas_nk --require-cache --epochs 25
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
from src.data.clas_nk_features import nk_default_target_len
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
    "ppg_nk": "PPG",
    "eda_nk": "EDA",
}


def _quality_modality_for(modality: str, default: str) -> str:
    if modality == "ppg_nk":
        return "ppg"
    if modality in ("eda_nk", "gsr"):
        return "eda"
    return default


def _target_len_for(modality: str, ecg_target_len: int, nk_target_len: int) -> int:
    if modality in ("ppg_nk", "eda_nk"):
        return nk_target_len
    return ecg_target_len

def _md_escape_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="CLAS LOSO summary table (ECG, PPG, EDA × 3 encoders)")
    parser.add_argument("--clas-root", type=Path, default=DEFAULT_CLAS_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/clas_stress_benchmark"))
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--stride-sec", type=float, default=8.0)
    parser.add_argument("--target-len", type=int, default=1024, help="Sequence length L for raw ECG")
    parser.add_argument(
        "--nk-target-len",
        type=int,
        default=None,
        help=f"Sequence length L for NK PPG/EDA (default {nk_default_target_len()})",
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path("data/processed_clas_nk"),
        help="Directory with Part*_ppg_nk.npz / Part*_eda_nk.npz caches",
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Require NK caches (do not compute NK features on the fly)",
    )
    parser.add_argument(
        "--sub-win-sec",
        type=float,
        default=2.0,
        help="Sub-window length inside each CLAS window for NK features",
    )
    parser.add_argument(
        "--sub-stride-sec",
        type=float,
        default=2.0,
        help="Sub-window stride for NK features (preprocess + on-the-fly)",
    )
    parser.add_argument(
        "--use-raw-ppg-eda",
        action="store_true",
        help="Use raw ppg/gsr instead of NeuroKit ppg_nk/eda_nk",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=None,
        metavar="MOD",
        help=(
            "Modalities to run. Options: ecg2, ppg_nk, eda_nk, ppg, gsr. "
            "Defaults to: ecg2+ppg_nk+eda_nk (or ecg2+ppg+gsr with --use-raw-ppg-eda)."
        ),
    )
    parser.add_argument(
        "--encoders",
        nargs="+",
        choices=["linear", "gru", "cnn_gru"],
        default=None,
        metavar="ENC",
        help="Encoders to run (subset of linear, gru, cnn_gru). Default: all three.",
    )
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

    nk_target_len = args.nk_target_len if args.nk_target_len is not None else nk_default_target_len()
    if args.use_raw_ppg_eda:
        default_modalities = ["ecg2", "ppg", "gsr"]
        processed_root = None
    else:
        default_modalities = ["ecg2", "ppg_nk", "eda_nk"]
        processed_root = args.processed_root

    # Allow subsetting runs from CLI.
    valid_modalities = {"ecg2", "ppg_nk", "eda_nk", "ppg", "gsr"}
    if args.modalities is None:
        modalities = default_modalities
    else:
        modalities = [m.strip() for m in args.modalities]
        unknown = sorted(set(modalities) - valid_modalities)
        if unknown:
            raise SystemExit(f"Unknown modalities: {unknown}. Valid: {sorted(valid_modalities)}")
    # Keep order, drop duplicates.
    modalities = list(dict.fromkeys(modalities))

    if args.encoders is None:
        encoders = ["linear", "gru", "cnn_gru"]
    else:
        encoders = list(dict.fromkeys(args.encoders))

    summary_rows: list[dict[str, object]] = []
    fold_all: list[dict[str, object]] = []

    for mi, modality in enumerate(modalities):
        mod_target_len = _target_len_for(modality, args.target_len, nk_target_len)
        q_mod = _quality_modality_for(modality, args.quality_modality)
        data = load_all_participants(
            args.clas_root,
            pids_all,
            modality,
            args.window_sec,
            args.stride_sec,
            mod_target_len,
            "high_vs_low",
            args.min_quality,
            q_mod,
            processed_root=processed_root,
            nk_sub_win_sec=args.sub_win_sec,
            nk_sub_stride_sec=args.sub_stride_sec,
            require_cache=args.require_cache,
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
                target_len=mod_target_len,
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
                quality_modality=q_mod,
                scale_z=args.scale_z,
                max_subjects=None,
                scheme="high_vs_low",
                data=data,
                processed_root=processed_root,
                nk_sub_win_sec=args.sub_win_sec,
                nk_sub_stride_sec=args.sub_stride_sec,
                require_cache=args.require_cache,
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
                    "data_stream": STREAM_NAMES.get(modality, modality),
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
        "PPG/EDA rows use NeuroKit2 PRV + morphology / SCR feature maps unless --use-raw-ppg-eda.",
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

    print("\n=== Results ===")
    for row in summary_rows:
        print(
            f"  {row['data_stream']} / {row['model']}  "
            f"Accuracy={row['accuracy_mean']:.4f}  "
            f"Precision={row['precision_macro_mean']:.4f}  "
            f"Recall={row['recall_macro_mean']:.4f}  "
            f"F1={row['f1_macro_mean']:.4f}  "
            f"({int(row['n_folds'])} folds)"
        )
    print()
    print(f"Wrote {summary_csv} and {md_path} ({len(summary_rows)} rows).")


if __name__ == "__main__":
    main()
