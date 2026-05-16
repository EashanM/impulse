#!/usr/bin/env python3
"""Preprocess CLAS ECG/EDA/PPG features, train encoders (LOSO), write summary table.

Run everything (recommended):
  uv run python scripts/clas_preprocess_and_benchmark.py

Pipeline:
  Step 1/3 — Extract features -> data/processed_clas_features/{ecg,eda,ppg}/
  Step 2/3 — LOSO: linear / GRU / CNN+GRU on feature tensors
  Step 3/3 — Write runs/clas_feature_benchmark/clas_stress_summary.csv (+ .md)

ECG defaults: 20 s window, 5 s stride.

Other examples:
  uv run python scripts/clas_preprocess_and_benchmark.py --skip-preprocess
  uv run python scripts/clas_preprocess_and_benchmark.py --max-subjects 5 --device cpu
"""

from __future__ import annotations

import _repo_root  # noqa: F401

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from extract_clas_features import DEFAULT_PROCESSED_ROOT, run_extract  # noqa: E402
from src.data.clas_dataset import DEFAULT_CLAS_ROOT, discover_participant_ids
from src.data.clas_feature_extract import PROCESSED_SUBDIRS, load_preprocessed_participants
from src.training.clas_loso_core import (
    resolve_device,
    run_loso_encoder_lr,
    seed_all,
)

STREAM_NAMES = {
    "ecg": "ECG",
    "eda": "EDA",
    "ppg": "PPG",
}

ENCODERS = ["linear", "gru", "cnn_gru"]


def _banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}", flush=True)


def _step(msg: str) -> None:
    print(f"\n>>> {msg}", flush=True)


def _emit(msg: str, *, show_progress: bool) -> None:
    if show_progress:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _md_escape_cell(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def write_summary_table(
    summary_rows: list[dict[str, object]],
    fold_all: list[dict[str, object]],
    out_dir: Path,
    *,
    write_fold_rows: bool,
    show_progress: bool,
) -> None:
    _step("Step 3/3: Writing summary table")
    out_dir.mkdir(parents=True, exist_ok=True)
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
    summary_csv = out_dir / "clas_stress_summary.csv"
    with summary_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row[k] for k in fields})

    md_path = out_dir / "clas_stress_summary.md"
    lines = [
        "# CLAS stress proxy (handcrafted features, high load vs low load)",
        "",
        "Preprocessed ECG HRV (50) / EDA strict (5) / PPG PRV (50). "
        "Metrics averaged over LOSO folds.",
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

    if write_fold_rows and fold_all:
        fold_csv = out_dir / "fold_details.csv"
        fn = list(fold_all[0].keys())
        with fold_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fn)
            w.writeheader()
            for row in fold_all:
                w.writerow(row)
        _emit(f"  Fold details: {fold_csv}", show_progress=show_progress)

    _emit(f"  Summary CSV:  {summary_csv}", show_progress=show_progress)
    _emit(f"  Summary MD:   {md_path}", show_progress=show_progress)

    print("\nResults preview:", flush=True)
    print(f"{'stream':<6} {'model':<10} {'acc':>8} {'f1_macro':>10}", flush=True)
    print("-" * 38, flush=True)
    for row in summary_rows:
        print(
            f"{row['data_stream']!s:<6} {row['model']!s:<10} "
            f"{row['accuracy_mean']!s:>8.4f} {row['f1_macro_mean']!s:>10.4f}",
            flush=True,
        )


def run_benchmark(
    *,
    processed_root: Path,
    out_dir: Path,
    modalities: list[str],
    epochs: int,
    batch_size: int,
    embedding_dim: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    gru_hidden: int,
    seed: int,
    device,
    scale_z: bool,
    max_subjects: int | None,
    show_progress: bool,
    write_fold_rows: bool,
    target_len_placeholder: int,
) -> list[dict[str, object]]:
    summary_rows: list[dict[str, object]] = []
    fold_all: list[dict[str, object]] = []

    total_runs = len(modalities) * len(ENCODERS)
    run_bar = (
        tqdm(total=total_runs, desc="Step 2 · benchmark runs", unit="run", position=0, leave=True)
        if show_progress
        else None
    )

    for mi, modality in enumerate(modalities):
        _emit(
            f"[Step 2] Loading preprocessed {STREAM_NAMES[modality]} from "
            f"{processed_root / PROCESSED_SUBDIRS[modality]}",
            show_progress=show_progress,
        )
        data = load_preprocessed_participants(processed_root, modality)
        if max_subjects is not None:
            keep = sorted(data.keys())[:max_subjects]
            data = {k: data[k] for k in keep if k in data}
        subjects = sorted(data.keys())
        n_windows = sum(int(data[p][0].shape[0]) for p in subjects)
        _emit(
            f"[Step 2] {STREAM_NAMES[modality]}: {len(subjects)} participants, "
            f"{n_windows} windows",
            show_progress=show_progress,
        )
        if len(subjects) < 2:
            _emit(f"  Skip {modality}: need >=2 participants.", show_progress=show_progress)
            if run_bar is not None:
                run_bar.update(len(ENCODERS))
            continue

        n_feat = int(data[subjects[0]][0].shape[2])

        for ei, encoder in enumerate(ENCODERS):
            run_seed = seed + mi * 100 + ei
            seed_all(run_seed)
            if run_bar is not None:
                run_bar.set_postfix(stream=STREAM_NAMES[modality], model=encoder)
            else:
                print(
                    f"  [{STREAM_NAMES[modality]}] encoder={encoder} "
                    f"({ei + 1}/{len(ENCODERS)})",
                    flush=True,
                )

            t0 = time.perf_counter()
            fold_rows = run_loso_encoder_lr(
                clas_root=DEFAULT_CLAS_ROOT,
                modality=modality,
                encoder=encoder,
                window_sec=0.0,
                stride_sec=0.0,
                target_len=max(n_feat, target_len_placeholder),
                embedding_dim=embedding_dim,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                dropout=dropout,
                gru_hidden=gru_hidden,
                seed=run_seed,
                device=device,
                min_quality=None,
                quality_modality="ecg",
                scale_z=scale_z,
                max_subjects=None,
                scheme="high_vs_low",
                data=data,
                show_progress=show_progress,
                progress_desc=f"Step 2 · {STREAM_NAMES[modality]}/{encoder} LOSO",
            )
            elapsed = time.perf_counter() - t0

            if not fold_rows:
                _emit(
                    f"    No valid folds for {modality}/{encoder} ({elapsed:.1f}s).",
                    show_progress=show_progress,
                )
                if run_bar is not None:
                    run_bar.update(1)
                continue

            accs = [float(r["acc"]) for r in fold_rows]
            f1s = [float(r["macro_f1"]) for r in fold_rows]
            precs = [float(r["precision_macro"]) for r in fold_rows]
            recalls = [float(r["recall_macro"]) for r in fold_rows]
            mean_acc = float(np.mean(accs))
            mean_f1 = float(np.mean(f1s))

            summary_rows.append(
                {
                    "data_stream": STREAM_NAMES[modality],
                    "modality_key": modality,
                    "model": encoder,
                    "n_folds": len(fold_rows),
                    "accuracy_mean": mean_acc,
                    "f1_macro_mean": mean_f1,
                    "precision_macro_mean": float(np.mean(precs)),
                    "recall_macro_mean": float(np.mean(recalls)),
                }
            )
            _emit(
                f"    {STREAM_NAMES[modality]}/{encoder}: "
                f"acc={mean_acc:.4f} f1={mean_f1:.4f} ({len(fold_rows)} folds, {elapsed:.1f}s)",
                show_progress=show_progress,
            )

            if write_fold_rows:
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

            if run_bar is not None:
                run_bar.update(1)

    if run_bar is not None:
        run_bar.close()

    write_summary_table(
        summary_rows, fold_all, out_dir, write_fold_rows=write_fold_rows, show_progress=show_progress
    )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLAS: preprocess ECG/EDA/PPG + LOSO encoder benchmark (one command)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--clas-root", type=Path, default=DEFAULT_CLAS_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument("--out-dir", type=Path, default=Path("runs/clas_feature_benchmark"))
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip Step 1; use existing files under --processed-root",
    )
    parser.add_argument("--ecg-window-sec", type=int, default=20)
    parser.add_argument("--ecg-stride-sec", type=int, default=5)
    parser.add_argument("--eda-window-sec", type=int, default=20)
    parser.add_argument("--eda-stride-sec", type=int, default=5)
    parser.add_argument("--ppg-window-sec", type=int, default=20)
    parser.add_argument("--ppg-stride-sec", type=int, default=5)
    parser.add_argument("--ecg-channel", choices=["ecg1", "ecg2", "mean"], default="ecg1")
    parser.add_argument("--min-quality", type=float, default=None)
    parser.add_argument("--quality-modality", choices=["ecg", "eda", "ppg"], default="ecg")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--scale-z", action="store_true")
    parser.add_argument("--write-fold-rows", action="store_true")
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (plain print only)",
    )
    args = parser.parse_args()

    show_progress = not args.no_progress
    seed_all(args.seed)
    device = resolve_device(args.device)
    modalities = ["ecg", "eda", "ppg"]

    n_participants = len(discover_participant_ids(args.clas_root))
    if args.max_subjects is not None:
        n_participants = min(n_participants, args.max_subjects)

    _banner("CLAS full pipeline")
    print(
        f"  Participants: {n_participants}  |  Device: {device}  |  Epochs: {args.epochs}",
        flush=True,
    )
    print(f"  CLAS root:       {args.clas_root}", flush=True)
    print(f"  Processed data:  {args.processed_root}", flush=True)
    print(f"  Results:         {args.out_dir}", flush=True)
    print(
        f"  Windows (sec):   ECG {args.ecg_window_sec}/{args.ecg_stride_sec}  "
        f"EDA {args.eda_window_sec}/{args.eda_stride_sec}  "
        f"PPG {args.ppg_window_sec}/{args.ppg_stride_sec}",
        flush=True,
    )

    if not args.skip_preprocess:
        _banner("Step 1/3 — Feature extraction (ECG + EDA + PPG)")
        t0 = time.perf_counter()
        run_extract(
            clas_root=args.clas_root,
            processed_root=args.processed_root,
            modalities=modalities,
            window_sec={
                "ecg": args.ecg_window_sec,
                "eda": args.eda_window_sec,
                "ppg": args.ppg_window_sec,
            },
            stride_sec={
                "ecg": args.ecg_stride_sec,
                "eda": args.eda_stride_sec,
                "ppg": args.ppg_stride_sec,
            },
            ecg_channel=args.ecg_channel,
            scheme="high_vs_low",
            min_quality=args.min_quality,
            quality_modality=args.quality_modality,
            max_subjects=args.max_subjects,
            verbose=not show_progress,
            show_progress=show_progress,
        )
        manifest = {
            "clas_root": str(args.clas_root),
            "window_sec": {
                "ecg": args.ecg_window_sec,
                "eda": args.eda_window_sec,
                "ppg": args.ppg_window_sec,
            },
            "stride_sec": {
                "ecg": args.ecg_stride_sec,
                "eda": args.eda_stride_sec,
                "ppg": args.ppg_stride_sec,
            },
            "ecg_channel": args.ecg_channel,
            "modalities": modalities,
        }
        args.processed_root.mkdir(parents=True, exist_ok=True)
        manifest_path = args.processed_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(
            f"\n[Step 1/3] Done in {time.perf_counter() - t0:.1f}s. "
            f"Manifest: {manifest_path}",
            flush=True,
        )
    else:
        _banner("Step 1/3 — Skipped (--skip-preprocess)")

    for mod in modalities:
        sub = args.processed_root / PROCESSED_SUBDIRS[mod]
        n_files = len(list(sub.glob("Part*.npz"))) if sub.is_dir() else 0
        if n_files == 0:
            raise SystemExit(
                f"No Part*.npz under {sub}. Run without --skip-preprocess or check paths."
            )
        print(f"  Found {n_files} files in {sub}", flush=True)

    _banner("Step 2/3 — LOSO encoder benchmark (linear, GRU, CNN+GRU)")
    print(
        f"  {len(modalities)} modalities × {len(ENCODERS)} encoders = "
        f"{len(modalities) * len(ENCODERS)} runs",
        flush=True,
    )
    t1 = time.perf_counter()
    summary = run_benchmark(
        processed_root=args.processed_root,
        out_dir=args.out_dir,
        modalities=modalities,
        epochs=args.epochs,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        gru_hidden=args.gru_hidden,
        seed=args.seed,
        device=device,
        scale_z=args.scale_z,
        max_subjects=args.max_subjects,
        show_progress=show_progress,
        write_fold_rows=args.write_fold_rows,
        target_len_placeholder=50,
    )
    print(f"\n[Step 2/3] Done in {time.perf_counter() - t1:.1f}s.", flush=True)

    if not summary:
        raise SystemExit("No benchmark rows produced.")

    _banner("All steps complete")
    print(f"  Preprocessed features: {args.processed_root}", flush=True)
    print(f"  Benchmark table:       {args.out_dir / 'clas_stress_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
