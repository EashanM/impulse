#!/usr/bin/env python3
"""Extract handcrafted CLAS features: ECG HRV, strict EDA, PPG PRV.

Saves under ``data/processed_clas_features/{ecg,eda,ppg}/Part{{N}}.npz``.

Default windowing:
  ECG / PPG: 20 s window, 5 s stride
  EDA:       20 s window, 5 s stride

Example:
  uv run python scripts/extract_clas_features.py --modality all

Full pipeline (extract + train + table):
  uv run python scripts/clas_preprocess_and_benchmark.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

import _repo_root  # noqa: F401

REPO_ROOT = Path(__file__).resolve().parent.parent
MPL_DIR = REPO_ROOT / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

import neurokit2 as nk  # noqa: E402

from src.data.clas_dataset import (
    DEFAULT_CLAS_ROOT,
    discover_participant_ids,
    estimate_fs,
    load_ecg_matrix,
    load_gsr_ppg_matrix,
)
from src.data.clas_feature_extract import (
    EDA_FEATURE_NAMES,
    HRV_FEATURE_NAMES,
    PROCESSED_SUBDIRS,
    bandpass_signal,
    compute_hrv_features,
    decompose_eda,
    detect_cardiac_peaks,
    iter_blocks,
    lowpass_eda,
    participant_baseline_stats,
    ratio_normalize,
    save_participant_npz,
    select_ecg_channel,
    slide_eda_windows,
    slide_hrv_windows,
    zscore_normalize,
)

DEFAULT_PROCESSED_ROOT = Path("data/processed_clas_features")

# Per-modality defaults (ECG/PPG HRV needs enough beats; EDA SCR works at 20s)
DEFAULT_WINDOWS = {
    "ecg": (20, 5),
    "eda": (20, 5),
    "ppg": (20, 5),
}


def _emit(msg: str, *, show_progress: bool) -> None:
    if show_progress:
        tqdm.write(msg)
    else:
        print(msg, flush=True)


def _concat_block_results(
    chunks: list[tuple[np.ndarray, ...]],
    n_features: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not chunks:
        return (
            np.zeros((0, n_features), np.float32),
            np.zeros((0,), np.int64),
            np.zeros((0,), np.int64),
            np.zeros((0,), "<U64"),
            np.zeros((0,), np.float64),
        )
    return tuple(np.concatenate([c[i] for c in chunks], axis=0) for i in range(5))  # type: ignore[return-value]


def extract_participant_ecg(
    clas_root: Path,
    pid: int,
    out_dir: Path,
    window_sec: int,
    stride_sec: int,
    ecg_channel: str,
    scheme: str,
    min_quality: float | None,
    quality_modality: str,
    verbose: bool,
    log: Callable[[str], None] | None = None,
) -> tuple[bool, int]:
    all_chunks: list[tuple[np.ndarray, ...]] = []
    n_blocks = 0
    for blk in iter_blocks(
        clas_root, pid, scheme=scheme, min_quality=min_quality, quality_modality=quality_modality
    ):
        n_blocks += 1
        ecg = select_ecg_channel(load_ecg_matrix(blk.ecg_path), ecg_channel)
        fs = max(1, int(round(estimate_fs(len(ecg), blk.length_sec))))
        peaks = detect_cardiac_peaks(ecg, fs)
        chunk = slide_hrv_windows(
            ecg,
            fs,
            peaks,
            blk.y,
            blk.block_id,
            blk.block_type,
            window_sec,
            stride_sec,
            nk,
        )
        if verbose and len(chunk[1]) == 0:
            (log or print)(f"  Part{pid} block {blk.block_id} ({blk.block_type}): 0 windows")
        elif verbose:
            (log or print)(f"  Part{pid} block {blk.block_id} ({blk.block_type}): {len(chunk[1])} windows")
        if len(chunk[1]) > 0:
            all_chunks.append(chunk)

    X, y, block_id, block_type, timestamp_sec = _concat_block_results(
        all_chunks, len(HRV_FEATURE_NAMES)
    )
    if len(y) == 0:
        if verbose and log:
            log(f"  Part{pid}: skip (no windows from {n_blocks} blocks)")
        elif verbose:
            print(f"  Part{pid}: skip (no windows from {n_blocks} blocks)")
        return False, 0

    save_participant_npz(
        out_dir / f"Part{pid}.npz",
        X=X,
        y=y,
        block_id=block_id,
        block_type=block_type,
        timestamp_sec=timestamp_sec,
        participant_id=pid,
        feature_names=HRV_FEATURE_NAMES,
        modality="ecg",
        window_sec=window_sec,
        stride_sec=stride_sec,
        extra={"ecg_channel": np.asarray(ecg_channel)},
    )
    counts = np.bincount(y, minlength=2)
    n_win = int(len(y))
    msg = (
        f"  Part{pid}: saved {n_win} windows [low={counts[0]} high={counts[1]}] "
        f"-> {out_dir / f'Part{pid}.npz'}"
    )
    if log:
        log(msg)
    elif verbose:
        print(msg)
    return True, n_win


def extract_participant_ppg(
    clas_root: Path,
    pid: int,
    out_dir: Path,
    window_sec: int,
    stride_sec: int,
    scheme: str,
    min_quality: float | None,
    quality_modality: str,
    verbose: bool,
    log: Callable[[str], None] | None = None,
) -> tuple[bool, int]:
    base_mean, base_std = participant_baseline_stats(
        clas_root,
        pid,
        signal_loader=lambda b: load_gsr_ppg_matrix(b.gsr_ppg_path)[:, 3],
        scheme=scheme,
        min_quality=min_quality,
        quality_modality=quality_modality,
    )

    all_chunks: list[tuple[np.ndarray, ...]] = []
    n_blocks = 0
    for blk in iter_blocks(
        clas_root, pid, scheme=scheme, min_quality=min_quality, quality_modality=quality_modality
    ):
        n_blocks += 1
        ppg = np.asarray(load_gsr_ppg_matrix(blk.gsr_ppg_path)[:, 3], dtype=np.float64)
        fs = max(1, int(round(estimate_fs(len(ppg), blk.length_sec))))
        ppg = zscore_normalize(ppg, base_mean, base_std)
        ppg = bandpass_signal(ppg, fs, 0.5, 8.0)
        peaks = detect_cardiac_peaks(ppg, fs, low_hz=0.5, high_hz=8.0)
        chunk = slide_hrv_windows(
            ppg,
            fs,
            peaks,
            blk.y,
            blk.block_id,
            blk.block_type,
            window_sec,
            stride_sec,
            nk,
        )
        if verbose and len(chunk[1]) == 0:
            (log or print)(f"  Part{pid} block {blk.block_id} ({blk.block_type}): 0 windows")
        elif verbose:
            (log or print)(f"  Part{pid} block {blk.block_id} ({blk.block_type}): {len(chunk[1])} windows")
        if len(chunk[1]) > 0:
            all_chunks.append(chunk)

    X, y, block_id, block_type, timestamp_sec = _concat_block_results(
        all_chunks, len(HRV_FEATURE_NAMES)
    )
    if len(y) == 0:
        if verbose and log:
            log(f"  Part{pid}: skip (no windows from {n_blocks} blocks)")
        elif verbose:
            print(f"  Part{pid}: skip (no windows from {n_blocks} blocks)")
        return False, 0

    save_participant_npz(
        out_dir / f"Part{pid}.npz",
        X=X,
        y=y,
        block_id=block_id,
        block_type=block_type,
        timestamp_sec=timestamp_sec,
        participant_id=pid,
        feature_names=HRV_FEATURE_NAMES,
        modality="ppg",
        window_sec=window_sec,
        stride_sec=stride_sec,
    )
    counts = np.bincount(y, minlength=2)
    n_win = int(len(y))
    msg = (
        f"  Part{pid}: saved {n_win} windows [low={counts[0]} high={counts[1]}] "
        f"-> {out_dir / f'Part{pid}.npz'}"
    )
    if log:
        log(msg)
    elif verbose:
        print(msg)
    return True, n_win


def extract_participant_eda(
    clas_root: Path,
    pid: int,
    out_dir: Path,
    window_sec: int,
    stride_sec: int,
    scheme: str,
    min_quality: float | None,
    quality_modality: str,
    verbose: bool,
    log: Callable[[str], None] | None = None,
) -> tuple[bool, int]:
    base_mean, _ = participant_baseline_stats(
        clas_root,
        pid,
        signal_loader=lambda b: load_gsr_ppg_matrix(b.gsr_ppg_path)[:, 4],
        scheme=scheme,
        min_quality=min_quality,
        quality_modality=quality_modality,
    )

    all_chunks: list[tuple[np.ndarray, ...]] = []
    n_blocks = 0
    for blk in iter_blocks(
        clas_root, pid, scheme=scheme, min_quality=min_quality, quality_modality=quality_modality
    ):
        n_blocks += 1
        gsr = np.asarray(load_gsr_ppg_matrix(blk.gsr_ppg_path)[:, 4], dtype=np.float64)
        fs = max(1, int(round(estimate_fs(len(gsr), blk.length_sec))))
        gsr = ratio_normalize(gsr, base_mean)
        gsr = lowpass_eda(gsr, float(fs))
        tonic, phasic = decompose_eda(gsr, fs, nk)
        chunk = slide_eda_windows(
            tonic,
            phasic,
            fs,
            blk.y,
            blk.block_id,
            blk.block_type,
            window_sec,
            stride_sec,
        )
        if verbose and len(chunk[1]) == 0:
            (log or print)(f"  Part{pid} block {blk.block_id} ({blk.block_type}): 0 windows")
        elif verbose:
            (log or print)(f"  Part{pid} block {blk.block_id} ({blk.block_type}): {len(chunk[1])} windows")
        if len(chunk[1]) > 0:
            all_chunks.append(chunk)

    X, y, block_id, block_type, timestamp_sec = _concat_block_results(
        all_chunks, len(EDA_FEATURE_NAMES)
    )
    if len(y) == 0:
        if verbose and log:
            log(f"  Part{pid}: skip (no windows from {n_blocks} blocks)")
        elif verbose:
            print(f"  Part{pid}: skip (no windows from {n_blocks} blocks)")
        return False, 0

    save_participant_npz(
        out_dir / f"Part{pid}.npz",
        X=X,
        y=y,
        block_id=block_id,
        block_type=block_type,
        timestamp_sec=timestamp_sec,
        participant_id=pid,
        feature_names=EDA_FEATURE_NAMES,
        modality="eda",
        window_sec=window_sec,
        stride_sec=stride_sec,
    )
    counts = np.bincount(y, minlength=2)
    n_win = int(len(y))
    msg = (
        f"  Part{pid}: saved {n_win} windows [low={counts[0]} high={counts[1]}] "
        f"-> {out_dir / f'Part{pid}.npz'}"
    )
    if log:
        log(msg)
    elif verbose:
        print(msg)
    return True, n_win


def run_extract(
    *,
    clas_root: Path,
    processed_root: Path,
    modalities: list[str],
    window_sec: dict[str, int],
    stride_sec: dict[str, int],
    ecg_channel: str,
    scheme: str,
    min_quality: float | None,
    quality_modality: str,
    max_subjects: int | None,
    verbose: bool = True,
    show_progress: bool = True,
) -> None:
    pids = discover_participant_ids(clas_root)
    if not pids:
        raise FileNotFoundError(f"No participants under {clas_root}")
    if max_subjects is not None:
        pids = pids[:max_subjects]

    runners = {
        "ecg": extract_participant_ecg,
        "eda": extract_participant_eda,
        "ppg": extract_participant_ppg,
    }

    use_bars = show_progress and len(pids) > 0
    mod_list = list(modalities)
    mod_outer = (
        tqdm(mod_list, desc="Step 1 · modalities", unit="modality", position=0, leave=True)
        if use_bars
        else mod_list
    )

    for mod in mod_outer:
        if mod not in runners:
            raise ValueError(f"Unknown modality {mod!r}")
        out_dir = processed_root / PROCESSED_SUBDIRS[mod]
        out_dir.mkdir(parents=True, exist_ok=True)
        w = window_sec[mod]
        s = stride_sec[mod]
        _emit(
            f"\n[Step 1] {mod.upper()}: window={w}s stride={s}s | "
            f"{len(pids)} participants -> {out_dir}",
            show_progress=show_progress,
        )

        per_verbose = verbose and not use_bars
        log_fn = (lambda m: _emit(m, show_progress=True)) if use_bars else None

        pid_iter: list[int] | tqdm = (
            tqdm(
                pids,
                desc=f"Step 1 · {mod.upper()} participants",
                unit="participant",
                position=1,
                leave=False,
            )
            if use_bars
            else pids
        )
        saved = 0
        total_windows = 0
        for pid in pid_iter:
            kwargs = dict(
                clas_root=clas_root,
                pid=pid,
                out_dir=out_dir,
                window_sec=w,
                stride_sec=s,
                scheme=scheme,
                min_quality=min_quality,
                quality_modality=quality_modality,
                verbose=per_verbose,
                log=log_fn,
            )
            if mod == "ecg":
                kwargs["ecg_channel"] = ecg_channel
            ok, n_win = runners[mod](**kwargs)
            if ok:
                saved += 1
                total_windows += n_win
            if use_bars and isinstance(pid_iter, tqdm):
                pid_iter.set_postfix(saved=f"{saved}/{len(pids)}", windows=total_windows)

        _emit(
            f"[Step 1] {mod.upper()} done: {saved}/{len(pids)} files, "
            f"{total_windows} total windows -> {out_dir}",
            show_progress=show_progress,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract CLAS ECG/EDA/PPG handcrafted features")
    parser.add_argument("--clas-root", type=Path, default=DEFAULT_CLAS_ROOT)
    parser.add_argument("--processed-root", type=Path, default=DEFAULT_PROCESSED_ROOT)
    parser.add_argument(
        "--modality",
        choices=["ecg", "eda", "ppg", "all"],
        default="all",
    )
    parser.add_argument("--window-sec", type=int, default=None, help="Override all modalities")
    parser.add_argument("--stride-sec", type=int, default=None, help="Override all modalities")
    parser.add_argument("--ecg-window-sec", type=int, default=DEFAULT_WINDOWS["ecg"][0])
    parser.add_argument("--ecg-stride-sec", type=int, default=DEFAULT_WINDOWS["ecg"][1])
    parser.add_argument("--eda-window-sec", type=int, default=DEFAULT_WINDOWS["eda"][0])
    parser.add_argument("--eda-stride-sec", type=int, default=DEFAULT_WINDOWS["eda"][1])
    parser.add_argument("--ppg-window-sec", type=int, default=DEFAULT_WINDOWS["ppg"][0])
    parser.add_argument("--ppg-stride-sec", type=int, default=DEFAULT_WINDOWS["ppg"][1])
    parser.add_argument("--ecg-channel", choices=["ecg1", "ecg2", "mean"], default="ecg1")
    parser.add_argument("--label-scheme", default="high_vs_low")
    parser.add_argument("--min-quality", type=float, default=None)
    parser.add_argument("--quality-modality", choices=["ecg", "eda", "ppg"], default="ecg")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    mods = ["ecg", "eda", "ppg"] if args.modality == "all" else [args.modality]

    window_sec = {
        "ecg": args.ecg_window_sec,
        "eda": args.eda_window_sec,
        "ppg": args.ppg_window_sec,
    }
    stride_sec = {
        "ecg": args.ecg_stride_sec,
        "eda": args.eda_stride_sec,
        "ppg": args.ppg_stride_sec,
    }
    if args.window_sec is not None:
        for m in mods:
            window_sec[m] = args.window_sec
    if args.stride_sec is not None:
        for m in mods:
            stride_sec[m] = args.stride_sec

    for m in mods:
        if stride_sec[m] > window_sec[m]:
            print(f"stride-sec must be <= window-sec for {m}", file=sys.stderr)
            sys.exit(1)

    run_extract(
        clas_root=args.clas_root,
        processed_root=args.processed_root,
        modalities=mods,
        window_sec=window_sec,
        stride_sec=stride_sec,
        ecg_channel=args.ecg_channel,
        scheme=args.label_scheme,
        min_quality=args.min_quality,
        quality_modality=args.quality_modality,
        max_subjects=args.max_subjects,
        verbose=not args.quiet,
        show_progress=not args.quiet,
    )
    print(f"\nDone. Processed root: {args.processed_root}")


if __name__ == "__main__":
    main()
