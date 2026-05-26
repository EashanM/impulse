#!/usr/bin/env python3
"""Extract Albaladejo-Gonzalez-style fixed-window HRV features from SWELL ECG.

This path is separate from the Mortensen extractor on purpose:
  - Mortensen: beat-count windows, 34 custom HRV features
  - Albaladejo: fixed-second windows, NeuroKit2 HRV feature family

Output per subject:
  data/processed_swell_hrv_albaladejo/PP{N}.npz

Each file contains:
  X               float32 (N_windows, 50)  HRV features
  y               int64   (N_windows,)     binary label: 0=neutral, 1=stress
  condition_code  int64   (N_windows,)     0=N, 1=T, 2=I
  block_num       int64   (N_windows,)     SWELL block number from overview xlsx
  timestamp_sec   float64 (N_windows,)     window start timestamp within source file
  is_neutral      bool    (N_windows,)     True for neutral windows used in baseline norm
  subject_id      int64   scalar
  feature_names   <U32    (50,)            column names

Important label handling:
  - Keep only N, T, I task windows
  - Exclude R rest/break windows entirely
  - Binary task: N -> 0, T/I -> 1

Default windowing matches the best SWELL MLP configuration reported in the paper:
  --window-sec 210
  --stride-sec 30
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

import numpy as np
import openpyxl
from scipy.signal import butter, filtfilt, find_peaks

# NeuroKit2 imports matplotlib on import. Point its cache inside the repo.
REPO_ROOT = Path(__file__).resolve().parent.parent
MPL_DIR = REPO_ROOT / ".mplcache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_DIR))

# NeuroKit2 still calls np.trapz; NumPy in this repo only exposes trapezoid.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

import neurokit2 as nk  # noqa: E402

from src.data.poly5_reader import read_poly5


HEART_PRE_CH = 7
ECG_FS = 2048

S00_DIR = Path(
    "data/raw/0_SWELL/0 - Raw data/D - Physiology - raw data/"
    "Mobi signals (raw and filtered)"
)
OVERVIEW_XLSX = Path("data/raw/0_SWELL/SWELL-KW - overview available data.xlsx")
OUT_DIR = Path("data/processed_swell_hrv_albaladejo")

# Match the classic NeuroKit2-style 52-feature family used in the paper,
# excluding ULF and VLF because they are not reliable / not computed here.
TIME_FEATURES = [
    "HRV_MeanNN",
    "HRV_SDNN",
    "HRV_SDANN1",
    "HRV_SDNNI1",
    "HRV_SDANN2",
    "HRV_SDNNI2",
    "HRV_SDANN5",
    "HRV_SDNNI5",
    "HRV_RMSSD",
    "HRV_SDSD",
    "HRV_CVNN",
    "HRV_CVSD",
    "HRV_MedianNN",
    "HRV_MadNN",
    "HRV_MCVNN",
    "HRV_IQRNN",
    "HRV_SDRMSSD",
    "HRV_Prc20NN",
    "HRV_Prc80NN",
    "HRV_pNN50",
    "HRV_pNN20",
    "HRV_MinNN",
    "HRV_MaxNN",
    "HRV_HTI",
    "HRV_TINN",
]
FREQ_FEATURES = [
    "HRV_LF",
    "HRV_HF",
    "HRV_VHF",
    "HRV_TP",
    "HRV_LFHF",
    "HRV_LFn",
    "HRV_HFn",
    "HRV_LnHF",
]
NONLINEAR_FEATURES = [
    "HRV_SD1",
    "HRV_SD2",
    "HRV_SD1SD2",
    "HRV_S",
    "HRV_CSI",
    "HRV_CVI",
    "HRV_CSI_Modified",
    "HRV_PIP",
    "HRV_IALS",
    "HRV_PSS",
    "HRV_PAS",
    "HRV_GI",
    "HRV_SI",
    "HRV_AI",
    "HRV_PI",
    "HRV_ApEn",
    "HRV_SampEn",
]
FEATURE_NAMES = TIME_FEATURES + FREQ_FEATURES + NONLINEAR_FEATURES
N_FEATURES = len(FEATURE_NAMES)

COND_TO_BINARY = {"N": 0, "T": 1, "I": 1}
COND_TO_CODE = {"N": 0, "T": 1, "I": 2}


def iter_poly5_s00_files(s00_dir: Path) -> list[Path]:
    """Return sorted unique paths to Poly5 physiology files (case-insensitive .s00)."""
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in ("*.S00", "*.s00"):
        for p in sorted(s00_dir.glob(pattern)):
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                out.append(p)
    out.sort(key=lambda x: x.name.lower())
    return out


def build_condition_map(xlsx_path: Path) -> dict[tuple[int, int], str]:
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sh = wb["Sheet3"]
    cond_map: dict[tuple[int, int], str] = {}
    for row in sh.iter_rows(values_only=True):
        pp, block, cond = row[0], row[1], row[2]
        if pp is None or block is None or cond is None:
            continue
        try:
            cond_map[(int(pp), int(block))] = str(cond).strip().upper()
        except (TypeError, ValueError):
            continue
    return cond_map


def parse_filename(stem: str) -> tuple[int, int] | None:
    match = re.match(r"pp?(\d+)[_-].*[_-]c(\d)", stem.lower())
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Simple full-file R-peak detector reused from the Mortensen SWELL path."""
    nyq = fs / 2.0
    lo, hi = 5.0 / nyq, min(40.0 / nyq, 0.99)
    try:
        b, a = butter(3, [lo, hi], btype="band")
        ecg_f = filtfilt(b, a, ecg)
    except Exception:
        ecg_f = np.asarray(ecg, dtype=np.float64)
    threshold = float(np.std(ecg_f) * 1.5)
    min_dist = int(0.33 * fs)
    peaks, _ = find_peaks(ecg_f, height=threshold, distance=min_dist)
    return np.asarray(peaks, dtype=np.int64)


def _safe_scalar(value: object) -> float:
    try:
        x = float(value)
    except Exception:
        return np.nan
    return x if np.isfinite(x) else np.nan


def compute_albaladejo_features(rpeaks_local: np.ndarray, fs: int) -> np.ndarray:
    """Compute the fixed 50-feature HRV vector for one ECG window."""
    feats = np.full(N_FEATURES, np.nan, dtype=np.float32)
    if len(rpeaks_local) < 4:
        return feats

    peaks = {"ECG_R_Peaks": np.asarray(rpeaks_local, dtype=np.int64)}
    feature_map: dict[str, float] = {}

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            time_df = nk.hrv_time(peaks, sampling_rate=fs, show=False)
            freq_df = nk.hrv_frequency(peaks, sampling_rate=fs, show=False)
            nonlinear_df = nk.hrv_nonlinear(peaks, sampling_rate=fs, show=False)
    except Exception:
        return feats

    for df in (time_df, freq_df, nonlinear_df):
        if df is None or len(df) == 0:
            continue
        row = df.iloc[0]
        for col in df.columns:
            feature_map[col] = _safe_scalar(row[col])

    for idx, name in enumerate(FEATURE_NAMES):
        feats[idx] = feature_map.get(name, np.nan)
    return feats


def process_file(
    path: Path,
    binary_label: int,
    condition_code: int,
    block_num: int,
    window_sec: int,
    stride_sec: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return features and metadata arrays for one SWELL physiology file."""
    sig = read_poly5(path)
    ecg = np.asarray(sig.data[HEART_PRE_CH], dtype=np.float64)
    fs = int(sig.fs)
    if fs != ECG_FS:
        print(f"    warning: {path.name} fs={fs}, expected {ECG_FS}")

    peaks = detect_r_peaks(ecg, fs)
    if len(peaks) < 4:
        empty_x = np.zeros((0, N_FEATURES), dtype=np.float32)
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_t = np.zeros((0,), dtype=np.float64)
        return empty_x, empty_i, empty_i, empty_i, empty_t

    window_samples = int(window_sec * fs)
    stride_samples = int(stride_sec * fs)
    if len(ecg) < window_samples:
        empty_x = np.zeros((0, N_FEATURES), dtype=np.float32)
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_t = np.zeros((0,), dtype=np.float64)
        return empty_x, empty_i, empty_i, empty_i, empty_t

    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    rows_cond: list[int] = []
    rows_block: list[int] = []
    rows_time: list[float] = []

    for start in range(0, len(ecg) - window_samples + 1, stride_samples):
        end = start + window_samples
        mask = (peaks >= start) & (peaks < end)
        rpeaks_local = peaks[mask] - start
        feats = compute_albaladejo_features(rpeaks_local, fs)
        if not np.isfinite(feats).any():
            continue
        rows_x.append(feats)
        rows_y.append(binary_label)
        rows_cond.append(condition_code)
        rows_block.append(block_num)
        rows_time.append(start / fs)

    if not rows_x:
        empty_x = np.zeros((0, N_FEATURES), dtype=np.float32)
        empty_i = np.zeros((0,), dtype=np.int64)
        empty_t = np.zeros((0,), dtype=np.float64)
        return empty_x, empty_i, empty_i, empty_i, empty_t

    return (
        np.stack(rows_x).astype(np.float32),
        np.asarray(rows_y, dtype=np.int64),
        np.asarray(rows_cond, dtype=np.int64),
        np.asarray(rows_block, dtype=np.int64),
        np.asarray(rows_time, dtype=np.float64),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract fixed-window SWELL HRV features for Albaladejo-style MLP"
    )
    parser.add_argument("--s00-dir", default=str(S00_DIR))
    parser.add_argument("--overview-xlsx", default=str(OVERVIEW_XLSX))
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    parser.add_argument("--window-sec", type=int, default=210)
    parser.add_argument("--stride-sec", type=int, default=30)
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    s00_dir = Path(args.s00_dir)
    if not s00_dir.is_dir():
        raise FileNotFoundError(f"S00 directory does not exist: {s00_dir}")

    overview_path = Path(args.overview_xlsx)
    if not overview_path.is_file():
        raise FileNotFoundError(
            f"Overview workbook not found: {overview_path}. "
            "Pass --overview-xlsx to SWELL-KW - overview available data.xlsx."
        )

    cond_map = build_condition_map(overview_path)
    files = iter_poly5_s00_files(s00_dir)
    if not files:
        raise FileNotFoundError(
            f"No Poly5 *.S00 / *.s00 files under {s00_dir}. "
            "Pass --s00-dir to the folder that contains the Mobi physiology exports."
        )

    pp_files: dict[int, list[tuple[Path, str, int]]] = {}
    for f in files:
        parsed = parse_filename(f.stem)
        if parsed is None:
            continue
        pp_num, block_num = parsed
        cond = cond_map.get((pp_num, block_num))
        if cond not in COND_TO_BINARY:
            continue
        pp_files.setdefault(pp_num, []).append((f, cond, block_num))

    pp_nums = sorted(pp_files.keys())
    if args.max_subjects is not None:
        pp_nums = pp_nums[: args.max_subjects]

    print(
        f"Processing {len(pp_nums)} participants | "
        f"window={args.window_sec}s stride={args.stride_sec}s | "
        f"features={N_FEATURES}"
    )

    for pp_num in pp_nums:
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_cond: list[np.ndarray] = []
        all_block: list[np.ndarray] = []
        all_time: list[np.ndarray] = []

        for path, cond, block_num in sorted(pp_files[pp_num], key=lambda x: x[2]):
            print(
                f"  PP{pp_num:02d} {path.name} -> cond={cond} block={block_num}",
                end="",
                flush=True,
            )
            x, y, cond_code, block_arr, t = process_file(
                path=path,
                binary_label=COND_TO_BINARY[cond],
                condition_code=COND_TO_CODE[cond],
                block_num=block_num,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
            )
            print(f" -> {len(y)} windows")
            if len(y) == 0:
                continue
            all_x.append(x)
            all_y.append(y)
            all_cond.append(cond_code)
            all_block.append(block_arr)
            all_time.append(t)

        if not all_x:
            print(f"  PP{pp_num:02d}: no valid windows, skipping")
            continue

        x_pp = np.concatenate(all_x, axis=0)
        y_pp = np.concatenate(all_y, axis=0)
        cond_pp = np.concatenate(all_cond, axis=0)
        block_pp = np.concatenate(all_block, axis=0)
        time_pp = np.concatenate(all_time, axis=0)
        is_neutral = y_pp == 0

        out_path = out_dir / f"PP{pp_num}.npz"
        np.savez_compressed(
            out_path,
            X=x_pp,
            y=y_pp,
            condition_code=cond_pp,
            block_num=block_pp,
            timestamp_sec=time_pp,
            is_neutral=is_neutral,
            subject_id=np.int64(pp_num),
            feature_names=np.asarray(FEATURE_NAMES, dtype="<U32"),
        )

        counts = np.bincount(y_pp, minlength=2)
        cond_counts = np.bincount(cond_pp, minlength=3)
        print(
            f"  PP{pp_num:02d}: saved {len(y_pp):4d} windows "
            f"[neutral={counts[0]} stress={counts[1]} | "
            f"N={cond_counts[0]} T={cond_counts[1]} I={cond_counts[2]}] -> {out_path}"
        )

    print(f"\nDone. Output: {out_dir}")
    print(f"Features ({N_FEATURES}): {FEATURE_NAMES}")


if __name__ == "__main__":
    main()
