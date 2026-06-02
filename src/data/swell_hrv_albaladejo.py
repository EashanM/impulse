"""Albaladejo-style fixed-window HRV features from SWELL ECG (NeuroKit2)."""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

# NeuroKit2 imports matplotlib on import.
_MPL_DIR = Path(__file__).resolve().parents[2] / ".mplcache"
_MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_DIR))

if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid  # type: ignore[attr-defined]

import neurokit2 as nk  # noqa: E402

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
FEATURE_NAMES: tuple[str, ...] = tuple(TIME_FEATURES + FREQ_FEATURES + NONLINEAR_FEATURES)
N_FEATURES = len(FEATURE_NAMES)

COND_TO_BINARY = {"N": 0, "T": 1, "I": 1}
COND_TO_CODE = {"N": 0, "T": 1, "I": 2}

DEFAULT_EXCLUDE_SUBJECTS = ("PP7", "PP8", "PP11", "PP23")


def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """Bandpass + peak pick on full-file ECG (same heuristic as replicate bundle)."""
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


def compute_hrv_feature_vector(rpeaks_local: np.ndarray, fs: int) -> np.ndarray:
    """Return shape (N_FEATURES,) float32; NaN when peaks or NeuroKit2 fail."""
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


def sliding_hrv_windows(
    ecg: np.ndarray,
    fs: int,
    peaks: np.ndarray,
    *,
    window_sec: float,
    stride_sec: float,
    binary_label: int,
    condition_code: int,
    block_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-window HRV rows for one physiology session."""
    ecg = np.asarray(ecg, dtype=np.float64)
    window_samples = max(1, int(round(window_sec * fs)))
    stride_samples = max(1, int(round(stride_sec * fs)))

    empty_x = np.zeros((0, N_FEATURES), dtype=np.float32)
    empty_i = np.zeros((0,), dtype=np.int64)
    empty_t = np.zeros((0,), dtype=np.float64)
    if len(ecg) < window_samples:
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
        feats = compute_hrv_feature_vector(rpeaks_local, fs)
        if not np.isfinite(feats).any():
            continue
        rows_x.append(feats)
        rows_y.append(binary_label)
        rows_cond.append(condition_code)
        rows_block.append(block_num)
        rows_time.append(start / fs)

    if not rows_x:
        return empty_x, empty_i, empty_i, empty_i, empty_t

    return (
        np.stack(rows_x).astype(np.float32),
        np.asarray(rows_y, dtype=np.int64),
        np.asarray(rows_cond, dtype=np.int64),
        np.asarray(rows_block, dtype=np.int64),
        np.asarray(rows_time, dtype=np.float64),
    )
