"""Fixed-window SWELL EDA features matching the Albaladejo NPZ schema."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from src.data.swell_hrv_albaladejo import COND_TO_BINARY, COND_TO_CODE, DEFAULT_EXCLUDE_SUBJECTS

FEATURE_NAMES: tuple[str, ...] = (
    "EDA_Mean",
    "EDA_Std",
    "EDA_Median",
    "EDA_Min",
    "EDA_Max",
    "EDA_Range",
    "EDA_IQR",
    "EDA_MAD",
    "EDA_Slope",
    "EDA_DiffMeanAbs",
    "EDA_DiffStd",
    "EDA_RMS",
    "EDA_AUC",
    "EDA_TonicMean",
    "EDA_TonicStd",
    "EDA_PhasicMean",
    "EDA_PhasicStd",
    "EDA_PhasicAbsMean",
    "EDA_PhasicPositiveAUC",
    "SCR_PeakCount",
    "SCR_PeakRatePerMin",
    "SCR_MeanPeakAmp",
    "SCR_MaxPeakAmp",
)
N_FEATURES = len(FEATURE_NAMES)


def _fill_missing(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if np.isfinite(x).all():
        return x
    out = x.copy()
    idx = np.arange(len(out))
    good = np.isfinite(out)
    if not good.any():
        return np.zeros_like(out, dtype=np.float64)
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def lowpass_eda(eda: np.ndarray, fs: int, cutoff_hz: float = 1.0) -> np.ndarray:
    """Low-pass filter EDA for window-level descriptive features."""
    eda = _fill_missing(eda)
    if len(eda) < max(16, fs * 2):
        return eda
    nyq = fs / 2.0
    cutoff = min(float(cutoff_hz) / nyq, 0.99)
    try:
        b, a = butter(3, cutoff, btype="low")
        return filtfilt(b, a, eda)
    except Exception:
        return eda


def _moving_average(x: np.ndarray, width: int) -> np.ndarray:
    width = max(1, min(int(width), len(x)))
    if width <= 1:
        return x.copy()
    pad_left = width // 2
    pad_right = width - 1 - pad_left
    padded = np.pad(x, (pad_left, pad_right), mode="edge")
    kernel = np.ones(width, dtype=np.float64) / width
    return np.convolve(padded, kernel, mode="valid")


def compute_eda_feature_vector(eda_window: np.ndarray, fs: int) -> np.ndarray:
    """Return fixed EDA descriptive features for one window."""
    x = _fill_missing(eda_window)
    feats = np.full(N_FEATURES, np.nan, dtype=np.float32)
    if len(x) < 2:
        return feats

    t = np.arange(len(x), dtype=np.float64) / float(fs)
    duration = max(float(t[-1] - t[0]), 1.0 / float(fs))
    dx = np.diff(x)
    q25, q75 = np.nanpercentile(x, [25, 75])
    median = float(np.nanmedian(x))
    tonic = _moving_average(x, width=max(1, int(round(10.0 * fs))))
    phasic = x - tonic
    positive_phasic = np.clip(phasic, 0.0, None)
    min_peak_distance = max(1, int(round(1.0 * fs)))
    prominence = max(float(np.nanstd(phasic)) * 0.5, 1e-8)
    peaks, props = find_peaks(phasic, distance=min_peak_distance, prominence=prominence)
    peak_amps = phasic[peaks] if len(peaks) else np.zeros((0,), dtype=np.float64)

    try:
        slope = float(np.polyfit(t, x, deg=1)[0])
    except Exception:
        slope = np.nan

    values = [
        float(np.nanmean(x)),
        float(np.nanstd(x)),
        median,
        float(np.nanmin(x)),
        float(np.nanmax(x)),
        float(np.nanmax(x) - np.nanmin(x)),
        float(q75 - q25),
        float(np.nanmedian(np.abs(x - median))),
        slope,
        float(np.nanmean(np.abs(dx))) if len(dx) else 0.0,
        float(np.nanstd(dx)) if len(dx) else 0.0,
        float(np.sqrt(np.nanmean(np.square(x)))),
        float(np.trapezoid(x, t) / duration),
        float(np.nanmean(tonic)),
        float(np.nanstd(tonic)),
        float(np.nanmean(phasic)),
        float(np.nanstd(phasic)),
        float(np.nanmean(np.abs(phasic))),
        float(np.trapezoid(positive_phasic, t) / duration),
        float(len(peaks)),
        float(len(peaks) / (duration / 60.0)),
        float(np.nanmean(peak_amps)) if len(peak_amps) else 0.0,
        float(np.nanmax(peak_amps)) if len(peak_amps) else 0.0,
    ]
    feats[:] = np.asarray(values, dtype=np.float32)
    return feats


def sliding_eda_windows(
    eda: np.ndarray,
    fs: int,
    *,
    window_sec: float,
    stride_sec: float,
    binary_label: int,
    condition_code: int,
    block_num: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract per-window EDA rows for one physiology session."""
    eda = lowpass_eda(eda, fs)
    window_samples = max(1, int(round(window_sec * fs)))
    stride_samples = max(1, int(round(stride_sec * fs)))

    empty_x = np.zeros((0, N_FEATURES), dtype=np.float32)
    empty_i = np.zeros((0,), dtype=np.int64)
    empty_t = np.zeros((0,), dtype=np.float64)
    if len(eda) < window_samples:
        return empty_x, empty_i, empty_i, empty_i, empty_t

    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    rows_cond: list[int] = []
    rows_block: list[int] = []
    rows_time: list[float] = []

    for start in range(0, len(eda) - window_samples + 1, stride_samples):
        end = start + window_samples
        feats = compute_eda_feature_vector(eda[start:end], fs)
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
