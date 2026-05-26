"""NeuroKit2 feature extraction for CLAS PPG and EDA (GSR) windows.

Each CLAS label window is tiled into sub-windows; per sub-window we compute a fixed
feature vector. Vectors are stacked as channels over time: tensor layout (C, L) per
sample, then padded/resampled to ``target_len`` along L for CNN/GRU/linear encoders.
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Minimum L after padding (two MaxPool1d(4) in CLASCnnGruEncoder → L / 16)
CNN_MIN_SEQ_LEN = 32

# Fixed feature order (numeric columns from typical NeuroKit2 outputs + manual EDA stats)
PPG_FEATURE_NAMES: tuple[str, ...] = (
    "HRV_RMSSD",
    "HRV_MeanNN",
    "HRV_SDNN",
    "HRV_SDSD",
    "HRV_CVNN",
    "HRV_CVSD",
    "HRV_MedianNN",
    "HRV_MadNN",
    "HRV_pNN50",
    "HRV_pNN20",
    "HRV_MinNN",
    "HRV_MaxNN",
    "HRV_HTI",
    "HRV_TINN",
    "HRV_LF",
    "HRV_HF",
    "HRV_LFHF",
    "HRV_TP",
    "HRV_SD1",
    "HRV_SD2",
    "HRV_S",
    "HRV_CSI",
    "HRV_CVI",
    "HRV_PIP",
    "HRV_IALS",
    "HRV_PSS",
    "HRV_PAS",
    "HRV_GI",
    "HRV_SI",
    "HRV_AI",
    "HRV_PI",
    "HRV_C1d",
    "HRV_C1a",
    "HRV_SD1d",
    "HRV_SD1a",
    "HRV_C2d",
    "HRV_C2a",
    "HRV_C",
    "HRV_DFA_alpha1",
    "PPG_Rate_Mean",
    "PPG_Rate_SD",
    "PPG_Interval_Mean",
    "PPG_Interval_SD",
)

EDA_FEATURE_NAMES: tuple[str, ...] = (
    "mean_tonic",
    "max_tonic",
    "std_tonic",
    "mean_phasic",
    "std_phasic",
    "max_phasic",
    "max_phasic_peak_amp",
    "std_phasic_rise_time",
    "num_phasic_peaks",
    "SCR_Peaks_N",
    "SCR_Amplitude_Mean",
    "SCR_Amplitude_SD",
    "SCR_RiseTime_Mean",
    "SCR_RecoveryTime_Mean",
    "SCR_Area_Mean",
    "EDA_Tonic_Mean",
    "EDA_Tonic_SD",
    "EDA_Phasic_Mean",
    "EDA_Phasic_SD",
    "SCL_Mean",
    "SCL_SD",
)

NK_MODALITIES = frozenset({"ppg_nk", "eda_nk"})


def nk_channel_count(modality: str) -> int:
    if modality == "ppg_nk":
        return len(PPG_FEATURE_NAMES)
    if modality == "eda_nk":
        return len(EDA_FEATURE_NAMES)
    raise ValueError(modality)


def nk_default_target_len() -> int:
    return CNN_MIN_SEQ_LEN


def _zscore_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    mu = float(np.nanmean(x))
    sd = float(np.nanstd(x))
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    return np.nan_to_num((x - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0)


def _lowpass(x: np.ndarray, fs: float, cutoff_hz: float = 5.0) -> np.ndarray:
    nyq = 0.5 * fs
    effective = min(cutoff_hz, 0.99 * nyq)
    if effective <= 0:
        return x.astype(np.float64)
    b, a = butter(4, effective / nyq, btype="low")
    return filtfilt(b, a, x.astype(np.float64))


def _manual_eda_features(tonic: np.ndarray, phasic: np.ndarray, fs: float) -> dict[str, float]:
    prominence = max(1e-6, 0.1 * float(np.nanstd(phasic)))
    peaks, props = find_peaks(phasic, prominence=prominence)
    peak_amplitudes = props.get("prominences", np.array([], dtype=np.float64))
    rise_times: list[float] = []
    for p in peaks:
        if p <= 1:
            continue
        start = max(0, p - int(4 * fs))
        pre = phasic[start:p]
        if len(pre) == 0:
            continue
        trough_idx = start + int(np.argmin(pre))
        rt = (p - trough_idx) / float(fs)
        if rt >= 0:
            rise_times.append(rt)
    return {
        "mean_tonic": float(np.mean(tonic)),
        "max_tonic": float(np.max(tonic)),
        "std_tonic": float(np.std(tonic)),
        "mean_phasic": float(np.mean(phasic)),
        "std_phasic": float(np.std(phasic)),
        "max_phasic": float(np.max(phasic)),
        "max_phasic_peak_amp": float(np.max(peak_amplitudes)) if len(peak_amplitudes) else 0.0,
        "std_phasic_rise_time": float(np.std(rise_times)) if len(rise_times) > 1 else 0.0,
        "num_phasic_peaks": float(len(peaks)),
    }


def _vector_from_names(names: tuple[str, ...], values: dict[str, float]) -> np.ndarray:
    out = np.full(len(names), np.nan, dtype=np.float32)
    for i, n in enumerate(names):
        v = values.get(n)
        if v is not None and np.isfinite(v):
            out[i] = float(v)
    return out


def _merge_nk_row(df, prefix: str = "") -> dict[str, float]:
    if df is None or getattr(df, "empty", True):
        return {}
    row = df.iloc[0]
    out: dict[str, float] = {}
    for col in df.columns:
        try:
            val = float(row[col])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(val):
            continue
        key = f"{prefix}{col}" if prefix and not str(col).startswith(prefix) else str(col)
        out[key] = val
    return out


def extract_ppg_feature_vector(segment: np.ndarray, fs: float) -> np.ndarray:
    """One feature vector for a PPG sub-window."""
    import neurokit2 as nk

    x = np.asarray(segment, dtype=np.float64).reshape(-1)
    if len(x) < max(8, int(fs * 0.75)):
        return _vector_from_names(PPG_FEATURE_NAMES, {})

    values: dict[str, float] = {}
    try:
        signals, info = nk.ppg_process(x, sampling_rate=fs)
        peaks = np.asarray(info.get("PPG_Peaks", []), dtype=int)
        values.update(_merge_nk_row(nk.ppg_intervalrelated(signals, sampling_rate=fs)))
        if len(peaks) >= 2:
            values.update(_merge_nk_row(nk.hrv(peaks, sampling_rate=fs, show=False)))
        try:
            values.update(_merge_nk_row(nk.ppg_analyze(signals, sampling_rate=fs)))
        except Exception:
            pass
    except Exception:
        pass
    return _vector_from_names(PPG_FEATURE_NAMES, values)


def extract_eda_feature_vector(segment: np.ndarray, fs: float) -> np.ndarray:
    """One feature vector for an EDA (GSR) sub-window."""
    import neurokit2 as nk

    x = _zscore_1d(np.asarray(segment, dtype=np.float64).reshape(-1))
    if len(x) < max(8, int(fs * 0.5)):
        return _vector_from_names(EDA_FEATURE_NAMES, {})

    values: dict[str, float] = {}
    try:
        x_f = _lowpass(x, fs=fs, cutoff_hz=min(5.0, 0.45 * fs))
        signals, _info = nk.eda_process(x_f, sampling_rate=fs)
        tonic = np.asarray(signals["EDA_Tonic"], dtype=np.float64)
        phasic = np.asarray(signals["EDA_Phasic"], dtype=np.float64)
        values.update(_manual_eda_features(tonic, phasic, fs))
        values["SCL_Mean"] = float(np.mean(tonic))
        values["SCL_SD"] = float(np.std(tonic))
        values["EDA_Tonic_Mean"] = values.get("mean_tonic", float(np.mean(tonic)))
        values["EDA_Tonic_SD"] = values.get("std_tonic", float(np.std(tonic)))
        values["EDA_Phasic_Mean"] = values.get("mean_phasic", float(np.mean(phasic)))
        values["EDA_Phasic_SD"] = values.get("std_phasic", float(np.std(phasic)))
        try:
            values.update(_merge_nk_row(nk.eda_analyze(signals, sampling_rate=fs)))
        except Exception:
            pass
        try:
            values.update(_merge_nk_row(nk.eda_intervalrelated(signals, sampling_rate=fs)))
        except Exception:
            pass
    except Exception:
        pass
    return _vector_from_names(EDA_FEATURE_NAMES, values)


def resample_feature_time_axis(cl: np.ndarray, target_len: int) -> np.ndarray:
    """Resample (C, L_cur) -> (C, target_len) along time axis L."""
    c, l_cur = cl.shape
    if l_cur == target_len:
        return cl.astype(np.float32)
    if l_cur < 1:
        return np.zeros((c, target_len), dtype=np.float32)
    xp = np.arange(l_cur, dtype=np.float64)
    xnew = np.linspace(0.0, float(l_cur - 1), num=target_len, dtype=np.float64)
    out = np.empty((c, target_len), dtype=np.float32)
    for i in range(c):
        row = cl[i]
        mask = np.isfinite(row)
        if mask.sum() < 2:
            out[i] = 0.0
        else:
            out[i] = np.interp(xnew, xp[mask], row[mask]).astype(np.float32)
    return out


def extract_nk_windows_from_block(
    signal_1d: np.ndarray,
    length_sec: float,
    window_sec: float,
    stride_sec: float,
    target_len: int,
    sub_win_sec: float,
    sub_stride_sec: float,
    kind: Literal["ppg_nk", "eda_nk"],
) -> np.ndarray:
    """
    Tile block into CLAS windows; each window -> (C, target_len) NK feature map.

    Returns (N, C, target_len) float32.
    """
    x = np.asarray(signal_1d, dtype=np.float64).reshape(-1)
    t = x.shape[0]
    if t < 2:
        return np.zeros((0, nk_channel_count(kind), target_len), dtype=np.float32)

    fs = length_sec and (t / float(length_sec)) or 1.0
    fs = max(fs, 1.0)
    c = nk_channel_count(kind)
    extract_fn = extract_ppg_feature_vector if kind == "ppg_nk" else extract_eda_feature_vector

    win = max(1, int(round(fs * window_sec)))
    stride = max(1, int(round(fs * stride_sec)))
    sub_win = max(1, int(round(fs * sub_win_sec)))
    sub_stride = max(1, int(round(fs * sub_stride_sec)))

    effective_target = max(target_len, CNN_MIN_SEQ_LEN)

    def _window_to_tensor(seg: np.ndarray) -> np.ndarray:
        cols: list[np.ndarray] = []
        if len(seg) < sub_win:
            cols.append(extract_fn(seg, fs))
        else:
            for s in range(0, len(seg) - sub_win + 1, sub_stride):
                cols.append(extract_fn(seg[s : s + sub_win], fs))
        if not cols:
            cols.append(extract_fn(seg, fs))
        cl = np.stack(cols, axis=1)  # (C, L_sub)
        return resample_feature_time_axis(cl, effective_target)

    windows: list[np.ndarray] = []
    if win > t:
        windows.append(_window_to_tensor(x))
    else:
        for start in range(0, t - win + 1, stride):
            windows.append(_window_to_tensor(x[start : start + win]))

    if not windows:
        return np.zeros((0, c, effective_target), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32)
