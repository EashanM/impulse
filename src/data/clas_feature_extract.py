"""Shared CLAS block-wise feature extraction (ECG/PPG HRV, strict EDA)."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from src.data.clas_dataset import BlockRow, estimate_fs, iter_labeled_blocks

# ---------------------------------------------------------------------------
# NeuroKit2 HRV feature names (50; no ULF/VLF)
# ---------------------------------------------------------------------------

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
HRV_FEATURE_NAMES = TIME_FEATURES + FREQ_FEATURES + NONLINEAR_FEATURES
N_HRV_FEATURES = len(HRV_FEATURE_NAMES)

EDA_FEATURE_NAMES = [
    "mean_tonic",
    "max_tonic",
    "max_phasic_peak_amp",
    "std_phasic_rise_time",
    "num_phasic_peaks",
]
N_EDA_FEATURES = len(EDA_FEATURE_NAMES)

PROCESSED_SUBDIRS = {
    "ecg": "ecg",
    "eda": "eda",
    "ppg": "ppg",
}


def select_ecg_channel(ecg_tc: np.ndarray, channel: str) -> np.ndarray:
    if ecg_tc.ndim != 2 or ecg_tc.shape[1] < 2:
        raise ValueError(f"Expected (T, 2) ECG, got {ecg_tc.shape}")
    if channel == "ecg1":
        return np.asarray(ecg_tc[:, 0], dtype=np.float64)
    if channel == "ecg2":
        return np.asarray(ecg_tc[:, 1], dtype=np.float64)
    if channel == "mean":
        return np.mean(ecg_tc[:, :2], axis=1, dtype=np.float64)
    raise ValueError(f"Unknown channel {channel!r}")


def bandpass_signal(x: np.ndarray, fs: int, low_hz: float, high_hz: float) -> np.ndarray:
    nyq = fs / 2.0
    lo = max(low_hz / nyq, 1e-6)
    hi = min(high_hz / nyq, 0.99)
    if lo >= hi:
        return np.asarray(x, dtype=np.float64)
    b, a = butter(3, [lo, hi], btype="band")
    return filtfilt(b, a, np.asarray(x, dtype=np.float64))


def detect_cardiac_peaks(signal: np.ndarray, fs: int, *, low_hz: float = 5.0, high_hz: float = 40.0) -> np.ndarray:
    """Bandpass + scipy peaks (ECG or PPG)."""
    ecg_f = bandpass_signal(signal, fs, low_hz, high_hz)
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


def compute_hrv_features(rpeaks_local: np.ndarray, fs: int, nk_module: object) -> np.ndarray:
    """50-dim NeuroKit2 HRV vector for peaks inside one window."""
    feats = np.full(N_HRV_FEATURES, np.nan, dtype=np.float32)
    if len(rpeaks_local) < 4:
        return feats

    peaks = {"ECG_R_Peaks": np.asarray(rpeaks_local, dtype=np.int64)}
    feature_map: dict[str, float] = {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            time_df = nk_module.hrv_time(peaks, sampling_rate=fs, show=False)
            freq_df = nk_module.hrv_frequency(peaks, sampling_rate=fs, show=False)
            nonlinear_df = nk_module.hrv_nonlinear(peaks, sampling_rate=fs, show=False)
    except Exception:
        return feats

    for df in (time_df, freq_df, nonlinear_df):
        if df is None or len(df) == 0:
            continue
        row = df.iloc[0]
        for col in df.columns:
            feature_map[col] = _safe_scalar(row[col])

    for idx, name in enumerate(HRV_FEATURE_NAMES):
        feats[idx] = feature_map.get(name, np.nan)
    return feats


def lowpass_eda(x: np.ndarray, fs: float, cutoff_hz: float = 5.0, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    effective = min(cutoff_hz, 0.99 * nyq)
    b, a = butter(order, effective / nyq, btype="low")
    return filtfilt(b, a, np.asarray(x, dtype=np.float64))


def ratio_normalize(x: np.ndarray, baseline_mean: float) -> np.ndarray:
    denom = baseline_mean if abs(baseline_mean) > 1e-8 else 1.0
    out = x / denom
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def zscore_normalize(x: np.ndarray, baseline_mean: float, baseline_std: float) -> np.ndarray:
    sd = baseline_std if baseline_std > 1e-8 else 1.0
    out = (x - baseline_mean) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def decompose_eda(eda_filt: np.ndarray, fs: int, nk_module: object) -> tuple[np.ndarray, np.ndarray]:
    try:
        sig_df, _ = nk_module.eda_process(eda_filt, sampling_rate=fs)
        tonic = np.asarray(sig_df["EDA_Tonic"], dtype=np.float64)
        phasic = np.asarray(sig_df["EDA_Phasic"], dtype=np.float64)
    except Exception:
        clean = nk_module.eda_clean(eda_filt, sampling_rate=fs)
        phasic_df = nk_module.eda_phasic(clean, sampling_rate=fs)
        tonic = np.asarray(phasic_df["EDA_Tonic"], dtype=np.float64)
        phasic = np.asarray(phasic_df["EDA_Phasic"], dtype=np.float64)
    return tonic, phasic


def extract_eda_window_features(tonic_win: np.ndarray, phasic_win: np.ndarray, fs: int) -> np.ndarray:
    prominence = max(1e-6, 0.1 * float(np.nanstd(phasic_win)))
    peaks, props = find_peaks(phasic_win, prominence=prominence)
    peak_amplitudes = props.get("prominences", np.array([], dtype=np.float64))

    rise_times: list[float] = []
    for p in peaks:
        if p <= 1:
            continue
        start = max(0, p - int(4 * fs))
        pre = phasic_win[start:p]
        if len(pre) == 0:
            continue
        trough_idx = start + int(np.argmin(pre))
        rt = (p - trough_idx) / float(fs)
        if rt >= 0:
            rise_times.append(rt)

    max_peak_amp = float(np.max(peak_amplitudes)) if len(peak_amplitudes) > 0 else 0.0
    std_rise = float(np.std(rise_times)) if len(rise_times) > 1 else 0.0
    return np.array(
        [
            float(np.mean(tonic_win)),
            float(np.max(tonic_win)),
            max_peak_amp,
            std_rise,
            float(len(peaks)),
        ],
        dtype=np.float32,
    )


def slide_hrv_windows(
    signal: np.ndarray,
    fs: int,
    peaks: np.ndarray,
    binary_label: int,
    block_id: int,
    block_type: str,
    window_sec: int,
    stride_sec: int,
    nk_module: object,
    *,
    min_beats: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sliding windows with HRV features from precomputed peaks."""
    window_samples = int(window_sec * fs)
    stride_samples = int(stride_sec * fs)
    if len(signal) < window_samples or len(peaks) < min_beats:
        return _empty_block_result(N_HRV_FEATURES)

    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    rows_block: list[int] = []
    rows_type: list[str] = []
    rows_time: list[float] = []

    for start in range(0, len(signal) - window_samples + 1, stride_samples):
        end = start + window_samples
        mask = (peaks >= start) & (peaks < end)
        rpeaks_local = peaks[mask] - start
        if len(rpeaks_local) < min_beats:
            continue
        feats = compute_hrv_features(rpeaks_local, fs, nk_module)
        if not np.isfinite(feats).any():
            continue
        rows_x.append(feats)
        rows_y.append(binary_label)
        rows_block.append(block_id)
        rows_type.append(block_type)
        rows_time.append(start / fs)

    return _pack_rows(rows_x, rows_y, rows_block, rows_type, rows_time, N_HRV_FEATURES)


def slide_eda_windows(
    tonic: np.ndarray,
    phasic: np.ndarray,
    fs: int,
    binary_label: int,
    block_id: int,
    block_type: str,
    window_sec: int,
    stride_sec: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    window_samples = int(window_sec * fs)
    stride_samples = int(stride_sec * fs)
    if len(tonic) < window_samples:
        return _empty_block_result(N_EDA_FEATURES)

    rows_x: list[np.ndarray] = []
    rows_y: list[int] = []
    rows_block: list[int] = []
    rows_type: list[str] = []
    rows_time: list[float] = []

    for start in range(0, len(tonic) - window_samples + 1, stride_samples):
        end = start + window_samples
        feats = extract_eda_window_features(
            tonic[start:end], phasic[start:end], fs
        )
        if not np.isfinite(feats).any():
            continue
        rows_x.append(feats)
        rows_y.append(binary_label)
        rows_block.append(block_id)
        rows_type.append(block_type)
        rows_time.append(start / fs)

    return _pack_rows(rows_x, rows_y, rows_block, rows_type, rows_time, N_EDA_FEATURES)


def _empty_block_result(n_feat: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    empty_i = np.zeros((0,), dtype=np.int64)
    return (
        np.zeros((0, n_feat), dtype=np.float32),
        empty_i,
        empty_i,
        np.zeros((0,), dtype="<U64"),
        np.zeros((0,), dtype=np.float64),
    )


def _pack_rows(
    rows_x: list[np.ndarray],
    rows_y: list[int],
    rows_block: list[int],
    rows_type: list[str],
    rows_time: list[float],
    n_feat: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not rows_x:
        return _empty_block_result(n_feat)
    return (
        np.stack(rows_x).astype(np.float32),
        np.asarray(rows_y, dtype=np.int64),
        np.asarray(rows_block, dtype=np.int64),
        np.asarray(rows_type, dtype="<U64"),
        np.asarray(rows_time, dtype=np.float64),
    )


def participant_baseline_stats(
    clas_root: Path,
    participant_id: int,
    *,
    signal_loader,
    scheme: str = "high_vs_low",
    min_quality: float | None = None,
    quality_modality: str = "ecg",
) -> tuple[float, float]:
    """Mean and std of raw signal over all low-load (y==0) blocks."""
    vals: list[np.ndarray] = []
    for blk in iter_labeled_blocks(
        clas_root,
        participant_id,
        scheme=scheme,
        min_quality=min_quality,
        quality_modality=quality_modality,
    ):
        if blk.y != 0:
            continue
        vals.append(np.asarray(signal_loader(blk), dtype=np.float64).reshape(-1))
    if not vals:
        return 0.0, 1.0
    cat = np.concatenate(vals)
    return float(np.nanmean(cat)), float(np.nanstd(cat))


def save_participant_npz(
    path: Path,
    *,
    X: np.ndarray,
    y: np.ndarray,
    block_id: np.ndarray,
    block_type: np.ndarray,
    timestamp_sec: np.ndarray,
    participant_id: int,
    feature_names: list[str],
    modality: str,
    window_sec: int,
    stride_sec: int,
    extra: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "X": X,
        "y": y,
        "block_id": block_id,
        "block_type": block_type,
        "timestamp_sec": timestamp_sec,
        "is_low_load": y == 0,
        "participant_id": np.int64(participant_id),
        "feature_names": np.asarray(feature_names, dtype="<U32"),
        "modality": np.asarray(modality),
        "window_sec": np.int64(window_sec),
        "stride_sec": np.int64(stride_sec),
    }
    if extra:
        payload.update(extra)
    np.savez_compressed(path, **payload)


def load_preprocessed_participants(
    processed_root: Path,
    modality: str,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load Part*.npz under processed_root/modality -> {pid: (X (N,1,F), y)}."""
    sub = processed_root / PROCESSED_SUBDIRS.get(modality, modality)
    if not sub.is_dir():
        sub = processed_root
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for p in sorted(sub.glob("Part*.npz")):
        m = p.stem
        if not m.startswith("Part"):
            continue
        try:
            pid = int(m[4:])
        except ValueError:
            continue
        z = np.load(p, allow_pickle=False)
        X = np.asarray(z["X"], dtype=np.float32)
        y = np.asarray(z["y"], dtype=np.int64)
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        out[pid] = (X, y)
    return out


def iter_blocks(
    clas_root: Path,
    participant_id: int,
    **kwargs,
) -> Iterator[BlockRow]:
    yield from iter_labeled_blocks(clas_root, participant_id, **kwargs)
