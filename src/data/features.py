"""
Feature extraction for cardiac (ECG) and somatic (ACC/Temp/Resp) signals.

All functions operate on a single window of raw signal at native sampling rate.
Features are computed per window and returned as scalar values.
If a feature cannot be computed (e.g., too few R-peaks), NaN is returned.

Cardiac agent features: mean_hr, pnn50, rmssd
Somatic agent features: acc_variance, acc_mag_slope, temp_slope, resp_irregularity
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from scipy.signal import butter, filtfilt

# Suppress neurokit2 warnings during batch processing
warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Cardiac features (from ECG at native Hz, typically 700 Hz)
# ---------------------------------------------------------------------------

def _bandpass_ecg(ecg_segment: np.ndarray, fs: int, low_hz: float = 3.0, high_hz: float = 45.0) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter to ECG."""
    if len(ecg_segment) < max(8, int(0.2 * fs)):
        return np.asarray(ecg_segment, dtype=np.float64)

    nyq = 0.5 * fs
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999999)
    if low >= high:
        return np.asarray(ecg_segment, dtype=np.float64)

    b, a = butter(4, [low, high], btype="bandpass")
    return filtfilt(b, a, np.asarray(ecg_segment, dtype=np.float64))


def _is_peak_series_plausible(r_peaks: np.ndarray, fs: int) -> bool:
    """Check if detected peaks produce physiologically plausible RR intervals."""
    if len(r_peaks) < 3:
        return False
    rr_sec = np.diff(r_peaks) / float(fs)
    if len(rr_sec) < 2:
        return False
    if np.any(rr_sec < 0.25) or np.any(rr_sec > 2.0):
        return False
    median_rr = float(np.median(rr_sec))
    return 0.35 <= median_rr <= 1.5


def _detect_with_neurokit(filtered_ecg: np.ndarray, fs: int) -> np.ndarray:
    import neurokit2 as nk

    cleaned = nk.ecg_clean(filtered_ecg, sampling_rate=fs)
    _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
    peaks = info.get("ECG_R_Peaks", np.array([], dtype=int))
    return np.asarray(peaks, dtype=int)


def detect_r_peaks(ecg_segment: np.ndarray, fs: int = 700) -> np.ndarray:
    """
        Detect R-peaks in an ECG segment using HeartPy-first strategy.

        Steps:
            1) Explicit 3-45 Hz bandpass filtering
            2) HeartPy adaptive thresholding peak detection
            3) NeuroKit2 fallback when HeartPy output is uncertain/invalid

    Returns array of R-peak sample indices within the segment.
    Falls back to empty array if detection fails.
    """
    filtered = _bandpass_ecg(ecg_segment, fs=fs)
    try:
        import heartpy as hp

        wd, _ = hp.process(filtered, sample_rate=fs)
        peaks = np.asarray(wd.get("peaklist", []), dtype=int)
        peaks = peaks[(peaks >= 0) & (peaks < len(filtered))]
        peaks = np.unique(peaks)

        if _is_peak_series_plausible(peaks, fs=fs):
            return peaks

        # Uncertain HeartPy detection -> verify/fallback with NeuroKit2
        nk_peaks = _detect_with_neurokit(filtered, fs=fs)
        if _is_peak_series_plausible(nk_peaks, fs=fs):
            return nk_peaks
        return peaks if len(peaks) >= len(nk_peaks) else nk_peaks
    except Exception:
        try:
            peaks = _detect_with_neurokit(filtered, fs=fs)
            return peaks
        except Exception:
            return np.array([], dtype=int)


def compute_rr_intervals(r_peaks: np.ndarray, fs: int = 700) -> np.ndarray:
    """Convert R-peak sample indices to R-R intervals in milliseconds."""
    if len(r_peaks) < 2:
        return np.array([], dtype=np.float64)
    return np.diff(r_peaks) / fs * 1000.0


def compute_rmssd(rr_intervals: np.ndarray) -> float:
    """Root mean square of successive differences (ms)."""
    if len(rr_intervals) < 2:
        return np.nan
    diffs = np.diff(rr_intervals)
    return float(np.sqrt(np.mean(diffs ** 2)))


def compute_mean_hr(rr_intervals: np.ndarray) -> float:
    """Mean heart rate (BPM) from RR intervals."""
    if len(rr_intervals) < 1:
        return np.nan
    mean_rr_sec = np.mean(rr_intervals) / 1000.0
    if mean_rr_sec <= 0:
        return np.nan
    return 60.0 / mean_rr_sec


def compute_pnn50(rr_intervals: np.ndarray) -> float:
    """Percentage of successive RR intervals differing by more than 50 ms."""
    if len(rr_intervals) < 2:
        return np.nan
    diffs = np.abs(np.diff(rr_intervals))
    return float(np.mean(diffs > 50.0) * 100.0)


def compute_nn50(rr_intervals: np.ndarray) -> float:
    """Count of successive RR interval differences > 50 ms."""
    if len(rr_intervals) < 2:
        return np.nan
    diffs = np.abs(np.diff(rr_intervals))
    return float(np.sum(diffs > 50.0))


def compute_std_hr(rr_intervals_ms: np.ndarray) -> float:
    """Standard deviation of instantaneous heart rate in beats/s."""
    if len(rr_intervals_ms) < 2:
        return np.nan
    rr_sec = rr_intervals_ms / 1000.0
    hr_bps = 1.0 / np.maximum(rr_sec, 1e-8)
    return float(np.std(hr_bps))


def compute_mean_rr_sec(rr_intervals_ms: np.ndarray) -> float:
    """Mean RR interval in seconds."""
    if len(rr_intervals_ms) < 1:
        return np.nan
    return float(np.mean(rr_intervals_ms) / 1000.0)


def compute_std_rr_sec(rr_intervals_ms: np.ndarray) -> float:
    """Standard deviation of RR intervals in seconds."""
    if len(rr_intervals_ms) < 2:
        return np.nan
    return float(np.std(rr_intervals_ms) / 1000.0)


def compute_rmssd_sec(rr_intervals_ms: np.ndarray) -> float:
    """RMSSD in seconds."""
    rmssd_ms = compute_rmssd(rr_intervals_ms)
    if np.isnan(rmssd_ms):
        return np.nan
    return float(rmssd_ms / 1000.0)


def _rr_histogram(rr_intervals_ms: np.ndarray, bins: int = 50) -> tuple[np.ndarray, np.ndarray]:
    if len(rr_intervals_ms) < 2:
        return np.array([]), np.array([])
    hist, edges = np.histogram(rr_intervals_ms, bins=bins)
    return hist.astype(np.float64), edges.astype(np.float64)


def compute_hti(rr_intervals_ms: np.ndarray) -> float:
    """HRV triangular index (N / modal bin height)."""
    hist, _ = _rr_histogram(rr_intervals_ms)
    if len(hist) == 0:
        return np.nan
    hmax = np.max(hist)
    if hmax <= 0:
        return np.nan
    return float(len(rr_intervals_ms) / hmax)


def compute_tinn(rr_intervals_ms: np.ndarray) -> float:
    """TINN approximation (ms) as support width of RR histogram."""
    hist, edges = _rr_histogram(rr_intervals_ms)
    if len(hist) == 0:
        return np.nan
    nonzero = np.where(hist > 0)[0]
    if len(nonzero) < 2:
        return np.nan
    left_edge = edges[nonzero[0]]
    right_edge = edges[nonzero[-1] + 1]
    return float(right_edge - left_edge)


def compute_psd_rr_features(r_peaks: np.ndarray, fs: int = 700) -> tuple[float, float, float]:
    """
    Compute (mean_freq_hz, std_freq_hz, total_power_ms2) from RR tachogram PSD.

    Uses linear interpolation of RR intervals to a uniform 4 Hz grid then Welch PSD.
    """
    from scipy.signal import welch

    if len(r_peaks) < 4:
        return np.nan, np.nan, np.nan

    peak_times = r_peaks / float(fs)
    rr_sec = np.diff(peak_times)
    t_rr = peak_times[1:]
    if len(rr_sec) < 4:
        return np.nan, np.nan, np.nan

    rr_ms = rr_sec * 1000.0
    fs_interp = 4.0
    t_uniform = np.arange(t_rr[0], t_rr[-1], 1.0 / fs_interp)
    if len(t_uniform) < 8:
        return np.nan, np.nan, np.nan

    rr_uniform_ms = np.interp(t_uniform, t_rr, rr_ms)
    rr_uniform_ms = rr_uniform_ms - np.mean(rr_uniform_ms)

    if np.allclose(rr_uniform_ms, 0.0):
        return np.nan, np.nan, 0.0

    nperseg = min(256, len(rr_uniform_ms))
    freqs, psd = welch(rr_uniform_ms, fs=fs_interp, nperseg=nperseg)
    if len(freqs) == 0 or np.sum(psd) <= 0:
        return np.nan, np.nan, np.nan

    total_power = float(np.trapezoid(psd, freqs))
    weight_sum = np.sum(psd)
    mean_freq = float(np.sum(freqs * psd) / weight_sum)
    std_freq = float(np.sqrt(np.sum(((freqs - mean_freq) ** 2) * psd) / weight_sum))
    return mean_freq, std_freq, total_power


def extract_cardiac_features_cardiomind(ecg_window: np.ndarray, fs: int = 700) -> np.ndarray:
    """
    CardioMind-style ECG HRV feature vector from a single ECG window.

    Returns
    -------
    np.ndarray of shape (12,):
      [
        mean_hr_bps, std_hr_bps,
        mean_rr_s, std_rr_s, rmssd_s,
        nn50_count, pnn50_pct,
        hti, tinn_ms,
        mean_freq_hz, std_freq_hz, total_psd_power_ms2
      ]
    """
    peaks = detect_r_peaks(ecg_window, fs=fs)
    rr_ms = compute_rr_intervals(peaks, fs=fs)

    mean_hr_bps = np.nan
    if len(rr_ms) >= 1:
        rr_sec = rr_ms / 1000.0
        mean_hr_bps = float(np.mean(1.0 / np.maximum(rr_sec, 1e-8)))

    mean_freq_hz, std_freq_hz, total_psd_power_ms2 = compute_psd_rr_features(peaks, fs=fs)

    return np.array([
        mean_hr_bps,
        compute_std_hr(rr_ms),
        compute_mean_rr_sec(rr_ms),
        compute_std_rr_sec(rr_ms),
        compute_rmssd_sec(rr_ms),
        compute_nn50(rr_ms),
        compute_pnn50(rr_ms),
        compute_hti(rr_ms),
        compute_tinn(rr_ms),
        mean_freq_hz,
        std_freq_hz,
        total_psd_power_ms2,
    ], dtype=np.float64)


def extract_cardiac_features(ecg_window: np.ndarray, fs: int = 700) -> np.ndarray:
    """
    ECG window -> [mean_hr, pnn50, rmssd].

    Parameters
    ----------
    ecg_window : np.ndarray
        Raw ECG signal for one window, at native sampling rate.
    fs : int
        Sampling rate in Hz.

    Returns
    -------
    np.ndarray of shape (3,). NaN for features that cannot be computed.
    """
    peaks = detect_r_peaks(ecg_window, fs=fs)
    rr = compute_rr_intervals(peaks, fs=fs)
    return np.array([
        compute_mean_hr(rr),
        compute_pnn50(rr),
        compute_rmssd(rr),
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Somatic features (from ACC, Temp, Resp — all at native Hz, typically 700 Hz)
# ---------------------------------------------------------------------------

def compute_acc_variance(acc_window: np.ndarray) -> float:
    """
    Variance of acceleration magnitude over the window.
    High during movement (amusement), low during seated stress.

    Parameters
    ----------
    acc_window : np.ndarray, shape (N, 3)
    """
    if acc_window.ndim != 2 or acc_window.shape[1] < 3:
        return np.nan
    magnitude = np.sqrt(np.sum(acc_window ** 2, axis=1))
    return float(np.var(magnitude))


def compute_acc_mag_slope(acc_window: np.ndarray, fs: int) -> float:
    """
    Linear trend (slope) of acceleration magnitude over the window.
    Captures onset/offset of physical movement.

    Parameters
    ----------
    acc_window : np.ndarray, shape (N, 3)
    fs : int
    """
    if acc_window.ndim != 2 or acc_window.shape[1] < 3:
        return np.nan
    magnitude = np.sqrt(np.sum(acc_window ** 2, axis=1))
    n = len(magnitude)
    if n < 2:
        return np.nan
    t = np.arange(n, dtype=np.float64) / fs
    t_mean = t.mean()
    mag_mean = magnitude.mean()
    denom = np.sum((t - t_mean) ** 2)
    if denom < 1e-12:
        return 0.0
    slope = np.sum((t - t_mean) * (magnitude - mag_mean)) / denom
    return float(slope)


def compute_temp_slope(temp_window: np.ndarray, fs: int) -> float:
    """
    Linear trend of skin temperature over the window (°C/s).
    Stress causes peripheral vasoconstriction -> negative slope;
    amusement does not produce systematic temperature drift.

    Parameters
    ----------
    temp_window : np.ndarray, shape (N,)
    fs : int
    """
    n = len(temp_window)
    if n < 2:
        return np.nan
    t = np.arange(n, dtype=np.float64) / fs
    t_mean = t.mean()
    temp_mean = temp_window.mean()
    denom = np.sum((t - t_mean) ** 2)
    if denom < 1e-12:
        return 0.0
    slope = np.sum((t - t_mean) * (temp_window - temp_mean)) / denom
    return float(slope)


def compute_resp_irregularity(resp_window: np.ndarray, fs: int) -> float:
    """
    Standard deviation of breath-to-breath intervals (seconds).
    Stress disrupts regular breathing; amusement (laughter) also does,
    but the pattern differs in context with other somatic signals.

    Parameters
    ----------
    resp_window : np.ndarray, shape (N,)
    fs : int
    """
    from scipy.signal import find_peaks

    if len(resp_window) < fs * 2:
        return np.nan
    peaks, _ = find_peaks(resp_window, distance=int(fs * 1.5))
    if len(peaks) < 3:
        return np.nan
    intervals = np.diff(peaks) / fs
    return float(np.std(intervals))


def extract_somatic_features(
    acc_window: np.ndarray,
    temp_window: np.ndarray,
    resp_window: np.ndarray,
    fs: int,
) -> np.ndarray:
    """
    ACC/Temp/Resp windows -> [acc_variance, acc_mag_slope, temp_slope, resp_irregularity].

    Parameters
    ----------
    acc_window : np.ndarray, shape (N, 3) — 3-axis accelerometer
    temp_window : np.ndarray, shape (N,) — skin temperature
    resp_window : np.ndarray, shape (N,) — respiration
    fs : int — sampling rate in Hz

    Returns
    -------
    np.ndarray of shape (4,). NaN for features that cannot be computed.
    """
    return np.array([
        compute_acc_variance(acc_window),
        compute_acc_mag_slope(acc_window, fs),
        compute_temp_slope(temp_window, fs),
        compute_resp_irregularity(resp_window, fs),
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# EDA and EMG features (WESAD chest signals at 700 Hz)
# ---------------------------------------------------------------------------

def compute_eda_mean(eda_window: np.ndarray) -> float:
    """Mean electrodermal activity (μS) over the window."""
    if len(eda_window) == 0:
        return np.nan
    return float(np.nanmean(eda_window))


def compute_eda_std(eda_window: np.ndarray) -> float:
    """Standard deviation of EDA over the window."""
    if len(eda_window) < 2:
        return np.nan
    return float(np.nanstd(eda_window))


def compute_emg_mean(emg_window: np.ndarray) -> float:
    """Mean EMG amplitude over the window."""
    if len(emg_window) == 0:
        return np.nan
    return float(np.nanmean(np.abs(emg_window)))


def compute_emg_variance(emg_window: np.ndarray) -> float:
    """Variance of EMG over the window."""
    if len(emg_window) < 2:
        return np.nan
    return float(np.var(emg_window))


def extract_eda_features(eda_window: np.ndarray) -> np.ndarray:
    """EDA window -> [eda_mean, eda_std]."""
    return np.array([
        compute_eda_mean(eda_window),
        compute_eda_std(eda_window),
    ], dtype=np.float64)


def extract_emg_features(emg_window: np.ndarray) -> np.ndarray:
    """EMG window -> [emg_mean, emg_variance]."""
    return np.array([
        compute_emg_mean(emg_window),
        compute_emg_variance(emg_window),
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Feature names (for logging / column headers)
# ---------------------------------------------------------------------------

CARDIAC_FEATURE_NAMES = ["mean_hr", "pnn50", "rmssd"]
CARDIOMIND_CARDIAC_FEATURE_NAMES = [
    "mean_hr_bps",
    "std_hr_bps",
    "mean_rr_s",
    "std_rr_s",
    "rmssd_s",
    "nn50_count",
    "pnn50_pct",
    "hti",
    "tinn_ms",
    "mean_freq_psd_hz",
    "std_freq_psd_hz",
    "total_psd_power_ms2",
]
SOMATIC_FEATURE_NAMES = ["acc_variance", "acc_mag_slope", "temp_slope", "resp_irregularity"]
EDA_FEATURE_NAMES = ["eda_mean", "eda_std"]
EMG_FEATURE_NAMES = ["emg_mean", "emg_variance"]
