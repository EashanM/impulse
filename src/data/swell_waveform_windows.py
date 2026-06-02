"""
Sliding windows on SWELL Poly5 waveforms ``(C, T)`` for a second analysis path.

Typical use: read ``.S00`` with :func:`poly5_portilab.read_poly5`, build windows
``(N, win_samples, C)``, optionally attach binary labels from the same subject's
minute ``S*.pt`` using :func:`window_center_labels`.
"""

from __future__ import annotations

import numpy as np


def waveform_sliding_windows(
    x: np.ndarray,
    fs: float,
    win_sec: float,
    hop_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Non-overlapping or overlapping sliding windows along time.

    Parameters
    ----------
    x
        Shape ``(C, T)``, any float dtype (output is ``float32``).
    fs
        Sample rate in Hz (from ``Poly5ReadResult.sample_rate_hz``).
    win_sec, hop_sec
        Window length and hop in seconds. Hop is rounded to at least one sample.

    Returns
    -------
    windows
        Shape ``(N, win_samples, C)`` — same axis order as minute tensors
        ``(B, T, F)`` in ``swell_baselines`` if you treat ``F = C`` channels.
    start_sample
        Shape ``(N,)``, ``int64``, index of the first sample in each window.
    """
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"x must be 2D (C, T), got shape {x.shape}")
    c, t = int(x.shape[0]), int(x.shape[1])
    fs_f = float(fs)
    win = max(1, int(round(float(win_sec) * fs_f)))
    hop = max(1, int(round(float(hop_sec) * fs_f)))
    if t < win:
        return np.zeros((0, win, c), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    starts = np.arange(0, t - win + 1, hop, dtype=np.int64)
    if starts.size == 0:
        return np.zeros((0, win, c), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    idx = starts[:, None] + np.arange(win, dtype=np.int64)
    # x[:, idx] -> (C, N, win); -> (N, win, C)
    out = np.transpose(x[:, idx], (1, 2, 0)).astype(np.float32, copy=False)
    return out, starts


def window_center_row_indices(
    timestamps_sec: np.ndarray,
    window_start_sec: np.ndarray,
    win_sec: float,
) -> np.ndarray:
    """
    For each window, index of the minute row whose timestamp is last ``<=`` window center.

    Returns indices in ``[0, len(timestamps_sec) - 1]`` suitable for fancy-indexing
    parallel arrays (``labels``, ``condition_code``, …).
    """
    ts = np.asarray(timestamps_sec, dtype=np.float64).reshape(-1)
    if ts.size == 0:
        raise ValueError("empty timestamps_sec")
    centers = np.asarray(window_start_sec, dtype=np.float64).reshape(-1) + float(win_sec) / 2.0
    idx = np.searchsorted(ts, centers, side="right") - 1
    return np.clip(idx, 0, ts.shape[0] - 1).astype(np.int64, copy=False)


def window_center_labels(
    timestamps_sec: np.ndarray,
    labels: np.ndarray,
    window_start_sec: np.ndarray,
    win_sec: float,
) -> np.ndarray:
    """
    Map each window to the minute row whose timestamp is last ``<=`` window center.

    ``timestamps_sec`` should match the minute tensor (e.g. from ``S*.pt``), same
    length as ``labels``. Window times are in the same clock as those stamps
    (usually recording-relative seconds starting near 0).
    """
    ts = np.asarray(timestamps_sec, dtype=np.float64).reshape(-1)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if ts.shape[0] != y.shape[0]:
        raise ValueError(f"timestamps_sec ({ts.shape[0]}) and labels ({y.shape[0]}) length mismatch")
    if ts.size == 0:
        raise ValueError("empty timestamps / labels")
    idx = window_center_row_indices(ts, window_start_sec, win_sec)
    return y[idx]


def window_starts_seconds(start_sample: np.ndarray, fs: float) -> np.ndarray:
    """Convert start indices to seconds since the start of the waveform array."""
    return np.asarray(start_sample, dtype=np.float64) / float(fs)
