"""
Pick ECG (cardiac) and EDA (skin conductance) traces from SWELL Poly5 waveforms.

The PortiLab / Poly5 file already contains **separate** channel streams. You do
not need blind source separation: **choose the correct rows** of ``data_uv``
(shape ``(C, T)`` in microvolts).

SWELL's MATLAB helper ``myDoReadData.m`` (1-based **logical** channel indices
on the paired TMSi layout) maps **7 → skin (EDA)**, **8 → heart (ECG)**.

After that, if you want **scalar features** like the minute CSV (HR, RMSSD,
SCL), you still need **signal processing** (R-peak / HRV pipeline on ECG;
filtering / modeling on EDA). This module only returns the continuous waveforms.
"""

from __future__ import annotations

import numpy as np

# 1-based indices as in SWELL MATLAB ``myDoReadData.m`` (logical / paired layout).
SWELL_EDA_LOGICAL_1BASED_MATLAB = 7
SWELL_ECG_LOGICAL_1BASED_MATLAB = 8


def pick_ecg_eda_uv(
    data_uv: np.ndarray,
    *,
    eda_logical_1based: int = SWELL_EDA_LOGICAL_1BASED_MATLAB,
    ecg_logical_1based: int = SWELL_ECG_LOGICAL_1BASED_MATLAB,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return ``(eda_uV, ecg_uV)`` each shape ``(T,)`` from ``data_uv`` ``(C, T)``.

    Uses zero-based row indices ``eda_logical_1based - 1`` and
    ``ecg_logical_1based - 1``.

    Notes
    -----
    If you used ``read_poly5(..., drop_empty_channels=True)`` (the default),
    row indices **no longer match** MATLAB's 1..NS/2 table whenever empty
    channels were removed. For drop-in parity with SWELL scripts, re-read with
    ``drop_empty_channels=False``, call this helper, then drop unused rows
    yourself if needed.
    """
    x = np.asarray(data_uv, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"data_uv must be (C, T), got shape {x.shape}")
    eda_i = int(eda_logical_1based) - 1
    ecg_i = int(ecg_logical_1based) - 1
    c = x.shape[0]
    if eda_i < 0 or ecg_i < 0 or eda_i >= c or ecg_i >= c:
        raise ValueError(
            f"Channel indices out of range: need rows {eda_i} and {ecg_i} but C={c}. "
            "See module docstring: use read_poly5(..., drop_empty_channels=False) "
            "when matching SWELL MATLAB indices."
        )
    return x[eda_i, :].copy(), x[ecg_i, :].copy()
