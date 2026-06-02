"""
SWELL-KW label conventions for minute-level physiology feature tables.

The CSV columns `Condition` and `C` are documented in the SWELL materials; this
module fixes a single binary stress definition for baselines that expect y in {0,1}.
"""

from __future__ import annotations

import numpy as np

# Per-minute protocol label from `Condition` column
STRESS_CONDITIONS: frozenset[str] = frozenset({"T", "I"})
"""Time pressure (T) and interruption (I) -> positive (stress) class."""

NON_STRESS_CONDITIONS: frozenset[str] = frozenset({"N", "R"})
"""Neutral (N) and rest (R) -> negative (non-stress) class."""

# Integer codes stored alongside binary labels for multiclass / analysis
CONDITION_TO_CODE: dict[str, int] = {"R": 0, "N": 1, "T": 2, "I": 3}
CODE_TO_CONDITION: dict[int, str] = {v: k for k, v in CONDITION_TO_CODE.items()}

MISSING_SENTINEL = 999
"""SWELL uses 999 for missing HR / RMSSD in some exports."""


def condition_to_binary_label(condition: str) -> int:
    """
    Map SWELL `Condition` to binary label.

    Returns
    -------
    0
        Non-stress: N or R
    1
        Stress: T or I
    """
    c = str(condition).strip().upper()
    if c in NON_STRESS_CONDITIONS:
        return 0
    if c in STRESS_CONDITIONS:
        return 1
    raise ValueError(f"Unknown SWELL Condition={condition!r}; expected one of {CONDITION_TO_CODE.keys()}")


def condition_to_code(condition: str) -> int:
    """Map `Condition` letter to a stable int 0..3."""
    c = str(condition).strip().upper()
    if c not in CONDITION_TO_CODE:
        raise ValueError(f"Unknown SWELL Condition={condition!r}")
    return CONDITION_TO_CODE[c]


def clean_physiology_values(x: np.ndarray, sentinel: int = MISSING_SENTINEL) -> np.ndarray:
    """Replace SWELL missing sentinel with NaN (float)."""
    out = np.asarray(x, dtype=np.float64)
    out = out.copy()
    out[out == float(sentinel)] = np.nan
    return out


def impute_rowwise_ffill_zero(features: np.ndarray) -> np.ndarray:
    """Same strategy as WESAD preprocessor: forward-fill along time, then zero."""
    result = features.copy()
    for col in range(result.shape[1]):
        mask = np.isnan(result[:, col])
        if not mask.any():
            continue
        valid_idx = np.where(~mask)[0]
        if len(valid_idx) > 0:
            for i in range(len(result)):
                if mask[i]:
                    prev = valid_idx[valid_idx < i]
                    if len(prev) > 0:
                        result[i, col] = result[prev[-1], col]
        result[np.isnan(result[:, col]), col] = 0.0
    return result
