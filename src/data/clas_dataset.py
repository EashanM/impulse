"""CLAS database helpers: participants, block metadata, file discovery, windowing.

Expected layout (default root: data/CLAS_Database/CLAS):
  Block_details/Part{N}_Block_Details.csv
  Participants/Part{N}/by_block/{block}_ecg*.csv, {block}_gsr_ppg*.csv
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths & discovery
# ---------------------------------------------------------------------------

DEFAULT_CLAS_ROOT = Path("data/CLAS_Database/CLAS")


def clas_root_from_data_dir(data_dir: Path | str) -> Path:
    return Path(data_dir) / "CLAS_Database" / "CLAS"


def discover_participant_ids(clas_root: Path) -> list[int]:
    """Return sorted numeric participant ids (excludes duplicates like 'Part25 - Copy')."""
    root = clas_root / "Participants"
    if not root.is_dir():
        return []
    ids: list[int] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = re.fullmatch(r"Part(\d+)", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def block_details_path(clas_root: Path, participant_id: int) -> Path:
    return clas_root / "Block_details" / f"Part{participant_id}_Block_Details.csv"


def by_block_dir(clas_root: Path, participant_id: int) -> Path:
    return clas_root / "Participants" / f"Part{participant_id}" / "by_block"


# ---------------------------------------------------------------------------
# Label schemes
# ---------------------------------------------------------------------------

HIGH_LOAD_BLOCK_TYPES = frozenset(
    {
        "Math Test",
        "Math Test Response",
        "Stroop Test",
        "Stroop Test Response",
        "IQ Test",
        "IQ Test Response",
        "Video clip",
        "Pictures",
    }
)

LOW_LOAD_BLOCK_TYPES = frozenset(
    {
        "Baseline",
        "Neutral",
    }
)


def block_type_to_binary_label(block_type: str, scheme: str = "high_vs_low") -> int | None:
    """Map CLAS block type to a binary label, or None if unknown / unused."""
    t = block_type.strip()
    if scheme == "high_vs_low":
        if t in HIGH_LOAD_BLOCK_TYPES:
            return 1
        if t in LOW_LOAD_BLOCK_TYPES:
            return 0
        return None
    raise ValueError(f"Unknown label scheme: {scheme!r}")


# ---------------------------------------------------------------------------
# Block metadata table
# ---------------------------------------------------------------------------


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_block_details_csv(path: Path) -> pd.DataFrame:
    """Load one Part*_Block_Details.csv with trimmed column names."""
    df = pd.read_csv(path)
    df = _normalize_column_names(df)
    df = df.loc[:, [c for c in df.columns if c and not str(c).startswith("Unnamed")]]
    return df


def find_segment_csv(by_block: Path, block_id: int, kind: str) -> Path | None:
    """Locate by_block CSV for a block id. kind is 'ecg' or 'gsr_ppg'."""
    if kind not in {"ecg", "gsr_ppg"}:
        raise ValueError(kind)
    pat = f"{block_id}_{kind}*.csv"
    matches = sorted(by_block.glob(pat))
    if not matches:
        return None
    return matches[0]


@dataclass
class BlockRow:
    participant_id: int
    block_id: int
    block_type: str
    length_sec: float
    eda_quality: float
    ecg_quality: float
    ppg_quality: float
    ecg_path: Path
    gsr_ppg_path: Path
    y: int


def iter_labeled_blocks(
    clas_root: Path,
    participant_id: int,
    scheme: str = "high_vs_low",
    min_quality: float | None = None,
    quality_modality: str = "ecg",
) -> Iterator[BlockRow]:
    """Yield labeled blocks that have matching by_block CSV pairs."""
    bd_path = block_details_path(clas_root, participant_id)
    if not bd_path.is_file():
        return
    bb = by_block_dir(clas_root, participant_id)
    if not bb.is_dir():
        return
    df = load_block_details_csv(bd_path)
    required = {"Block", "Block Type", "Length(s)", "EDA Quality", "ECG Quality", "PPG Quality"}
    missing = required - set(df.columns)
    if missing:
        return
    for _, row in df.iterrows():
        try:
            bid = int(row["Block"])
        except (TypeError, ValueError):
            continue
        btype = str(row["Block Type"]).strip()
        y = block_type_to_binary_label(btype, scheme=scheme)
        if y is None:
            continue
        try:
            length_sec = float(row["Length(s)"])
        except (TypeError, ValueError):
            continue
        eda_q = float(row["EDA Quality"])
        ecg_q = float(row["ECG Quality"])
        ppg_q = float(row["PPG Quality"])
        if min_quality is not None:
            q = {"ecg": ecg_q, "eda": eda_q, "ppg": ppg_q}.get(quality_modality, ecg_q)
            if q < min_quality:
                continue
        ecg_p = find_segment_csv(bb, bid, "ecg")
        gsr_p = find_segment_csv(bb, bid, "gsr_ppg")
        if ecg_p is None or gsr_p is None:
            continue
        yield BlockRow(
            participant_id=participant_id,
            block_id=bid,
            block_type=btype,
            length_sec=length_sec,
            eda_quality=eda_q,
            ecg_quality=ecg_q,
            ppg_quality=ppg_q,
            ecg_path=ecg_p,
            gsr_ppg_path=gsr_p,
            y=y,
        )


# ---------------------------------------------------------------------------
# Numeric CSV loading
# ---------------------------------------------------------------------------


def load_ecg_matrix(path: Path) -> np.ndarray:
    """Return float array (T, 2) for columns ecg1, ecg2 (numeric rows only)."""
    df = pd.read_csv(path, low_memory=False)
    if "ecg1" not in df.columns or "ecg2" not in df.columns:
        raise ValueError(f"Missing ecg columns in {path}")
    e1 = pd.to_numeric(df["ecg1"], errors="coerce")
    e2 = pd.to_numeric(df["ecg2"], errors="coerce")
    mask = e1.notna() & e2.notna()
    out = np.stack([e1[mask].to_numpy(np.float64), e2[mask].to_numpy(np.float64)], axis=1)
    return out.astype(np.float32)


def load_gsr_ppg_matrix(path: Path) -> np.ndarray:
    """Return float array (T, 5): accelx, accely, accelz, ppg, gsr."""
    df = pd.read_csv(path, low_memory=False)
    cols = ["accelx", "accely", "accelz", "ppg", "gsr"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {path}")
    mask = pd.Series(True, index=df.index)
    numeric: dict[str, pd.Series] = {}
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce")
        numeric[c] = v
        mask &= v.notna()
    arrs = [numeric[c][mask].to_numpy(np.float64) for c in cols]
    return np.stack(arrs, axis=1).astype(np.float32)


def estimate_fs(n_samples: int, length_sec: float) -> float:
    if length_sec <= 0:
        return 1.0
    return float(n_samples) / float(length_sec)


def resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    """Linearly resample 1D signal x to target_len samples."""
    if target_len <= 1:
        return x[:1].copy()
    n = x.shape[0]
    if n == target_len:
        return x.copy()
    xp = np.linspace(0.0, 1.0, num=n, dtype=np.float64)
    xnew = np.linspace(0.0, 1.0, num=target_len, dtype=np.float64)
    return np.interp(xnew, xp, x.astype(np.float64)).astype(np.float32)


def resample_multichannel(x: np.ndarray, target_len: int) -> np.ndarray:
    """x shape (T, C) -> (target_len, C)."""
    c = x.shape[1]
    out = np.empty((target_len, c), dtype=np.float32)
    for j in range(c):
        out[:, j] = resample_1d(x[:, j], target_len)
    return out


def extract_windows_from_block(
    signal_tc: np.ndarray,
    length_sec: float,
    window_sec: float,
    stride_sec: float,
    target_len: int,
) -> np.ndarray:
    """Tile windows along time, resample each window to target_len.

    Parameters
    ----------
    signal_tc
        (T, C) float32.
    length_sec
        Nominal duration from metadata (used for estimated fs).
    target_len
        Fixed time length after resampling (for batched encoders).

    Returns
    -------
    np.ndarray
        (N, C, target_len) float32; N==0 if signal too short.
    """
    t, c = signal_tc.shape
    if t < 2:
        return np.zeros((0, c, target_len), dtype=np.float32)
    fs = estimate_fs(t, length_sec)
    win = max(1, int(round(fs * window_sec)))
    stride = max(1, int(round(fs * stride_sec)))
    if win > t:
        # whole block as one window (padded via resample shrink)
        w = resample_multichannel(signal_tc, target_len)
        return w.T[np.newaxis, :, :]
    windows: list[np.ndarray] = []
    for start in range(0, t - win + 1, stride):
        sl = signal_tc[start : start + win]
        windows.append(resample_multichannel(sl, target_len).T)
    if not windows:
        return np.zeros((0, c, target_len), dtype=np.float32)
    return np.stack(windows, axis=0).astype(np.float32)


def modality_tensor(block: BlockRow, modality: str) -> np.ndarray:
    """Load (T, C) for modality: ecg2 | ppg | gsr | accel3."""
    if modality == "ecg2":
        return load_ecg_matrix(block.ecg_path)
    g = load_gsr_ppg_matrix(block.gsr_ppg_path)
    # columns: accelx, accely, accelz, ppg, gsr
    if modality == "ppg":
        return g[:, 3:4]
    if modality == "gsr":
        return g[:, 4:5]
    if modality == "accel3":
        return g[:, 0:3]
    raise ValueError(f"Unknown modality {modality!r}")


def collect_windows_for_participant(
    clas_root: Path,
    participant_id: int,
    modality: str,
    window_sec: float,
    stride_sec: float,
    target_len: int,
    scheme: str = "high_vs_low",
    min_quality: float | None = None,
    quality_modality: str = "ecg",
) -> tuple[np.ndarray, np.ndarray]:
    """Return X (N, C, L), y (N,) for all labeled blocks of one participant."""
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for blk in iter_labeled_blocks(
        clas_root,
        participant_id,
        scheme=scheme,
        min_quality=min_quality,
        quality_modality=quality_modality,
    ):
        sig = modality_tensor(blk, modality)
        w = extract_windows_from_block(sig, blk.length_sec, window_sec, stride_sec, target_len)
        if w.shape[0] == 0:
            continue
        xs.append(w)
        ys.extend([blk.y] * w.shape[0])
    if not xs:
        return np.zeros((0, 1, target_len), np.float32), np.zeros((0,), np.int64)
    X = np.concatenate(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    return X, y
