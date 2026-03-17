#!/usr/bin/env python3
"""Raw BVP preprocessing for WESAD (CardioMind-compatible windows).

Core behavior:
- Uses wrist BVP from Empatica E4 (64 Hz).
- Applies 0.5–8 Hz band-pass Butterworth filtering to isolate cardiac pulse.
- Baseline-normalizes using initial baseline phase (label=1) z-score.
- Extracts 20 s raw waveform windows with 1 s stride.
- Retains WESAD labels {1,2,3,4}; assigns window label by majority vote.

Output per subject .pt:
    bvp_windows: (num_windows, 1280) float32   — raw filtered BVP samples
    labels:      (num_windows,)      int32      — majority label per window
    timestamps_sec: (num_windows,)   float64    — window start time (seconds)
    subject_id:  int
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, filtfilt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BVP_HZ = 64
WINDOW_SEC = 20
STRIDE_SEC = 1
WINDOW_SAMPLES = WINDOW_SEC * BVP_HZ       # 1280
STRIDE_SAMPLES = STRIDE_SEC * BVP_HZ       # 64
WESAD_LABEL_HZ = 700


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_wesad_wrist_bvp_and_labels(
    wesad_root: str, subject_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Load raw wrist BVP (64 Hz) and chest labels (700 Hz) from WESAD pickle."""
    pkl_path = Path(wesad_root) / f"S{subject_id}" / f"S{subject_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"WESAD pickle not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    bvp = np.asarray(raw["signal"]["wrist"]["BVP"], dtype=np.float64).reshape(-1)
    labels_700hz = np.asarray(raw["label"], dtype=np.int32).reshape(-1)
    return bvp, labels_700hz


def _labels_700hz_to_64hz(labels_700hz: np.ndarray, target_len: int) -> np.ndarray:
    """Down-sample 700 Hz labels to 64 Hz by nearest-neighbour mapping."""
    t = np.arange(target_len, dtype=np.float64) / BVP_HZ
    raw_idx = np.clip((t * WESAD_LABEL_HZ).astype(int), 0, len(labels_700hz) - 1)
    return labels_700hz[raw_idx].astype(np.int32)


def _bandpass_butterworth(
    x: np.ndarray,
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 8.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase band-pass Butterworth filter."""
    nyq = 0.5 * fs
    low = low_hz / nyq
    high = min(high_hz / nyq, 0.99)
    b, a = butter(order, [low, high], btype="band")
    return filtfilt(b, a, x).astype(np.float64)


def _initial_baseline_mask(labels: np.ndarray, baseline_label: int = 1) -> np.ndarray:
    """Return mask for the first contiguous baseline run."""
    run_end = 0
    while run_end < len(labels) and labels[run_end] == baseline_label:
        run_end += 1
    mask = np.zeros(len(labels), dtype=bool)
    if run_end > 0:
        mask[:run_end] = True
    else:
        mask = labels == baseline_label
    return mask


def _zscore_normalize(x: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    """Subject-level z-score normalization using baseline segment."""
    if not baseline_mask.any():
        baseline_mask = np.ones(len(x), dtype=bool)
    baseline = x[baseline_mask]
    mu = float(np.nanmean(baseline))
    sd = float(np.nanstd(baseline))
    if not np.isfinite(sd) or sd < 1e-8:
        sd = 1.0
    out = (x - mu) / sd
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def _majority_label(window_labels: np.ndarray) -> int:
    """Return the most frequent label in {1,2,3,4}; 0 if none present."""
    valid = window_labels[np.isin(window_labels, [1, 2, 3, 4])]
    if len(valid) == 0:
        return 0
    vals, counts = np.unique(valid, return_counts=True)
    return int(vals[np.argmax(counts)])


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_subject_bvp(
    wesad_root: str,
    subject_id: int,
) -> dict[str, np.ndarray | int]:
    """Process one WESAD subject: load BVP, filter, normalize, window.

    Returns
    -------
    dict with keys:
        bvp_windows:    (num_windows, 1280) float32
        labels:         (num_windows,) int32  in {1,2,3,4}
        timestamps_sec: (num_windows,) float64
        subject_id:     int
    """
    bvp_raw, labels_700hz = _load_wesad_wrist_bvp_and_labels(wesad_root, subject_id)

    # Map labels to BVP sample rate
    labels_64hz = _labels_700hz_to_64hz(labels_700hz, target_len=len(bvp_raw))

    # Baseline normalization (z-score from initial baseline segment)
    baseline_mask = _initial_baseline_mask(labels_64hz, baseline_label=1)
    bvp_norm = _zscore_normalize(bvp_raw, baseline_mask)

    # Band-pass filter
    bvp_filt = _bandpass_butterworth(bvp_norm, fs=BVP_HZ, low_hz=0.5, high_hz=8.0)

    # Sliding windows
    windows, lbls, ts = [], [], []
    n_windows = (len(bvp_filt) - WINDOW_SAMPLES) // STRIDE_SAMPLES + 1

    for i in range(max(0, n_windows)):
        start = i * STRIDE_SAMPLES
        end = start + WINDOW_SAMPLES
        w_labels = labels_64hz[start:end]

        # Keep only windows fully within labels {1,2,3,4}
        if not np.all(np.isin(w_labels, [1, 2, 3, 4])):
            continue

        label = _majority_label(w_labels)
        if label == 0:
            continue

        windows.append(bvp_filt[start:end])
        lbls.append(label)
        ts.append(start / BVP_HZ)

    if windows:
        bvp_windows = np.vstack(windows).astype(np.float32)
    else:
        bvp_windows = np.zeros((0, WINDOW_SAMPLES), dtype=np.float32)
    labels = np.asarray(lbls, dtype=np.int32) if lbls else np.zeros((0,), dtype=np.int32)
    timestamps_sec = np.asarray(ts, dtype=np.float64) if ts else np.zeros((0,), dtype=np.float64)

    return {
        "bvp_windows": bvp_windows,
        "labels": labels,
        "timestamps_sec": timestamps_sec,
        "subject_id": subject_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Raw BVP preprocessing for WESAD")
    parser.add_argument("--wesad-root", default="data/raw/WESAD")
    parser.add_argument(
        "--subjects",
        nargs="*",
        type=int,
        default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17],
    )
    parser.add_argument("--out-root", default="data/processed_bvp_raw")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for sid in args.subjects:
        try:
            result = process_subject_bvp(wesad_root=args.wesad_root, subject_id=sid)
        except FileNotFoundError as e:
            print(f"S{sid}: SKIPPED — {e}")
            continue

        save_path = out_root / f"S{sid}.pt"
        torch.save(
            {
                "subject_id": result["subject_id"],
                "bvp_windows": result["bvp_windows"],
                "labels": result["labels"],
                "timestamps_sec": result["timestamps_sec"],
            },
            save_path,
        )

        n = result["bvp_windows"].shape[0]
        lbl_counts = {
            k: int(np.sum(result["labels"] == k)) for k in [1, 2, 3, 4]
        }
        print(
            f"S{sid}: windows={n} shape={result['bvp_windows'].shape} "
            f"labels={lbl_counts} -> {save_path}"
        )

    print(f"\nDone. Saved raw BVP files to: {out_root}")


if __name__ == "__main__":
    main()
