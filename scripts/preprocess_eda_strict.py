#!/usr/bin/env python3
"""Strict EDA preprocessing for WESAD (CardioMind-compatible labels/windows).

Core behavior:
- Uses wrist EDA from Empatica E4 (4 Hz) and upsamples to 8 Hz.
- Applies low-pass Butterworth filtering (requested 5 Hz; clamped below Nyquist when needed).
- Decomposes EDA into tonic/phasic streams using neurokit2.
- Extracts 5 dominant EDA features over 20 s windows with 1 s stride.
- Retains WESAD labels {1,2,3,4} and assigns window label by majority vote.
- Applies subject-specific baseline normalization from initial baseline phase (label=1)
  to the continuous EDA signal before decomposition.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Literal

import neurokit2 as nk
import numpy as np
import torch
from scipy.signal import butter, filtfilt, find_peaks, resample_poly

EDA_FEATURE_NAMES = [
    "mean_tonic",
    "max_tonic",
    "max_phasic_peak_amp",
    "std_phasic_rise_time",
    "num_phasic_peaks",
]


def _load_wesad_wrist_eda_and_labels(wesad_root: str, subject_id: int) -> tuple[np.ndarray, np.ndarray]:
    pkl_path = Path(wesad_root) / f"S{subject_id}" / f"S{subject_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"WESAD pickle not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        raw = pickle.load(f, encoding="latin1")

    wrist_eda = np.asarray(raw["signal"]["wrist"]["EDA"], dtype=np.float64).reshape(-1)
    labels_700hz = np.asarray(raw["label"], dtype=np.int32).reshape(-1)
    return wrist_eda, labels_700hz


def _upsample_4hz_to_8hz(x: np.ndarray) -> np.ndarray:
    return resample_poly(x.astype(np.float64), up=2, down=1).astype(np.float64)


def _labels_700hz_to_8hz(labels_700hz: np.ndarray, target_len: int) -> np.ndarray:
    t = np.arange(target_len, dtype=np.float64) / 8.0
    raw_idx = np.clip((t * 700.0).astype(int), 0, len(labels_700hz) - 1)
    return labels_700hz[raw_idx].astype(np.int32)


def _lowpass_butterworth(x: np.ndarray, fs: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    # At fs=8 Hz, Nyquist=4 Hz. Requested 5 Hz is not realizable; clamp safely.
    nyq = 0.5 * fs
    effective_cutoff = min(cutoff_hz, 0.99 * nyq)
    b, a = butter(order, effective_cutoff / nyq, btype="low")
    return filtfilt(b, a, x).astype(np.float64)


def _initial_baseline_mask(labels_8hz: np.ndarray, baseline_label: int = 1) -> np.ndarray:
    run_end = 0
    while run_end < len(labels_8hz) and labels_8hz[run_end] == baseline_label:
        run_end += 1
    mask = np.zeros(len(labels_8hz), dtype=bool)
    if run_end > 0:
        mask[:run_end] = True
    else:
        mask = labels_8hz == baseline_label
    return mask


def _normalize_signal(
    x: np.ndarray,
    baseline_mask: np.ndarray,
    mode: Literal["ratio", "zscore"] = "ratio",
) -> np.ndarray:
    if not baseline_mask.any():
        baseline_mask = np.ones(len(x), dtype=bool)

    baseline = x[baseline_mask]
    mean_b = float(np.nanmean(baseline))
    std_b = float(np.nanstd(baseline))
    if not np.isfinite(std_b) or std_b < 1e-8:
        std_b = 1.0

    if mode == "zscore":
        out = (x - mean_b) / std_b
    else:
        denom = mean_b if abs(mean_b) > 1e-8 else 1.0
        out = x / denom

    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def _majority_label(window_labels: np.ndarray) -> int:
    valid = window_labels[np.isin(window_labels, [1, 2, 3, 4])]
    if len(valid) == 0:
        return 0
    vals, counts = np.unique(valid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _extract_window_features(
    tonic_win: np.ndarray,
    phasic_win: np.ndarray,
    fs: int,
) -> np.ndarray:
    # SCR peaks from phasic component
    prominence = max(1e-6, 0.1 * float(np.nanstd(phasic_win)))
    peaks, props = find_peaks(phasic_win, prominence=prominence)
    peak_amplitudes = props.get("prominences", np.array([], dtype=np.float64))

    # Rise times: time from previous local minimum to peak
    rise_times = []
    for p in peaks:
        if p <= 1:
            continue
        start = max(0, p - int(4 * fs))
        pre = phasic_win[start:p]
        if len(pre) == 0:
            continue
        trough_rel = int(np.argmin(pre))
        trough_idx = start + trough_rel
        rt = (p - trough_idx) / float(fs)
        if rt >= 0:
            rise_times.append(rt)

    max_peak_amp = float(np.max(peak_amplitudes)) if len(peak_amplitudes) > 0 else 0.0
    std_rise = float(np.std(rise_times)) if len(rise_times) > 1 else 0.0
    num_peaks = float(len(peaks))

    return np.array(
        [
            float(np.mean(tonic_win)),
            float(np.max(tonic_win)),
            max_peak_amp,
            std_rise,
            num_peaks,
        ],
        dtype=np.float32,
    )


def process_subject_eda(
    wesad_root: str,
    subject_id: int,
    norm_mode: Literal["ratio", "zscore"] = "ratio",
    lowpass_cutoff_hz: float = 5.0,
) -> dict[str, np.ndarray]:
    """Core strict-EDA processing for one WESAD subject.

    Steps
    -----
    1) Load wrist EDA (4 Hz) and 700 Hz labels from raw WESAD pickle.
    2) Upsample EDA to 8 Hz and remap labels to 8 Hz timeline.
    3) Baseline-normalize continuous EDA signal using initial baseline phase (label=1).
    4) Low-pass Butterworth filter normalized signal.
    5) Decompose into tonic/phasic streams with neurokit2.
    6) Extract IEEE 5 dominant features in 20 s windows with 1 s stride.
    7) Keep only windows fully within labels {1,2,3,4}; assign majority label per window.

    Returns
    -------
    dict with keys:
      - features: (num_windows, 5) float32
      - labels: (num_windows,) int32 in {1,2,3,4}
      - timestamps_sec: (num_windows,) float64
      - feature_names: (5,) object
      - eda_tonic: (T,) float64
      - eda_phasic: (T,) float64
      - labels_8hz: (T,) int32
    """
    eda_4hz, labels_700hz = _load_wesad_wrist_eda_and_labels(wesad_root, subject_id)

    eda_8hz = _upsample_4hz_to_8hz(eda_4hz)
    labels_8hz = _labels_700hz_to_8hz(labels_700hz, target_len=len(eda_8hz))

    baseline_mask = _initial_baseline_mask(labels_8hz, baseline_label=1)
    eda_norm = _normalize_signal(eda_8hz, baseline_mask=baseline_mask, mode=norm_mode)

    eda_filt = _lowpass_butterworth(eda_norm, fs=8.0, cutoff_hz=lowpass_cutoff_hz, order=4)

    try:
        sig_df, _ = nk.eda_process(eda_filt, sampling_rate=8)
        tonic = np.asarray(sig_df["EDA_Tonic"], dtype=np.float64)
        phasic = np.asarray(sig_df["EDA_Phasic"], dtype=np.float64)
    except Exception:
        # fallback decomposition if nk.eda_process fails
        clean = nk.eda_clean(eda_filt, sampling_rate=8)
        phasic_df = nk.eda_phasic(clean, sampling_rate=8)
        tonic = np.asarray(phasic_df["EDA_Tonic"], dtype=np.float64)
        phasic = np.asarray(phasic_df["EDA_Phasic"], dtype=np.float64)

    win = 20 * 8
    stride = 1 * 8

    feats, lbls, ts = [], [], []
    n = (len(tonic) - win) // stride + 1
    for i in range(max(0, n)):
        s = i * stride
        e = s + win
        w_labels = labels_8hz[s:e]

        # CardioMind strict override: keep only windows fully composed of {1,2,3,4}
        if not np.all(np.isin(w_labels, [1, 2, 3, 4])):
            continue

        f = _extract_window_features(tonic[s:e], phasic[s:e], fs=8)
        y = _majority_label(w_labels)
        if y == 0:
            continue

        feats.append(f)
        lbls.append(y)
        ts.append(s / 8.0)

    features = np.vstack(feats).astype(np.float32) if feats else np.zeros((0, 5), dtype=np.float32)
    labels = np.asarray(lbls, dtype=np.int32) if lbls else np.zeros((0,), dtype=np.int32)
    timestamps_sec = np.asarray(ts, dtype=np.float64) if ts else np.zeros((0,), dtype=np.float64)

    return {
        "features": features,
        "labels": labels,
        "timestamps_sec": timestamps_sec,
        "feature_names": np.array(EDA_FEATURE_NAMES, dtype=object),
        "eda_tonic": tonic.astype(np.float64),
        "eda_phasic": phasic.astype(np.float64),
        "labels_8hz": labels_8hz.astype(np.int32),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict EDA preprocessing for WESAD")
    parser.add_argument("--wesad-root", default="data/raw/WESAD")
    parser.add_argument("--subjects", nargs="*", type=int, default=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17])
    parser.add_argument("--out-root", default="data/processed_eda_strict_ratio")
    parser.add_argument("--norm-mode", choices=["ratio", "zscore"], default="ratio")
    args = parser.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    for sid in args.subjects:
        result = process_subject_eda(
            wesad_root=args.wesad_root,
            subject_id=sid,
            norm_mode=args.norm_mode,
            lowpass_cutoff_hz=5.0,
        )

        save_path = out_root / f"S{sid}.pt"
        torch.save(
            {
                "subject_id": sid,
                "cardiac_features": result["features"],
                "somatic_features": np.zeros((result["features"].shape[0], 0), dtype=np.float32),
                "cardiac_feature_names": result["feature_names"],
                "somatic_feature_names": [],
                "labels": result["labels"],
                "timestamps_sec": result["timestamps_sec"],
            },
            save_path,
        )
        print(
            f"S{sid}: windows={result['features'].shape[0]} "
            f"labels={{1:{int(np.sum(result['labels']==1))},2:{int(np.sum(result['labels']==2))},"
            f"3:{int(np.sum(result['labels']==3))},4:{int(np.sum(result['labels']==4))}}} "
            f"-> {save_path}"
        )


if __name__ == "__main__":
    main()
