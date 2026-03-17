"""
Preprocessing pipeline: raw WESAD signals -> windowed, normalized feature tensors.

Pipeline:
    1. Slide windows (30 s @ 1 s stride) through full subject recording
    2. Extract all currently implemented WESAD chest features per window:
             - Cardiac: ECG-derived HRV
             - Somatic: ACC/Temp/Resp
             - EDA
             - EMG
    3. Assign window-level labels: 0=baseline/non-stress, 1=stress
    4. Per-subject Z-score normalization using only true-baseline windows
    5. NaN imputation (forward-fill then zero)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.config import Config
from src.data.features import (
    CARDIAC_FEATURE_NAMES,
    CARDIOMIND_CARDIAC_FEATURE_NAMES,
    EDA_FEATURE_NAMES,
    EMG_FEATURE_NAMES,
    SOMATIC_FEATURE_NAMES,
    extract_cardiac_features,
    extract_cardiac_features_cardiomind,
    extract_eda_features,
    extract_emg_features,
    extract_somatic_features,
)
from src.data.wesad_loader import SubjectData, load_all_subjects


# Window-level label encoding: 0=baseline, 1=stress (no pre-stress)
LABEL_BASELINE = 0
LABEL_STRESS = 1

WESAD_DEFAULT_CARDIAC_FEATURE_NAMES = CARDIAC_FEATURE_NAMES
WESAD_CARDIOMIND_CARDIAC_FEATURE_NAMES = CARDIOMIND_CARDIAC_FEATURE_NAMES
WESAD_SOMATIC_FEATURE_NAMES = SOMATIC_FEATURE_NAMES + EDA_FEATURE_NAMES + EMG_FEATURE_NAMES


def _get_wesad_feature_profile(config: Config) -> str:
    """Return selected WESAD feature profile: default | cardiomind."""
    profile = getattr(config.preprocessing, "feature_profile", "default")
    if profile is None:
        return "default"
    profile = str(profile).strip().lower()
    if profile not in {"default", "cardiomind"}:
        raise ValueError(f"Unknown WESAD feature profile: {profile}")
    return profile


def _get_allowed_raw_labels(config: Config, profile: str) -> set[int]:
    """Return allowed raw WESAD labels for window inclusion."""
    if profile == "cardiomind":
        labels = getattr(config.preprocessing, "labels_of_interest", None)
        if labels is None or len(labels) == 0:
            return {1, 2, 3, 4}
        return {int(v) for v in labels}
    return {0, 1, 2, 3, 4, 5, 6, 7}


def _get_cardiomind_norm_mode(config: Config) -> str:
    """Return CardioMind normalization mode: ratio | difference."""
    method = getattr(config.normalization, "method", "cardiomind_ratio")
    method = str(method).strip().lower()
    alias = {
        "ratio": "ratio",
        "cardiomind_ratio": "ratio",
        "difference": "difference",
        "cardiomind_difference": "difference",
    }
    if method not in alias:
        raise ValueError(
            "For feature_profile='cardiomind', normalization.method must be one of "
            "['cardiomind_ratio', 'ratio', 'cardiomind_difference', 'difference']. "
            f"Got: {method}"
        )
    return alias[method]


def _extract_wesad_cardiac_features(ecg_window: np.ndarray, fs: int, profile: str) -> np.ndarray:
    if profile == "cardiomind":
        return extract_cardiac_features_cardiomind(ecg_window, fs=fs)
    return extract_cardiac_features(ecg_window, fs=fs)


def _get_wesad_cardiac_feature_names(profile: str) -> list[str]:
    if profile == "cardiomind":
        return WESAD_CARDIOMIND_CARDIAC_FEATURE_NAMES
    return WESAD_DEFAULT_CARDIAC_FEATURE_NAMES


def _get_wesad_somatic_feature_names(profile: str) -> list[str]:
    if profile == "cardiomind":
        return []
    return WESAD_SOMATIC_FEATURE_NAMES


@dataclass
class ProcessedSubject:
    subject_id: int | str  # int for WESAD (e.g. 2), str for Wearable (e.g. "S01")
    cardiac_features: np.ndarray   # (T, F_cardiac)
    somatic_features: np.ndarray   # (T, F_somatic)
    labels: np.ndarray             # (T,) values in {0, 1}
    timestamps_sec: np.ndarray     # (T,) seconds from recording start


def assign_window_label(window_labels: np.ndarray) -> int:
    """
    Determine the label for a single window.
    0 = baseline, 1 = stress. No pre-stress.
    """
    stress_frac = np.mean(window_labels == 2)  # raw WESAD stress = 2
    return LABEL_STRESS if stress_frac > 0.5 else LABEL_BASELINE


def _impute_nans(features: np.ndarray) -> np.ndarray:
    """Forward-fill NaNs along axis=0, then zero-fill any remaining."""
    result = features.copy()
    for col in range(result.shape[1]):
        mask = np.isnan(result[:, col])
        if not mask.any():
            continue
        # Forward fill
        valid_idx = np.where(~mask)[0]
        if len(valid_idx) > 0:
            for i in range(len(result)):
                if mask[i]:
                    prev_valid = valid_idx[valid_idx < i]
                    if len(prev_valid) > 0:
                        result[i, col] = result[prev_valid[-1], col]
        # Zero-fill anything still NaN (e.g., leading NaNs)
        result[np.isnan(result)] = 0.0
    return result


def process_subject(subject: SubjectData, config: Config) -> Optional[ProcessedSubject]:
    """
    Full preprocessing pipeline for one WESAD subject.

    Uses full recording and extracts all currently implemented chest features.
    Returns None if no usable stress segments are present.
    """
    labels_raw = subject.labels
    ecg = subject.chest_ecg
    acc = subject.chest_acc   # (N, 3) at 700 Hz
    temp = subject.chest_temp  # (N,) at 700 Hz
    resp = subject.chest_resp  # (N,) at 700 Hz
    eda = subject.chest_eda   # (N,) at 700 Hz
    emg = subject.chest_emg   # (N,) at 700 Hz
    feature_profile = _get_wesad_feature_profile(config)
    allowed_raw_labels = _get_allowed_raw_labels(config, feature_profile)

    if np.sum(labels_raw == 2) == 0:
        print(f"  S{subject.subject_id}: no stress segments after truncation, skipping")
        return None

    ecg_hz = config.data.ecg_hz
    window_samples = config.preprocessing.window_sec * ecg_hz
    stride_samples = config.preprocessing.stride_sec * ecg_hz

    n_windows = (len(ecg) - window_samples) // stride_samples + 1
    cardiac_feats_list = []
    somatic_feats_list = []
    label_list = []
    is_true_baseline = []
    timestamp_list = []

    for i in range(n_windows):
        start = i * stride_samples
        end = start + window_samples

        window_labels = labels_raw[start:end]

        if feature_profile == "cardiomind":
            # Paper-faithful filtering: only use windows fully composed of selected
            # WESAD protocol labels (typically {1,2,3,4}). This removes transitions
            # and undefined/other label codes.
            if not np.all(np.isin(window_labels, list(allowed_raw_labels))):
                continue

        wlabel = assign_window_label(window_labels)

        ecg_window = ecg[start:end]
        acc_window = acc[start:end]   # (window_samples, 3)
        temp_window = temp[start:end]
        resp_window = resp[start:end]
        eda_window = eda[start:end]
        emg_window = emg[start:end]

        c_feats = _extract_wesad_cardiac_features(ecg_window, fs=ecg_hz, profile=feature_profile)
        if feature_profile == "cardiomind":
            s_feats = np.zeros((0,), dtype=np.float64)
        else:
            s_feats_core = extract_somatic_features(
                acc_window, temp_window, resp_window, fs=ecg_hz
            )
            e_feats = extract_eda_features(eda_window)
            m_feats = extract_emg_features(emg_window)
            s_feats = np.concatenate([s_feats_core, e_feats, m_feats], axis=0)

        # CardioMind-style quality filtering: exclude noisy windows where HRV cannot be
        # reliably computed (e.g., missing/invalid R-peak-derived features).
        if feature_profile == "cardiomind":
            if np.any(~np.isfinite(c_feats)):
                continue

        cardiac_feats_list.append(c_feats)
        somatic_feats_list.append(s_feats)
        label_list.append(wlabel)
        is_true_baseline.append(np.mean(window_labels == 1) > 0.5)  # raw WESAD baseline=1
        timestamp_list.append(start / ecg_hz)

    if len(cardiac_feats_list) == 0:
        print(f"  S{subject.subject_id}: no valid windows, skipping")
        return None

    cardiac_features = np.stack(cardiac_feats_list)   # (T, 3)
    somatic_features = np.stack(somatic_feats_list)   # (T, 8) = 4 somatic + 2 EDA + 2 EMG
    labels = np.array(label_list, dtype=np.int32)
    true_baseline_mask = np.array(is_true_baseline, dtype=bool)
    timestamps = np.array(timestamp_list, dtype=np.float64)

    # Normalization (baseline-only statistics)
    # - default profile: per-subject z-score for all features
    # - cardiomind profile: ratio or difference normalization for cardiac features
    #   using first 90s neutral baseline when available; z-score for somatic features.
    if true_baseline_mask.any():
        if feature_profile == "cardiomind":
            cardiomind_norm_mode = _get_cardiomind_norm_mode(config)
            first_90s_baseline = true_baseline_mask & (timestamps <= 90.0)
            ref_mask = first_90s_baseline if first_90s_baseline.any() else true_baseline_mask

            cardiac_base = np.nanmean(cardiac_features[ref_mask], axis=0)
            if cardiomind_norm_mode == "ratio":
                cardiac_base = np.where(np.abs(cardiac_base) < 1e-8, 1.0, cardiac_base)
                cardiac_features = cardiac_features / cardiac_base
            else:
                cardiac_features = cardiac_features - cardiac_base

            # Keep somatic features stable with z-score when present
            if somatic_features.shape[1] > 0:
                s_mu = np.nanmean(somatic_features[true_baseline_mask], axis=0)
                s_sigma = np.nanstd(somatic_features[true_baseline_mask], axis=0)
                s_sigma[s_sigma < 1e-8] = 1.0
                somatic_features = (somatic_features - s_mu) / s_sigma
        else:
            for feats in [cardiac_features, somatic_features]:
                mu = np.nanmean(feats[true_baseline_mask], axis=0)
                sigma = np.nanstd(feats[true_baseline_mask], axis=0)
                sigma[sigma < 1e-8] = 1.0
                feats -= mu
                feats /= sigma

    cardiac_features = _impute_nans(cardiac_features)
    somatic_features = _impute_nans(somatic_features)

    label_counts = {
        "baseline": int((labels == 0).sum()),
        "stress": int((labels == 1).sum()),
    }
    print(
        f"  S{subject.subject_id}: {len(labels)} windows — "
        f"{label_counts['baseline']} baseline, "
        f"{label_counts['stress']} stress"
    )

    return ProcessedSubject(
        subject_id=subject.subject_id,
        cardiac_features=cardiac_features,
        somatic_features=somatic_features,
        labels=labels,
        timestamps_sec=timestamps,
    )


def process_and_save_all(config: Config) -> Dict[int, ProcessedSubject]:
    """
    Process all subjects and save to disk as .pt files.
    Returns dict of successfully processed subjects.
    """
    out_dir = Path(config.data.processed_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_profile = _get_wesad_feature_profile(config)
    cardiac_feature_names = _get_wesad_cardiac_feature_names(feature_profile)
    somatic_feature_names = _get_wesad_somatic_feature_names(feature_profile)

    print(f"Loading WESAD subjects from {config.data.wesad_root} ...")
    raw_subjects = load_all_subjects(config.data.wesad_root, config.data.subjects)

    processed = {}
    for sid, raw in tqdm(raw_subjects.items(), desc="Processing subjects"):
        result = process_subject(raw, config)
        if result is None:
            continue

        save_path = out_dir / f"S{sid}.pt"
        torch.save(
            {
                "subject_id": result.subject_id,
                "cardiac_features": result.cardiac_features,
                "somatic_features": result.somatic_features,
                "cardiac_feature_names": cardiac_feature_names,
                "somatic_feature_names": somatic_feature_names,
                "labels": result.labels,
                "timestamps_sec": result.timestamps_sec,
            },
            save_path,
        )
        processed[sid] = result

    print(f"\nSaved {len(processed)} subjects to {out_dir}")
    return processed


def load_processed_subject(processed_root: str, subject_id: int | str) -> ProcessedSubject:
    """Load a previously processed subject from a .pt file."""
    if isinstance(subject_id, str):
        path = Path(processed_root) / f"{subject_id}.pt"
    else:
        path = Path(processed_root) / f"S{subject_id}.pt"
    data = torch.load(path, weights_only=False)
    return ProcessedSubject(
        subject_id=data["subject_id"],
        cardiac_features=data["cardiac_features"],
        somatic_features=data["somatic_features"],
        labels=data["labels"],
        timestamps_sec=data["timestamps_sec"],
    )


def load_all_processed(
    processed_root: str, subject_ids: List[int] | List[str]
) -> Dict[int | str, ProcessedSubject]:
    """Load all processed subjects from disk. subject_ids: int for WESAD, str for Wearable."""
    subjects = {}
    for sid in subject_ids:
        try:
            subjects[sid] = load_processed_subject(processed_root, sid)
        except FileNotFoundError:
            print(f"  WARNING: processed file for {sid} not found, skipping")
    return subjects


def get_subject_ids_from_config(config) -> List[int] | List[str]:
    """Return subject IDs to use based on config.dataset."""
    if getattr(config.data, "dataset", "wesad") == "wearable":
        ids = getattr(config.data, "wearable_subjects", None)
        if ids is not None:
            return ids
        # Discover from processed folder
        root = Path(config.data.processed_root)
        if not root.exists():
            return []
        return sorted([f.stem for f in root.glob("*.pt")])
    return config.data.subjects
