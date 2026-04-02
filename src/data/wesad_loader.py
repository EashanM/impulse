"""
WESAD dataset loader.

Parses per-subject pickle files from the WESAD public release.
Each pickle is a dict with keys:
  - 'signal' -> {'chest': {...}, 'wrist': {...}}
  - 'label'  -> np.ndarray at 700 Hz
  - 'subject' -> str like 'S2'

Chest signals (all 700 Hz): ACC(3), ECG(1), EDA(1), EMG(1), Resp(1), Temp(1)
Wrist signals (Empatica E4): ACC(32 Hz, 3), BVP(64 Hz, 1), EDA(4 Hz, 1), TEMP(4 Hz, 1)

Label encoding: 0=transient, 1=baseline, 2=stress, 3=amusement, 4=meditation
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


@dataclass
class SubjectData:
    subject_id: int
    chest_ecg: np.ndarray   # (N,) float64, 700 Hz
    chest_acc: np.ndarray   # (N, 3) float64, 700 Hz
    chest_eda: np.ndarray   # (N,) float64, 700 Hz
    chest_emg: np.ndarray   # (N,) float64, 700 Hz
    chest_resp: np.ndarray  # (N,) float64, 700 Hz
    chest_temp: np.ndarray  # (N,) float64, 700 Hz
    wrist_eda: np.ndarray   # (M,) float64, 4 Hz
    labels: np.ndarray      # (N,) int, 700 Hz — aligned with chest signals


def load_subject(wesad_root: str, subject_id: int) -> SubjectData:
    """
    Load a single subject's pickle file.

    Parameters
    ----------
    wesad_root : str
        Path to the WESAD directory containing S2/, S3/, etc.
    subject_id : int
        Subject number (e.g. 2 for S2).

    Returns
    -------
    SubjectData with raw numpy arrays.
    """
    pkl_path = Path(wesad_root) / f"S{subject_id}" / f"S{subject_id}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(f"WESAD pickle not found: {pkl_path}")

    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")

    chest = data["signal"]["chest"]
    wrist = data["signal"]["wrist"]
    labels = data["label"]

    chest_ecg = np.asarray(chest["ECG"]).flatten().astype(np.float64)
    chest_acc = np.asarray(chest["ACC"]).astype(np.float64)  # (N, 3)
    chest_eda = np.asarray(chest["EDA"]).flatten().astype(np.float64)
    chest_emg = np.asarray(chest["EMG"]).flatten().astype(np.float64)
    chest_resp = np.asarray(chest["Resp"]).flatten().astype(np.float64)
    chest_temp = np.asarray(chest["Temp"]).flatten().astype(np.float64)
    wrist_eda = np.asarray(wrist["EDA"]).flatten().astype(np.float64)
    labels = np.asarray(labels).flatten().astype(np.int32)

    assert len(chest_ecg) == len(labels), (
        f"S{subject_id}: ECG length {len(chest_ecg)} != label length {len(labels)}"
    )

    return SubjectData(
        subject_id=subject_id,
        chest_ecg=chest_ecg,
        chest_acc=chest_acc,
        chest_eda=chest_eda,
        chest_emg=chest_emg,
        chest_resp=chest_resp,
        chest_temp=chest_temp,
        wrist_eda=wrist_eda,
        labels=labels,
    )


def load_all_subjects(
    wesad_root: str, subject_ids: List[int]
) -> Dict[int, SubjectData]:
    """
    Load multiple subjects into a dict keyed by subject_id.
    Skips subjects whose files are missing (with a warning).
    """
    subjects: Dict[int, SubjectData] = {}
    for sid in subject_ids:
        try:
            subjects[sid] = load_subject(wesad_root, sid)
            n_samples = len(subjects[sid].labels)
            duration_min = n_samples / 700 / 60
            stress_pct = np.mean(subjects[sid].labels == 2) * 100
            print(
                f"  S{sid}: {n_samples:,} samples "
                f"({duration_min:.1f} min), "
                f"{stress_pct:.1f}% stress"
            )
        except FileNotFoundError as e:
            print(f"  WARNING: {e}")
    return subjects


def summarize_subject(subject: SubjectData) -> Dict[str, any]:
    """Return a summary dict for quick inspection."""
    labels = subject.labels
    unique, counts = np.unique(labels, return_counts=True)
    label_dist = dict(zip(unique.tolist(), counts.tolist()))
    return {
        "subject_id": subject.subject_id,
        "total_samples": len(labels),
        "duration_sec": len(labels) / 700,
        "duration_min": len(labels) / 700 / 60,
        "label_distribution": label_dist,
        "ecg_range": (float(subject.chest_ecg.min()), float(subject.chest_ecg.max())),
        "chest_eda_range": (float(subject.chest_eda.min()), float(subject.chest_eda.max())),
        "wrist_eda_range": (float(subject.wrist_eda.min()), float(subject.wrist_eda.max())),
    }
