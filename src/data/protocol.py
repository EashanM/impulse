"""
WESAD protocol inference and truncation for Version B.

Version A: Baseline -> Amusement -> Medi I -> Stress -> Rest -> Medi II
Version B: Baseline -> Stress -> Rest -> Medi I -> Amusement -> Medi II

For Version B, we truncate at the end of the first Rest period (first block after Stress).
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

# WESAD raw labels: 0=transient, 1=baseline, 2=stress, 3=amusement, 4=meditation, 5,6,7=self-report
FS = 700


def get_label_blocks(
    labels: np.ndarray,
    min_block_sec: float = 30,
) -> List[Tuple[int, int, int]]:
    """
    Extract contiguous blocks of labels.
    Returns list of (label, start_sample, end_sample).

    Merges blocks shorter than min_block_sec with neighbors; ignores transient (0).
    """
    changes = np.diff(labels, prepend=labels[0] - 1, append=labels[-1] - 1)
    run_starts = np.where(changes != 0)[0]
    run_ends = np.concatenate([run_starts[1:], [len(labels)]])

    blocks = []
    for start, end in zip(run_starts, run_ends):
        if start >= len(labels):
            continue
        lbl = int(labels[start])
        duration_sec = (end - start) / FS
        if lbl == 0 or duration_sec < 5:
            continue
        blocks.append((lbl, start, end))

    # Merge short blocks with neighbors
    merged = []
    i = 0
    while i < len(blocks):
        lbl, start, end = blocks[i]
        while i + 1 < len(blocks) and (blocks[i + 1][2] - blocks[i + 1][1]) / FS < min_block_sec:
            i += 1
            end = blocks[i][2]
        merged.append((lbl, start, end))
        i += 1

    return merged


def infer_protocol(blocks: List[Tuple[int, int, int]]) -> str:
    """
    Infer Version A or B from the sequence of label blocks.
    Version A: amusement (3) before stress (2)
    Version B: stress (2) before amusement (3)
    """
    stress_idx = next((i for i, (lbl, _, _) in enumerate(blocks) if lbl == 2), -1)
    amusement_idx = next((i for i, (lbl, _, _) in enumerate(blocks) if lbl == 3), -1)

    if stress_idx < 0:
        return "unknown"
    if amusement_idx < 0:
        return "B"  # No amusement, assume Version B (stress after baseline)

    if amusement_idx < stress_idx:
        return "A"
    return "B"


# Protocol phase order from paper (Schmidt et al.)
# Each tuple: (expected_label, phase_name)
VERSION_A_PHASES = [
    (1, "Baseline"),
    (3, "Amusement"),
    (4, "Medi I"),
    (2, "Stress"),
    (1, "Rest"),
    (4, "Medi II"),
]
VERSION_B_PHASES = [
    (1, "Baseline"),
    (2, "Stress"),
    (1, "Rest"),
    (4, "Medi I"),
    (3, "Amusement"),
    (4, "Medi II"),
]


def get_phase_name(protocol: str, block_idx: int) -> tuple[int, str] | None:
    """
    Return (expected_label, phase_name) for block_idx in the given protocol.
    Returns None if block_idx is beyond the known phases.
    """
    phases = VERSION_A_PHASES if protocol == "A" else VERSION_B_PHASES
    if block_idx < len(phases):
        return phases[block_idx]
    return None


def truncate_at_first_rest(
    labels: np.ndarray,
    ecg: np.ndarray,
    acc: np.ndarray,
    temp: np.ndarray,
    resp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    For Version B: truncate at end of first Rest block (first block after first Stress).
    Returns (labels, ecg, acc, temp, resp) truncated.
    """
    blocks = get_label_blocks(labels)
    if infer_protocol(blocks) != "B":
        return labels, ecg, acc, temp, resp

    # Find first stress block, then find the next block (Rest)
    stress_idx = next((i for i, (lbl, _, _) in enumerate(blocks) if lbl == 2), -1)
    if stress_idx < 0 or stress_idx + 1 >= len(blocks):
        return labels, ecg, acc, temp, resp

    # Rest is the block immediately after Stress
    _, _, truncate_end = blocks[stress_idx + 1]

    return (
        labels[:truncate_end],
        ecg[:truncate_end],
        acc[:truncate_end],
        temp[:truncate_end],
        resp[:truncate_end],
    )


def get_version_b_subjects(wesad_root: str, subject_ids: List[int]) -> List[int]:
    """
    Return list of subject IDs that follow Version B protocol.
    """
    from src.data.wesad_loader import load_subject

    version_b = []
    for sid in subject_ids:
        try:
            subject = load_subject(wesad_root, sid)
            blocks = get_label_blocks(subject.labels)
            if infer_protocol(blocks) == "B":
                version_b.append(sid)
        except FileNotFoundError:
            pass
    return version_b
