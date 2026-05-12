"""Sanity checks for CLAS helpers (no large data reads)."""

from pathlib import Path

import numpy as np

from src.data.clas_dataset import (
    block_type_to_binary_label,
    discover_participant_ids,
    find_segment_csv,
    load_gsr_ppg_matrix,
)


def test_block_type_mapping() -> None:
    assert block_type_to_binary_label("Baseline") == 0
    assert block_type_to_binary_label("Math Test") == 1
    assert block_type_to_binary_label("Unknown block") is None


def test_find_segment_csv_part1(tmp_path: Path) -> None:
    root = Path("data/CLAS_Database/CLAS")
    if not (root / "Participants" / "Part1" / "by_block").is_dir():
        return
    bb = root / "Participants" / "Part1" / "by_block"
    p = find_segment_csv(bb, 2, "ecg")
    assert p is not None and p.name.startswith("2_ecg")


def test_load_gsr_ppg_matrix_synthetic(tmp_path: Path) -> None:
    p = tmp_path / "t.csv"
    p.write_text(
        "Timestamp,PythonTimestamp,accelx,accely,accelz,ppg,gsr\n"
        "1,0.0,1,2,3,0.5,10.0\n"
        "2,0.1,x,2,3,0.5,10.0\n"  # bad row
        "3,0.2,1,2,3,0.6,11.0\n"
    )
    m = load_gsr_ppg_matrix(p)
    assert m.shape == (2, 5)
    assert np.allclose(m[0], [1, 2, 3, 0.5, 10.0])


def test_discover_skips_copy_folder() -> None:
    root = Path("data/CLAS_Database/CLAS")
    if not root.is_dir():
        return
    ids = discover_participant_ids(root)
    assert all(i > 0 for i in ids)
