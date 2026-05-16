#!/usr/bin/env python3
"""
Build per-subject ``S*.pt`` files from SWELL Poly5 waveforms (EDA + ECG channels).

Like ``preprocess_swell.py``, this reads the official minute feature CSV for
``PP``, ``C``, ``Condition``, and ``timestamp`` so labels align with the same
binary stress rule. Each ``pp{N}_*_c{K}.S00`` file is treated as one physiology
session (block ``C=K``); windows from all sessions of a participant are
concatenated in CSV time order into one ``S{N}.pt``.

Waveform layout (per window): two channels ``[EDA uV, ECG uV]`` using SWELL
MATLAB logical indices 7 (skin) and 8 (heart); see ``src/data/swell_poly5_channels.py``.

Outputs (per subject) mirror minute tensors where useful:

- ``cardiac_features``: ``float32`` ``(N, 2 * win_samples)`` — flattened
  ``[EDA samples | ECG samples]`` (see ``cardiac_feature_layout``).
  ``cardiac_feature_names`` is empty (too many columns for long windows); use
  ``win_samples`` and ``cardiac_feature_layout`` instead.
- ``waveform_windows``: ``float32`` ``(N, win_samples, 2)`` — same data, conv-friendly.
- ``labels``, ``condition_code``, ``block_c``, ``timestamps_sec`` (window-center
  time on a stitched timeline across sessions).

Example::

    python scripts/preprocess_swell_waveform.py \\
        --csv \"data/raw/SWELL/3 - Feature dataset/per sensor/D - Physiology features (HR_HRV_SCL - final).csv\" \\
        --s00-dir \"data/raw/SWELL/0 - Raw data/D - Physiology - raw data/Mobi signals (raw and filtered)\" \\
        --out-dir data/processed_swell_waveform \\
        --win-sec 10 --hop-sec 5

Omit windows whose assigned minute is rest (``R``); use a separate ``--out-dir`` to compare runs::

    python scripts/preprocess_swell_waveform.py \\
        --csv \"data/raw/SWELL/3 - Feature dataset/per sensor/D - Physiology features (HR_HRV_SCL - final).csv\" \\
        --s00-dir \"data/raw/SWELL/0 - Raw data/D - Physiology - raw data/Mobi signals (raw and filtered)\" \\
        --exclude-rest \\
        --out-dir data/processed_swell_waveform_no_rest \\
        --win-sec 10 --hop-sec 5
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.poly5_portilab import read_poly5
from src.data.swell_labels import condition_to_binary_label, condition_to_code
from src.data.swell_poly5_channels import pick_ecg_eda_uv
from src.data.swell_waveform_windows import (
    waveform_sliding_windows,
    window_center_row_indices,
    window_starts_seconds,
)

_S00_STEM = re.compile(r"^pp(\d+)_\d+-\d+-\d+_c(\d+)$", re.IGNORECASE)


def pp_to_subject_id(pp: str) -> int:
    """Map ``PP12`` -> 12 (same as ``preprocess_swell.py``)."""
    s = str(pp).strip()
    if not s.upper().startswith("PP"):
        raise ValueError(f"Expected PP-prefixed id, got {pp!r}")
    return int(s[2:])


def parse_swell_timestamp_to_epoch_sec(ts: str) -> float:
    """Parse SWELL timestamp strings like ``20120918T131600000`` (naive local)."""
    ts = str(ts).strip()
    if len(ts) < 15:
        return float("nan")
    head = ts[:15]
    tail = ts[15:18] if len(ts) >= 18 else "000"
    base = datetime.strptime(head, "%Y%m%dT%H%M%S")
    ms = int(tail.ljust(3, "0")[:3])
    return base.timestamp() + ms / 1000.0


def parse_swell_s00_stem(stem: str) -> tuple[int, int] | None:
    """
    Parse ``pp12_18-9-2014_c3`` -> ``(12, 3)``. Extension should be stripped.
    """
    m = _S00_STEM.match(str(stem).strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def discover_s00_sessions(s00_dir: Path) -> list[tuple[Path, int, int]]:
    """Return sorted list of ``(path, pp_int, c_block)`` for recognizable files."""
    out: list[tuple[Path, int, int]] = []
    if not s00_dir.is_dir():
        raise NotADirectoryError(s00_dir)
    for path in sorted(s00_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() != ".s00":
            continue
        parsed = parse_swell_s00_stem(path.stem)
        if parsed is None:
            continue
        pp_i, c_i = parsed
        out.append((path, pp_i, c_i))
    return out


def _rows_for_session(df: pd.DataFrame, pp_int: int, c_block: int) -> pd.DataFrame:
    mask = df["PP"].map(lambda s: pp_to_subject_id(str(s)) == pp_int) & (df["C"].astype(int) == int(c_block))
    g = df.loc[mask].copy()
    if g.empty:
        return g
    return g.sort_values("timestamp")


def _session_csv_arrays(g: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Minute-relative times (session starts at 0), labels, condition codes, first epoch."""
    ts_raw = g["timestamp"].astype(str)
    epoch = np.array([parse_swell_timestamp_to_epoch_sec(t) for t in ts_raw], dtype=np.float64)
    n = len(epoch)
    if n == 0:
        raise ValueError("empty session group")
    # Match preprocess_swell.py: tolerate bad/missing timestamps
    if np.isfinite(epoch).any():
        t0 = float(np.nanmin(epoch))
        ts_rel = np.where(
            np.isfinite(epoch),
            epoch - t0,
            np.arange(n, dtype=np.float64),
        ).astype(np.float64)
    else:
        t0 = float("nan")
        ts_rel = (np.arange(n, dtype=np.float64) * 60.0).astype(np.float64)
    cond = g["Condition"].astype(str).str.strip().str.upper()
    y = np.array([condition_to_binary_label(c) for c in cond], dtype=np.int64)
    code = np.array([condition_to_code(c) for c in cond], dtype=np.int32)
    return ts_rel, y, code, t0


def main() -> None:
    p = argparse.ArgumentParser(description="Preprocess SWELL .S00 waveforms to S*.pt (EDA+ECG windows)")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/SWELL/3 - Feature dataset/per sensor/D - Physiology features (HR_HRV_SCL - final).csv"),
        help="Same SWELL minute CSV as preprocess_swell.py (labels + PP/C/timestamp)",
    )
    p.add_argument(
        "--s00-dir",
        type=Path,
        required=True,
        help="Directory containing ``pp*_date_c*.S00`` Poly5 files",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/processed_swell_waveform"),
        help="Output directory for S{n}.pt and subject_id_map.json",
    )
    p.add_argument("--win-sec", type=float, default=10.0)
    p.add_argument("--hop-sec", type=float, default=5.0)
    p.add_argument("--max-blocks", type=int, default=None, help="Debug: cap Poly5 blocks read per file")
    p.add_argument(
        "--eda-logical-1based",
        type=int,
        default=7,
        help="MATLAB 1-based logical index for EDA (skin), default 7",
    )
    p.add_argument(
        "--ecg-logical-1based",
        type=int,
        default=8,
        help="MATLAB 1-based logical index for ECG (heart), default 8",
    )
    p.add_argument(
        "--exclude-rest",
        action="store_true",
        help="Drop windows whose assigned minute row is rest (R). Minute rows stay for time alignment.",
    )
    args = p.parse_args()

    if not args.csv.is_file():
        raise FileNotFoundError(args.csv)

    df = pd.read_csv(args.csv)
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", na=False)]
    required = {"PP", "C", "Condition", "timestamp"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    sessions = discover_s00_sessions(args.s00_dir)
    if not sessions:
        raise FileNotFoundError(f"No matching *.S00 files under {args.s00_dir} (expected names like pp1_18-9-2014_c3.S00)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.exclude_rest:
        print("preprocess_swell_waveform: --exclude-rest (dropping windows labeled from R minutes)")

    by_pp: dict[int, list[tuple[Path, int]]] = {}
    seen_session: set[tuple[int, int]] = set()
    for path, pp_i, c_i in sessions:
        key = (pp_i, c_i)
        if key in seen_session:
            print(f"  warn: duplicate session PP{pp_i} C={c_i}, skipping {path.name}")
            continue
        seen_session.add(key)
        by_pp.setdefault(pp_i, []).append((path, c_i))

    id_map: dict[str, str] = {}
    n_written = 0

    for pp_int in sorted(by_pp.keys()):
        pp_tag = f"PP{pp_int}"
        g_any = df[df["PP"].map(lambda s: pp_to_subject_id(str(s)) == pp_int)]
        if len(g_any):
            pp_tag = str(g_any["PP"].iloc[0]).strip()

        id_map[str(pp_int)] = pp_tag

        session_list = by_pp[pp_int]
        # Order sessions by first minute timestamp in CSV for that (PP, C)
        def sort_key(item: tuple[Path, int]) -> float:
            path, c_blk = item
            g0 = _rows_for_session(df, pp_int, c_blk)
            if g0.empty:
                return float("inf")
            ts_raw = str(g0.iloc[0]["timestamp"])
            return parse_swell_timestamp_to_epoch_sec(ts_raw)

        session_list = sorted(session_list, key=sort_key)

        win_list: list[np.ndarray] = []
        flat_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        code_list: list[np.ndarray] = []
        bc_list: list[np.ndarray] = []
        ts_center_list: list[np.ndarray] = []
        fs_ref: float | None = None
        win_samples_ref: int | None = None
        time_cursor = 0.0

        for path, c_blk in session_list:
            g = _rows_for_session(df, pp_int, c_blk)
            if g.empty:
                print(f"  skip (no CSV rows) {path.name} | PP{pp_int} C={c_blk}")
                continue

            ts_rel, y_min, code_min, _epoch0 = _session_csv_arrays(g)

            poly = read_poly5(path, max_blocks=args.max_blocks, drop_empty_channels=False)
            fs = float(poly.sample_rate_hz)
            if fs_ref is None:
                fs_ref = fs
            elif abs(fs_ref - fs) > 1e-3:
                print(f"  warn: fs mismatch {fs_ref} vs {fs} in {path.name}, using per-file fs for windows only")

            eda_uv, ecg_uv = pick_ecg_eda_uv(
                poly.data_uv,
                eda_logical_1based=args.eda_logical_1based,
                ecg_logical_1based=args.ecg_logical_1based,
            )
            x2 = np.stack([eda_uv, ecg_uv], axis=0)
            windows, start_sample = waveform_sliding_windows(x2, fs, args.win_sec, args.hop_sec)
            if windows.shape[0] == 0:
                print(f"  skip (too short for one window) {path.name} | PP{pp_int} C={c_blk}")
                continue

            if win_samples_ref is None:
                win_samples_ref = int(windows.shape[1])
            elif int(windows.shape[1]) != win_samples_ref:
                raise ValueError(
                    f"Inconsistent win_samples {windows.shape[1]} vs {win_samples_ref} in {path.name}; "
                    "use one --win-sec across all files or split by fs."
                )

            t_win = window_starts_seconds(start_sample, fs)
            row_ix = window_center_row_indices(ts_rel, t_win, args.win_sec)
            y_w = y_min[row_ix]
            c_w = code_min[row_ix]
            bc_w = np.full(len(y_w), int(c_blk), dtype=np.int32)
            centers_session = t_win.astype(np.float64) + float(args.win_sec) / 2.0
            centers_concat = centers_session + time_cursor

            if args.exclude_rest:
                cond_row = g["Condition"].astype(str).str.strip().str.upper().to_numpy()
                keep = cond_row[row_ix] != "R"
                if not np.any(keep):
                    print(f"  skip (all windows rest-labeled) {path.name} | PP{pp_int} C={c_blk}")
                    duration = float(x2.shape[1]) / fs
                    time_cursor += duration
                    continue
                windows = windows[keep]
                y_w = y_w[keep]
                c_w = c_w[keep]
                bc_w = bc_w[keep]
                centers_concat = centers_concat[keep]
                flat = windows.reshape(windows.shape[0], -1).astype(np.float32, copy=False)
            else:
                flat = windows.reshape(windows.shape[0], -1).astype(np.float32, copy=False)

            win_list.append(windows)
            flat_list.append(flat)
            y_list.append(y_w)
            code_list.append(c_w)
            bc_list.append(bc_w)
            ts_center_list.append(centers_concat.astype(np.float64))

            duration = float(x2.shape[1]) / fs
            time_cursor += duration

        if not win_list:
            print(f"  {pp_tag}: no windows produced, skip S{pp_int}.pt")
            continue

        W_all = np.concatenate(win_list, axis=0)
        X_flat = np.concatenate(flat_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        cond = np.concatenate(code_list, axis=0)
        block_c = np.concatenate(bc_list, axis=0)
        timestamps_sec = np.concatenate(ts_center_list, axis=0)

        W = int(W_all.shape[1])
        layout = (
            f"cardiac_features shape (N,{2 * W}): columns 0:{W}=EDA_uV, {W}:{2 * W}=ECG_uV; "
            f"waveform_windows (N,{W},2) channel0=EDA, channel1=ECG; win_sec={args.win_sec}, hop_sec={args.hop_sec}"
        )

        out_path = out_dir / f"S{pp_int}.pt"
        torch.save(
            {
                "kind": "swell_waveform",
                "subject_id": pp_int,
                "swell_pp": pp_tag,
                "exclude_rest": bool(args.exclude_rest),
                "sample_rate_hz": float(fs_ref or 0.0),
                "win_sec": float(args.win_sec),
                "hop_sec": float(args.hop_sec),
                "win_samples": W,
                "cardiac_features": X_flat,
                "cardiac_feature_names": [],
                "cardiac_feature_layout": layout,
                "eda_waveform_windows": W_all[:, :, 0].astype(np.float32, copy=False),
                "ecg_waveform_windows": W_all[:, :, 1].astype(np.float32, copy=False),
                "somatic_features": np.zeros((len(y), 0), dtype=np.float32),
                "somatic_feature_names": [],
                "labels": y,
                "condition_code": cond,
                "block_c": block_c,
                "timestamps_sec": timestamps_sec,
            },
            out_path,
        )
        n_written += 1
        n_pos = int((y == 1).sum())
        n_neg = int((y == 0).sum())
        print(f"  {pp_tag} -> {out_path.name} | windows={len(y)} binary1={n_pos} binary0={n_neg}")

    map_path = out_dir / "subject_id_map.json"
    if args.exclude_rest:
        legend = (
            "labels: 0=non-stress (N only; windows with R minute label dropped), 1=stress (T,I); "
            "condition_code R=0,N=1,T=2,I=3; timestamps_sec = window-center time on stitched session timeline"
        )
    else:
        legend = (
            "labels: 0=non-stress (N,R), 1=stress (T,I); condition_code R=0,N=1,T=2,I=3; "
            "timestamps_sec = window-center time (seconds) on stitched session timeline"
        )
    map_path.write_text(
        json.dumps(
            {
                "swell_pp_by_S_id": id_map,
                "label_legend": legend,
                "data_kind": "swell_waveform",
                "exclude_rest": bool(args.exclude_rest),
            },
            indent=2,
        )
    )
    print(f"\nWrote {n_written} subjects to {out_dir}")
    print(f"Wrote id map: {map_path}")


if __name__ == "__main__":
    main()
