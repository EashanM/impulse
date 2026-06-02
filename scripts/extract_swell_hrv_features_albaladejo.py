#!/usr/bin/env python3
"""Extract Albaladejo-style HRV features from SWELL Poly5 ECG (.S00).

Default windowing: 20 s length, 5 s stride (replicate paper uses 210 s / 30 s).

Output per subject: ``data/processed_swell_hrv_albaladejo_w20_s5/PP{N}.npz``
  - X (N_windows, 50), y, block_num, timestamp_sec, is_neutral, feature_names, ...

Labels from overview xlsx Sheet3: N -> 0, T/I -> 1; rest (R) omitted.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.poly5_portilab import read_poly5
from src.data.swell_hrv_albaladejo import (
    COND_TO_BINARY,
    COND_TO_CODE,
    FEATURE_NAMES,
    N_FEATURES,
    detect_r_peaks,
    sliding_hrv_windows,
)
from src.data.swell_poly5_channels import pick_ecg_eda_uv

S00_DIR = Path(
    "data/raw/SWELL/0 - Raw data/D - Physiology - raw data/"
    "Mobi signals (raw and filtered)"
)
OVERVIEW_XLSX = Path("data/raw/SWELL/SWELL-KW - overview available data.xlsx")
OUT_DIR = Path("data/processed_swell_hrv_albaladejo_w20_s5")


def iter_poly5_s00_files(s00_dir: Path) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pattern in ("*.S00", "*.s00"):
        for p in sorted(s00_dir.glob(pattern)):
            key = p.resolve()
            if key not in seen:
                seen.add(key)
                out.append(p)
    out.sort(key=lambda x: x.name.lower())
    return out


def build_condition_map(xlsx_path: Path) -> dict[tuple[int, int], str]:
    import openpyxl

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sh = wb["Sheet3"]
    cond_map: dict[tuple[int, int], str] = {}
    for row in sh.iter_rows(values_only=True):
        pp, block, cond = row[0], row[1], row[2]
        if pp is None or block is None or cond is None:
            continue
        try:
            cond_map[(int(pp), int(block))] = str(cond).strip().upper()
        except (TypeError, ValueError):
            continue
    return cond_map


def parse_filename(stem: str) -> tuple[int, int] | None:
    match = re.match(r"pp?(\d+)[_-].*[_-]c(\d+)", stem.lower())
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def process_file(
    path: Path,
    binary_label: int,
    condition_code: int,
    block_num: int,
    window_sec: float,
    stride_sec: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sig = read_poly5(path, drop_empty_channels=False)
    _, ecg = pick_ecg_eda_uv(sig.data_uv)
    ecg = np.asarray(ecg, dtype=np.float64)
    fs = int(round(sig.sample_rate_hz))

    peaks = detect_r_peaks(ecg, fs)
    return sliding_hrv_windows(
        ecg,
        fs,
        peaks,
        window_sec=window_sec,
        stride_sec=stride_sec,
        binary_label=binary_label,
        condition_code=condition_code,
        block_num=block_num,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SWELL Albaladejo-style HRV extraction")
    parser.add_argument("--s00-dir", type=Path, default=S00_DIR)
    parser.add_argument("--overview-xlsx", type=Path, default=OVERVIEW_XLSX)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--window-sec", type=float, default=20.0)
    parser.add_argument("--stride-sec", type=float, default=5.0)
    parser.add_argument("--max-subjects", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not args.s00_dir.is_dir():
        raise FileNotFoundError(f"S00 directory does not exist: {args.s00_dir}")
    if not args.overview_xlsx.is_file():
        raise FileNotFoundError(f"Overview workbook not found: {args.overview_xlsx}")

    cond_map = build_condition_map(args.overview_xlsx)
    files = iter_poly5_s00_files(args.s00_dir)
    if not files:
        raise FileNotFoundError(f"No *.S00 under {args.s00_dir}")

    pp_files: dict[int, list[tuple[Path, str, int]]] = {}
    for f in files:
        parsed = parse_filename(f.stem)
        if parsed is None:
            continue
        pp_num, block_num = parsed
        cond = cond_map.get((pp_num, block_num))
        if cond not in COND_TO_BINARY:
            continue
        pp_files.setdefault(pp_num, []).append((f, cond, block_num))

    pp_nums = sorted(pp_files.keys())
    if args.max_subjects is not None:
        pp_nums = pp_nums[: args.max_subjects]

    print(
        f"Processing {len(pp_nums)} participants | "
        f"window={args.window_sec}s stride={args.stride_sec}s | features={N_FEATURES}"
    )

    for pp_num in pp_nums:
        all_x: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_cond: list[np.ndarray] = []
        all_block: list[np.ndarray] = []
        all_time: list[np.ndarray] = []

        for path, cond, block_num in sorted(pp_files[pp_num], key=lambda x: x[2]):
            print(
                f"  PP{pp_num:02d} {path.name} -> cond={cond} block={block_num}",
                end="",
                flush=True,
            )
            x, y, cond_code, block_arr, t = process_file(
                path=path,
                binary_label=COND_TO_BINARY[cond],
                condition_code=COND_TO_CODE[cond],
                block_num=block_num,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
            )
            print(f" -> {len(y)} windows")
            if len(y) == 0:
                continue
            all_x.append(x)
            all_y.append(y)
            all_cond.append(cond_code)
            all_block.append(block_arr)
            all_time.append(t)

        if not all_x:
            print(f"  PP{pp_num:02d}: no valid windows, skipping")
            continue

        x_pp = np.concatenate(all_x, axis=0)
        y_pp = np.concatenate(all_y, axis=0)
        cond_pp = np.concatenate(all_cond, axis=0)
        block_pp = np.concatenate(all_block, axis=0)
        time_pp = np.concatenate(all_time, axis=0)
        is_neutral = y_pp == 0

        out_path = out_dir / f"PP{pp_num}.npz"
        np.savez_compressed(
            out_path,
            X=x_pp,
            y=y_pp,
            condition_code=cond_pp,
            block_num=block_pp,
            timestamp_sec=time_pp,
            is_neutral=is_neutral,
            subject_id=np.int64(pp_num),
            feature_names=np.asarray(FEATURE_NAMES, dtype="<U32"),
            window_sec=np.float64(args.window_sec),
            stride_sec=np.float64(args.stride_sec),
        )

        counts = np.bincount(y_pp, minlength=2)
        cond_counts = np.bincount(cond_pp, minlength=3)
        print(
            f"  PP{pp_num:02d}: saved {len(y_pp):4d} windows "
            f"[neutral={counts[0]} stress={counts[1]} | "
            f"N={cond_counts[0]} T={cond_counts[1]} I={cond_counts[2]}] -> {out_path}"
        )

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
