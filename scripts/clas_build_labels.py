#!/usr/bin/env python3
"""Build per-block label table for CLAS and print class / imbalance statistics.

Writes a CSV listing every labeled block with paths, qualities, and binary y
under scheme ``high_vs_low`` (high load = cognitive tests, pictures, videos).

Example:
  python scripts/clas_build_labels.py --out-csv runs/clas_block_labels.csv
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from src.data.clas_dataset import (
    DEFAULT_CLAS_ROOT,
    discover_participant_ids,
    iter_labeled_blocks,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLAS block-level label table + stats")
    parser.add_argument("--clas-root", type=Path, default=DEFAULT_CLAS_ROOT)
    parser.add_argument("--out-csv", type=Path, default=Path("runs/clas_block_labels.csv"))
    parser.add_argument(
        "--scheme",
        default="high_vs_low",
        help="Label scheme (only high_vs_low supported for now)",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=None,
        help="If set, drop blocks where quality_modality score is below this (1–2 scale in CLAS)",
    )
    parser.add_argument(
        "--quality-modality",
        choices=["ecg", "eda", "ppg"],
        default="ecg",
        help="Which quality column to apply --min-quality to",
    )
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    pids = discover_participant_ids(args.clas_root)
    missing_details: list[int] = []
    for pid in pids:
        bd = args.clas_root / "Block_details" / f"Part{pid}_Block_Details.csv"
        if not bd.is_file():
            missing_details.append(pid)
            continue
        n_before = len(rows)
        for blk in iter_labeled_blocks(
            args.clas_root,
            pid,
            scheme=args.scheme,
            min_quality=args.min_quality,
            quality_modality=args.quality_modality,
        ):
            rows.append(
                {
                    "participant_id": blk.participant_id,
                    "block_id": blk.block_id,
                    "block_type": blk.block_type,
                    "length_sec": blk.length_sec,
                    "eda_quality": blk.eda_quality,
                    "ecg_quality": blk.ecg_quality,
                    "ppg_quality": blk.ppg_quality,
                    "y": blk.y,
                    "ecg_path": str(blk.ecg_path.relative_to(args.clas_root)),
                    "gsr_ppg_path": str(blk.gsr_ppg_path.relative_to(args.clas_root)),
                }
            )
        if len(rows) == n_before:
            # participant had block_details but no matching segments
            pass

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    y_counts = Counter(int(v) for v in df["y"]) if len(df) else Counter()
    n_total = sum(y_counts.values())
    type_counts = Counter(str(v) for v in df["block_type"]) if len(df) else Counter()

    stats = {
        "n_participants_with_block_details_csv": len(pids) - len(missing_details),
        "n_participants_missing_block_details": len(missing_details),
        "missing_block_details_participant_ids": sorted(missing_details),
        "n_labeled_blocks_written": int(len(df)),
        "binary_label_counts": {str(k): int(v) for k, v in sorted(y_counts.items())},
        "class_fractions": {
            str(k): float(v) / max(n_total, 1) for k, v in sorted(y_counts.items())
        },
        "imbalance_ratio_majority_to_minority": (
            float(max(y_counts.values())) / float(min(y_counts.values()))
            if len(y_counts) == 2 and min(y_counts.values()) > 0
            else None
        ),
        "block_type_counts": dict(type_counts.most_common()),
        "out_csv": str(args.out_csv.resolve()),
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
