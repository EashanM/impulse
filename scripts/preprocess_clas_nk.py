#!/usr/bin/env python3
"""Precompute NeuroKit2 PPG / EDA feature tensors for CLAS (cached .npz per participant).

Each file contains X (N, C, L), y (N,), and metadata. Run before benchmarks with
``--processed-root`` / ``--require-cache``.

Example:
  uv run python scripts/preprocess_clas_nk.py \\
      --out-root data/processed_clas_nk --window-sec 8 --stride-sec 8 --target-len 32
"""

from __future__ import annotations

import _repo_root  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.data.clas_dataset import DEFAULT_CLAS_ROOT, collect_windows_for_participant, discover_participant_ids
from src.data.clas_nk_features import EDA_FEATURE_NAMES, PPG_FEATURE_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute CLAS NeuroKit PPG/EDA feature caches")
    parser.add_argument("--clas-root", type=Path, default=DEFAULT_CLAS_ROOT)
    parser.add_argument("--out-root", type=Path, default=Path("data/processed_clas_nk"))
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--stride-sec", type=float, default=8.0)
    parser.add_argument("--target-len", type=int, default=32, help="Time axis L (>=32 for CNN)")
    parser.add_argument("--sub-win-sec", type=float, default=2.0)
    parser.add_argument("--sub-stride-sec", type=float, default=2.0)
    parser.add_argument("--min-quality", type=float, default=None)
    parser.add_argument("--quality-modality", choices=["ecg", "eda", "ppg"], default="ecg")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=["ppg_nk", "eda_nk"],
        choices=["ppg_nk", "eda_nk"],
    )
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    pids = discover_participant_ids(args.clas_root)
    if args.max_subjects is not None:
        pids = pids[: args.max_subjects]

    meta = {
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "target_len": args.target_len,
        "sub_win_sec": args.sub_win_sec,
        "sub_stride_sec": args.sub_stride_sec,
        "ppg_feature_names": list(PPG_FEATURE_NAMES),
        "eda_feature_names": list(EDA_FEATURE_NAMES),
    }
    (args.out_root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for modality in args.modalities:
        q_mod = args.quality_modality
        if modality == "ppg_nk":
            q_mod = "ppg"
        elif modality == "eda_nk":
            q_mod = "eda"

        for pid in tqdm(pids, desc=modality):
            X, y = collect_windows_for_participant(
                args.clas_root,
                pid,
                modality=modality,
                window_sec=args.window_sec,
                stride_sec=args.stride_sec,
                target_len=args.target_len,
                scheme="high_vs_low",
                min_quality=args.min_quality,
                quality_modality=q_mod,
                processed_root=None,
                nk_sub_win_sec=args.sub_win_sec,
                nk_sub_stride_sec=args.sub_stride_sec,
            )
            out = args.out_root / f"Part{pid}_{modality}.npz"
            np.savez_compressed(
                out,
                X=X,
                y=y,
                participant_id=pid,
                modality=modality,
                feature_names=np.array(
                    PPG_FEATURE_NAMES if modality == "ppg_nk" else EDA_FEATURE_NAMES,
                    dtype=object,
                ),
            )
            tqdm.write(f"  Part{pid} {modality}: windows={X.shape[0]} shape={X.shape} -> {out}")

    print(f"Done. Wrote caches under {args.out_root.resolve()}")
    print(f"Metadata: {args.out_root / 'meta.json'}")


if __name__ == "__main__":
    main()
