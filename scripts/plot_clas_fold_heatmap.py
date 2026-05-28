#!/usr/bin/env python3
"""Plot per-subject metric heatmap from CLAS LOSO fold details.

Input:
  fold_details.csv written by:
    - scripts/clas_benchmark_stress_table.py --write-fold-rows
    - scripts/clas_preprocess_and_benchmark.py --write-fold-rows

This produces a heatmap with:
  rows    = held_out subject id
  columns = "<data_stream> / <model>"
  cell    = chosen metric (default: macro_f1)

Example:
  uv run python scripts/plot_clas_fold_heatmap.py \
    --input runs/clas_stress_benchmark/fold_details.csv \
    --metric macro_f1 --out plots/clas_fold_macro_f1.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CLAS per-subject heatmap from fold_details.csv")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to fold_details.csv (written with --write-fold-rows)",
    )
    parser.add_argument(
        "--metric",
        default="macro_f1",
        choices=["macro_f1", "acc", "precision_macro", "recall_macro"],
        help="Which per-fold metric to plot in heatmap cells",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("plots/clas_fold_heatmap.png"),
        help="Output PNG path",
    )
    parser.add_argument("--title", default=None, help="Optional plot title override")
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=1.0)
    parser.add_argument("--annot", action="store_true", help="Annotate each cell with its value")
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input.resolve()}")

    df = pd.read_csv(args.input)
    required = {"held_out", "data_stream", "model", args.metric}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)} in {args.input}")

    df["arch"] = df["data_stream"].astype(str) + " / " + df["model"].astype(str)
    mat = df.pivot_table(index="held_out", columns="arch", values=args.metric, aggfunc="mean")
    mat = mat.sort_index()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    fig_w = max(10.0, 0.6 * max(1, mat.shape[1]))
    fig_h = max(6.0, 0.3 * max(1, mat.shape[0]))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    title = args.title or f"Per-subject {args.metric} across architectures"
    ax.set_title(title)

    im = ax.imshow(mat.values, aspect="auto", vmin=args.vmin, vmax=args.vmax)
    ax.set_yticks(np.arange(mat.shape[0]))
    ax.set_yticklabels(mat.index.tolist())
    ax.set_xticks(np.arange(mat.shape[1]))
    ax.set_xticklabels(mat.columns.tolist(), rotation=45, ha="right")
    ax.set_ylabel("Subject ID (held out)")
    ax.set_xlabel("Model architecture")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(args.metric)

    if args.annot:
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="black")

    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print(f"Wrote {args.out.resolve()}")


if __name__ == "__main__":
    main()

