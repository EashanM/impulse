#!/usr/bin/env python3
"""Plot CLAS label summaries from clas_build_labels.py output.

The build script writes CSV; the path may end in .csv or .json (same format).

Figures:
  1) Binary class balance (y=0 vs y=1)
  2) Horizontal bar chart of block_type counts

Example:
  uv run python scripts/plot_clas_labels.py --input runs/clas_block_labels.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot CLAS label distributions")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("runs/clas_block_labels.json"),
        help="Path to block table (CSV written by clas_build_labels.py)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("plots"),
        help="Directory for PNG outputs",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise SystemExit(f"Input not found: {args.input.resolve()}")

    df = pd.read_csv(args.input)
    if "y" not in df.columns or "block_type" not in df.columns:
        raise SystemExit("Expected columns 'y' and 'block_type' in input table.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input.stem

    # --- Figure 1: binary balance ---
    counts = df["y"].value_counts().sort_index()
    label_map = {0: "Low load (y=0)", 1: "High load (y=1)"}
    labels = [label_map.get(int(i), str(i)) for i in counts.index]
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    colors = ["#4C72B0", "#DD8452"]
    bars = ax1.bar(labels, counts.values, color=colors[: len(counts)], edgecolor="black", linewidth=0.6)
    ax1.set_ylabel("Number of blocks")
    ax1.set_title("CLAS blocks: binary label balance (high vs low load)")
    total = int(counts.sum())
    for bar, c in zip(bars, counts.values, strict=True):
        pct = 100.0 * float(c) / max(total, 1)
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.02 * max(counts.max(), 1),
            f"{int(c)}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax1.set_ylim(0, float(counts.max()) * 1.18)
    fig1.tight_layout()
    out1 = args.out_dir / f"{stem}_binary_balance.png"
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)

    # --- Figure 2: block type counts (horizontal bars) ---
    type_counts = df["block_type"].value_counts().sort_values(ascending=True)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    y_pos = range(len(type_counts))
    ax2.barh(list(y_pos), type_counts.values, color="#55A868", edgecolor="black", linewidth=0.5)
    ax2.set_yticks(list(y_pos))
    ax2.set_yticklabels(type_counts.index.tolist())
    ax2.set_xlabel("Number of blocks")
    ax2.set_title("CLAS blocks: count by block type")
    ax2.invert_yaxis()
    fig2.tight_layout()
    out2 = args.out_dir / f"{stem}_block_types.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)

    print(f"Wrote {out1.resolve()}")
    print(f"Wrote {out2.resolve()}")


if __name__ == "__main__":
    main()
