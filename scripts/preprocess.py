#!/usr/bin/env python3
"""
Preprocess dataset: extract features, window, normalize, save to disk.

Usage:
    python scripts/preprocess.py                           # WESAD
    python scripts/preprocess.py --config configs/default.yaml
    python scripts/preprocess.py --config configs/wearable.yaml  # Wearable
"""

import argparse
from pathlib import Path

import _repo_root  # noqa: F401

from src.config import load_config
from src.data.preprocessor import process_and_save_all


def main():
    parser = argparse.ArgumentParser(description="Preprocess stress detection data")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    parser.add_argument(
        "--cardiomind",
        action="store_true",
        help="Use CardioMind-style ECG HRV feature profile for WESAD (20s windows, 1s overlap stride by default)",
    )
    parser.add_argument(
        "--processed-root",
        default=None,
        help="Override output directory for processed files",
    )
    parser.add_argument(
        "--cardiomind-norm",
        choices=["ratio", "difference"],
        default="ratio",
        help="CardioMind cardiac normalization mode when --cardiomind is enabled",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.cardiomind and config.data.dataset != "wesad":
        raise ValueError("--cardiomind is only supported for WESAD config")

    if args.cardiomind:
        config.preprocessing.feature_profile = "cardiomind"
        config.preprocessing.window_sec = 20
        config.preprocessing.stride_sec = 1
        config.preprocessing.labels_of_interest = [1, 2, 3, 4]
        config.normalization.method = f"cardiomind_{args.cardiomind_norm}"
        if args.processed_root is None:
            config.data.processed_root = "data/processed_cardiomind"

    if args.processed_root is not None:
        config.data.processed_root = args.processed_root

    print(f"Config loaded: dataset={config.data.dataset}, encoder={config.model.encoder}, "
          f"window={config.preprocessing.window_sec}s, stride={config.preprocessing.stride_sec}s, "
          f"feature_profile={config.preprocessing.feature_profile}, "
                        f"labels_of_interest={config.preprocessing.labels_of_interest}, "
            f"normalization={config.normalization.method}, "
          f"processed_root={config.data.processed_root}")

    if getattr(config.data, "dataset", "wesad") == "wearable":
        print("Processing Wearable Device dataset (Empatica E4)\n")
        # processed = process_and_save_wearable(config)
    else:
        if args.cardiomind:
            print("Processing WESAD with CardioMind feature profile\n")
        else:
            print("Processing WESAD with default feature profile\n")
        processed = process_and_save_all(config)

    print(f"\nDone. Processed {len(processed)} subjects.")


if __name__ == "__main__":
    main()
