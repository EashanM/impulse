#!/usr/bin/env python3
"""
Benchmark: simple linear classifier on stress vs baseline.

Uses the same processed features and labels as the plotting code (from preprocessor).
All 11 features: 4 cardiac (mean_hr, pnn50, rmssd, hr_device) + 7 somatic
(acc_var, acc_slope, temp_slope, eda_mean, temp_mean, acc_mag_mean, bvp_mean).
Runs LOSO to assess whether features are predictive of stress state.

Usage:
    uv run python scripts/benchmark_linear.py
    uv run python scripts/benchmark_linear.py --config configs/wearable.yaml
"""

import argparse
import sys
from pathlib import Path

import _repo_root  # noqa: F401

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import load_config
from src.data.preprocessor import get_subject_ids_from_config, load_all_processed

FEATURE_NAMES = [
    "mean_hr", "pnn50", "rmssd", "hr_device",
    "acc_var", "acc_slope", "temp_slope", "eda_mean", "temp_mean", "acc_mag_mean", "bvp_mean",
]


def main():
    parser = argparse.ArgumentParser(description="Linear classifier benchmark for stress detection")
    parser.add_argument("--config", default="configs/wearable.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    subject_ids = get_subject_ids_from_config(config)
    if not subject_ids:
        print("No subjects found. Run preprocess first: uv run python scripts/preprocess.py --config configs/wearable.yaml")
        sys.exit(1)

    subjects = load_all_processed(config.data.processed_root, subject_ids)
    subject_ids = sorted(subjects.keys())

    if len(subject_ids) < 2:
        print("Need at least 2 subjects for LOSO")
        return

    n_cardiac = subjects[subject_ids[0]].cardiac_features.shape[1]
    n_somatic = subjects[subject_ids[0]].somatic_features.shape[1]
    n_total = n_cardiac + n_somatic

    print("=" * 60)
    print("Linear classifier benchmark (LOSO) — stress vs baseline")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Subjects: {subject_ids}")
    print(f"Features: cardiac ({n_cardiac}) + somatic ({n_somatic}) = {n_total} total")
    print(f"Labels: from preprocessor (same as plot_features_over_time)\n")

    results = []

    for held_out in subject_ids:
        train_ids = [s for s in subject_ids if s != held_out]

        X_train_list, y_train_list = [], []
        for sid in train_ids:
            subj = subjects[sid]
            X = np.concatenate([subj.cardiac_features, subj.somatic_features], axis=1)
            y = subj.labels
            X_train_list.append(X)
            y_train_list.append(y)

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # Drop any rows with NaN (from feature extraction failures)
        valid = ~np.any(np.isnan(X_train), axis=1)
        X_train = X_train[valid]
        y_train = y_train[valid]

        if len(np.unique(y_train)) < 2:
            print(f"  {held_out}: skip (only one class in train)")
            continue

        clf = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)

        # Test on held-out subject
        subj = subjects[held_out]
        X_test = np.concatenate([subj.cardiac_features, subj.somatic_features], axis=1)
        y_test = subj.labels

        valid_test = ~np.any(np.isnan(X_test), axis=1)
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]

        y_pred = clf.predict(X_test)

        acc = (y_pred == y_test).mean()
        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        n_stress = (y_test == 1).sum()
        n_baseline = (y_test == 0).sum()

        results.append({
            "held_out": held_out,
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "n_stress": n_stress,
            "n_baseline": n_baseline,
        })

        print(f"  {held_out}: acc={acc:.3f} F1={f1:.3f} prec={prec:.3f} rec={rec:.3f} "
              f"(stress={n_stress}, baseline={n_baseline})")

    if not results:
        print("No results.")
        return

    print("\n" + "=" * 60)
    print("LOSO summary (mean ± std)")
    print("=" * 60)
    for metric in ["accuracy", "f1", "precision", "recall"]:
        vals = [r[metric] for r in results]
        print(f"  {metric:12s}: {np.mean(vals):.3f} ± {np.std(vals):.3f}")

    # Per-feature coefficients (from last LOSO fold, for interpretability)
    if results:
        print("\nFeature coefficients (last fold, standardized features):")
        for i, name in enumerate(FEATURE_NAMES[:n_total]):
            coef = clf.coef_[0][i] if i < len(clf.coef_[0]) else 0
            print(f"  {name:15s}: {coef:+.3f}")
    print()


if __name__ == "__main__":
    main()
