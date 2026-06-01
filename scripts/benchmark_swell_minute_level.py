#!/usr/bin/env python3
"""
Minute-level logistic regression LOSO benchmark for SWELL processed tensors.

"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Column order in preprocess_swell.py `cardiac_features`: HR, RMSSD, SCL
FEATURE_SUBSET_INDICES: dict[str, list[int]] = {
    "all": [0, 1, 2],
    "hr_rmssd": [0, 1],
    "hr_scl": [0, 2],
    "rmssd_scl": [1, 2],
    "hr": [0],
    "rmssd": [1],
    "scl": [2],
}


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_subject_pt(path: Path, feature_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    d = torch.load(path, weights_only=False)
    X = np.asarray(d["cardiac_features"], dtype=np.float32)
    X = X[:, feature_indices]
    y = np.asarray(d["labels"], dtype=np.int64)
    valid = np.isfinite(X).all(axis=1)
    return X[valid], y[valid]


def _fit_scaler(train_subjects: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    Xcat = np.concatenate([x for x, _ in train_subjects], axis=0)
    mu = Xcat.mean(axis=0)
    sigma = Xcat.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma


def _apply_scaler(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def _normalize_subject_neutral_baseline(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Per-subject z-score from non-stress minutes before LOSO fold scaling.

    With ``--exclude-rest`` preprocessing, y==0 is neutral only. Uses all rows if
    no neutral minutes exist.
    """
    mask = y == 0
    ref = X[mask] if mask.any() else X
    mu = ref.mean(axis=0)
    sigma = ref.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return ((X - mu) / sigma).astype(np.float32)


def _parse_subject_id_list(spec: str) -> list[int]:
    """Parse comma-separated subject ids, e.g. ``7,8,11,23``."""
    if not str(spec).strip():
        return []
    out: list[int] = []
    for part in str(spec).replace(" ", "").split(","):
        if part:
            out.append(int(part))
    return sorted(set(out))


def _split_train_val_subjects(subjects: list[int], seed: int) -> tuple[list[int], list[int]]:
    if len(subjects) <= 2:
        return subjects, []
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(0.2 * len(shuffled))))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])
    return train_ids, val_ids


def _stack_subject_minutes(
    subj_data: dict[int, tuple[np.ndarray, np.ndarray]],
    subject_ids: list[int],
    mu: np.ndarray,
    sigma: np.ndarray,
    input_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for sid in subject_ids:
        X, y = subj_data[sid]
        X = _apply_scaler(X, mu, sigma)
        if len(y) > 0:
            xs.append(X)
            ys.append(y)
    if not ys:
        return np.zeros((0, input_dim), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """Binary metrics; explicit labels avoids sklearn warnings when one class is absent."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0, 0.0
    bin_kw = dict(labels=[0, 1], average="binary", pos_label=1, zero_division=0)
    return (
        float(accuracy_score(y_true, y_pred)),
        float(f1_score(y_true, y_pred, **bin_kw)),
        float(precision_score(y_true, y_pred, **bin_kw)),
        float(recall_score(y_true, y_pred, **bin_kw)),
    )


def _diagnostic_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    """Specificity, balanced accuracy, and percent of minutes predicted stress."""
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    spec = float(
        recall_score(
            y_true,
            y_pred,
            labels=[0, 1],
            average="binary",
            pos_label=0,
            zero_division=0,
        )
    )
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    psp = float(np.mean(y_pred == 1) * 100.0)
    return spec, bacc, psp


def _best_threshold_f1(y_true: np.ndarray, probs: np.ndarray, n_steps: int = 49) -> float:
    """Threshold in (0,1) that maximizes F1 on validation labels and positive-class probabilities."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, n_steps):
        pred = (probs >= t).astype(np.int64)
        f1v = float(
            f1_score(y_true, pred, labels=[0, 1], average="binary", pos_label=1, zero_division=0)
        )
        if f1v > best_f1 or (f1v == best_f1 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_f1, best_t = f1v, float(t)
    return best_t


def _majority_train_label(y_tr: np.ndarray) -> int:
    """Predict this constant on the test fold (train-only majority, ties -> 0)."""
    if len(y_tr) == 0:
        return 0
    y_tr = y_tr.astype(np.int64)
    c0 = int(np.sum(y_tr == 0))
    c1 = int(np.sum(y_tr == 1))
    if c1 > c0:
        return 1
    return 0


def _majority_baseline_metrics(y_te: np.ndarray, y_tr: np.ndarray) -> tuple[int, float, float, float, float]:
    """Always predict majority class from train minutes; score on test minutes."""
    maj = _majority_train_label(y_tr)
    if len(y_te) == 0:
        return maj, 0.0, 0.0, 0.0, 0.0
    pred = np.full(len(y_te), maj, dtype=np.int64)
    acc, f1, prec, rec = _metrics_from_arrays(y_te, pred)
    return maj, acc, f1, prec, rec


def _logistic_regression_kwargs(
    args: argparse.Namespace,
    *,
    class_weight: str | None,
) -> dict:
    """L2 (default) or L1 (lasso) binary logistic regression."""
    common = dict(
        C=args.sklearn_C,
        max_iter=args.sklearn_max_iter,
        class_weight=class_weight,
        random_state=args.seed,
    )
    if args.penalty == "l1":
        return {**common, "penalty": "l1", "solver": "saga"}
    return {**common, "penalty": "l2", "solver": "lbfgs"}


def _build_logistic_pipeline(args: argparse.Namespace, *, class_weight: str | None):
    estimator = LogisticRegression(**_logistic_regression_kwargs(args, class_weight=class_weight))
    if args.scaler == "standard":
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    return estimator


def _default_csv_path(args: argparse.Namespace) -> Path:
    tag = "minute_logistic"
    if args.scaler == "standard" and args.fit_on == "loso_train":
        tag = f"{tag}_standard_loso_train"
    elif args.scaler == "standard":
        tag = f"{tag}_standard"
    elif args.fit_on == "loso_train":
        tag = f"{tag}_loso_train"

    extra: list[str] = []
    if args.feature_subset != "all":
        extra.append(args.feature_subset)
    if args.class_weight == "balanced":
        extra.append("balanced")
    if args.feature_scale == "subject_neutral":
        extra.append("neutral_z")
    if args.penalty == "l1":
        extra.append("lasso")
    excl = _parse_subject_id_list(args.exclude_subjects or "")
    if excl:
        extra.append("excl" + "_".join(str(s) for s in excl))
    if extra:
        tag = f"{tag}_" + "_".join(extra)
    return Path("runs") / f"swell_{tag}_loso.csv"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SWELL minute-level LOSO logistic regression baseline")
    parser.add_argument("--data-root", default="data/processed_swell")
    parser.add_argument(
        "--feature-scale",
        choices=["raw", "subject_neutral"],
        default="raw",
        help="raw: use stored features only. subject_neutral: z-score each subject using non-stress minutes first.",
    )
    parser.add_argument(
        "--feature-subset",
        choices=["all", "hr_rmssd", "hr_scl", "rmssd_scl", "hr", "rmssd", "scl"],
        default="all",
        help="Which cardiac_features columns to use.",
    )
    parser.add_argument("--sklearn-C", type=float, default=1.0, help="LogisticRegression C.")
    parser.add_argument("--sklearn-max-iter", type=int, default=2000, help="LogisticRegression max_iter.")
    parser.add_argument(
        "--penalty",
        "--linear-sklearn-penalty",
        choices=["l2", "l1"],
        default="l2",
        help="LogisticRegression penalty. l1 uses solver=saga.",
    )
    parser.add_argument(
        "--scaler",
        "--linear-sklearn-scaler",
        choices=["none", "standard"],
        default="standard",
        help="Optional sklearn StandardScaler fit on train minutes only.",
    )
    parser.add_argument(
        "--fit-on",
        "--linear-sklearn-fit-on",
        choices=["inner_train", "loso_train"],
        default="loso_train",
        help="Which training minutes to fit logistic regression on.",
    )
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--exclude-subjects",
        default="",
        metavar="IDS",
        help="Comma-separated subject ids to omit from LOSO (e.g. 7,8,11,23).",
    )
    parser.add_argument("--out-csv", default=None)
    parser.add_argument(
        "--out-predictions-csv",
        default=None,
        help="Optional long CSV of per-minute y_true/y_pred/p_stress. Defaults to <out-csv stem>_predictions.csv.",
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--decision-threshold-tune",
        choices=["none", "f1_val"],
        default="f1_val",
        help="f1_val: choose probability threshold on inner val minutes to maximize F1.",
    )

    # Compatibility with the old benchmark_swell_baselines.py minute-level command.
    parser.add_argument("--architecture", choices=["linear"], default="linear", help=argparse.SUPPRESS)
    parser.add_argument("--linear-head", choices=["sklearn"], default="sklearn", help=argparse.SUPPRESS)
    parser.add_argument("--linear-sklearn-estimator", choices=["logistic"], default="logistic", help=argparse.SUPPRESS)
    parser.add_argument("--seq-len", type=int, default=1, help=argparse.SUPPRESS)

    args = parser.parse_args()
    if args.seq_len != 1:
        parser.error("This script is minute-level only; --seq-len must be 1.")
    if args.out_csv is None:
        args.out_csv = str(_default_csv_path(args))
    return args


def main() -> None:
    args = _parse_args()
    _seed_all(args.seed)

    data_root = Path(args.data_root)
    files = sorted(data_root.glob("S*.pt"), key=lambda p: int(p.stem.lstrip("S")))
    if not files:
        raise FileNotFoundError(f"No S*.pt files in {data_root}")

    subjects = [int(p.stem.lstrip("S")) for p in files]
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
        files = [data_root / f"S{sid}.pt" for sid in subjects]

    exclude_ids = _parse_subject_id_list(args.exclude_subjects)
    if exclude_ids:
        excl_set = set(exclude_ids)
        subjects = [sid for sid in subjects if sid not in excl_set]
        files = [data_root / f"S{sid}.pt" for sid in subjects]
        if len(subjects) < 2:
            raise ValueError(f"Need at least 2 subjects after --exclude-subjects {exclude_ids}; got {subjects}")
        print(f"exclude_subjects={exclude_ids} -> LOSO folds on {subjects}")

    feat_idx = FEATURE_SUBSET_INDICES[args.feature_subset]
    subj_data = {sid: _load_subject_pt(p, feat_idx) for sid, p in zip(subjects, files)}
    if args.feature_scale == "subject_neutral":
        subj_data = {
            sid: (_normalize_subject_neutral_baseline(X, y), y) for sid, (X, y) in subj_data.items()
        }
    input_dim = next(iter(subj_data.values()))[0].shape[1]

    print(
        "SWELL minute-level LOSO logistic regression"
        f" | subjects={subjects} | data_root={data_root} | feature_subset={args.feature_subset}"
        f" | feature_scale={args.feature_scale} | F={input_dim} | class_weight={args.class_weight}"
        f" | penalty={args.penalty} | scaler={args.scaler} | fit_on={args.fit_on}"
    )

    rows = []
    prediction_rows = []
    for held_out in subjects:
        train_ids_all = [sid for sid in subjects if sid != held_out]
        train_ids, val_ids = _split_train_val_subjects(train_ids_all, seed=args.seed + held_out)

        mu, sigma = _fit_scaler([subj_data[sid] for sid in train_ids_all])

        X_te, y_te = subj_data[held_out]
        X_te = _apply_scaler(X_te, mu, sigma)
        X_tr_inner, y_tr_inner = _stack_subject_minutes(subj_data, train_ids, mu, sigma, input_dim)
        X_va, y_va = _stack_subject_minutes(subj_data, val_ids, mu, sigma, input_dim)
        if args.fit_on == "loso_train":
            X_tr_lr, y_tr_lr = _stack_subject_minutes(subj_data, train_ids_all, mu, sigma, input_dim)
        else:
            X_tr_lr, y_tr_lr = X_tr_inner, y_tr_inner

        sk_class_weight = "balanced" if args.class_weight == "balanced" else None
        eval_threshold = float(args.decision_threshold)
        probs = np.full(len(y_te), np.nan, dtype=np.float64)
        pred_model = np.zeros(len(y_te), dtype=np.int64)

        if len(y_tr_lr) == 0 or len(y_te) == 0:
            acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0
        else:
            clf = _build_logistic_pipeline(args, class_weight=sk_class_weight)
            clf.fit(X_tr_lr, y_tr_lr)
            if (
                args.decision_threshold_tune == "f1_val"
                and len(y_va) >= 2
                and len(np.unique(y_va)) >= 2
                and X_va.shape[0] > 0
            ):
                p_va = clf.predict_proba(X_va)[:, 1].astype(np.float64)
                eval_threshold = _best_threshold_f1(y_va, p_va)
            probs = clf.predict_proba(X_te)[:, 1]
            pred_model = (probs >= eval_threshold).astype(np.int64)
            acc, f1, prec, rec = _metrics_from_arrays(y_te, pred_model)

        maj_label, maj_acc, maj_f1, maj_prec, maj_rec = _majority_baseline_metrics(y_te, y_tr_inner)
        train_stress_pct = float(np.mean(y_tr_inner == 1) * 100.0) if len(y_tr_inner) else 0.0
        spec, bacc, pstress = _diagnostic_metrics(y_te, pred_model) if len(pred_model) == len(y_te) else (0.0, 0.0, 0.0)
        tn = int(np.sum((y_te == 0) & (pred_model == 0)))
        fp = int(np.sum((y_te == 0) & (pred_model == 1)))
        fn = int(np.sum((y_te == 1) & (pred_model == 0)))
        tp = int(np.sum((y_te == 1) & (pred_model == 1)))

        row = {
            "subject": held_out,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(spec),
            "balanced_accuracy": float(bacc),
            "pred_stress_pct": float(pstress),
            "decision_threshold_used": float(eval_threshold),
            "n_samples": int(len(y_te)),
            "stress_pct": float(np.mean(y_te == 1) * 100.0) if len(y_te) else 0.0,
            "train_stress_pct": train_stress_pct,
            "maj_label": maj_label,
            "maj_accuracy": float(maj_acc),
            "maj_f1": float(maj_f1),
            "maj_precision": float(maj_prec),
            "maj_recall": float(maj_rec),
        }
        rows.append(row)
        for minute_index, y_true, y_pred, p_stress in zip(
            np.arange(len(y_te), dtype=np.int64),
            y_te,
            pred_model,
            probs,
            strict=True,
        ):
            prediction_rows.append(
                {
                    "subject": held_out,
                    "minute_index": int(minute_index),
                    "y_true": int(y_true),
                    "y_pred": int(y_pred),
                    "p_stress": float(p_stress),
                    "decision_threshold": float(eval_threshold),
                }
            )
        warn = ""
        if pstress >= 95.0 and rec >= 0.99:
            warn = " | WARN: near-all-stress predictions (recall~=1)"
        print(
            f"  S{held_out}: acc={row['accuracy']:.3f} f1={row['f1']:.3f} "
            f"prec={row['precision']:.3f} rec={row['recall']:.3f} "
            f"spec={spec:.3f} bacc={bacc:.3f} pred_stress%={pstress:.1f} thr={eval_threshold:.3f} "
            f"(n={row['n_samples']}, stress={row['stress_pct']:.1f}%) | "
            f"maj(baseline) f1={row['maj_f1']:.3f} acc={row['maj_accuracy']:.3f} maj_label={maj_label}{warn}"
        )

    acc = np.asarray([r["accuracy"] for r in rows], dtype=np.float64)
    f1 = np.asarray([r["f1"] for r in rows], dtype=np.float64)
    pr = np.asarray([r["precision"] for r in rows], dtype=np.float64)
    rc = np.asarray([r["recall"] for r in rows], dtype=np.float64)
    spec = np.asarray([r["specificity"] for r in rows], dtype=np.float64)
    bacc = np.asarray([r["balanced_accuracy"] for r in rows], dtype=np.float64)
    psp = np.asarray([r["pred_stress_pct"] for r in rows], dtype=np.float64)
    thr_u = np.asarray([r["decision_threshold_used"] for r in rows], dtype=np.float64)
    maj_f1 = np.asarray([r["maj_f1"] for r in rows], dtype=np.float64)
    maj_acc = np.asarray([r["maj_accuracy"] for r in rows], dtype=np.float64)
    maj_pr = np.asarray([r["maj_precision"] for r in rows], dtype=np.float64)
    maj_rc = np.asarray([r["maj_recall"] for r in rows], dtype=np.float64)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subject",
        "tn",
        "fp",
        "fn",
        "tp",
        "accuracy",
        "f1",
        "precision",
        "recall",
        "specificity",
        "balanced_accuracy",
        "pred_stress_pct",
        "decision_threshold_used",
        "n_samples",
        "stress_pct",
        "train_stress_pct",
        "maj_label",
        "maj_accuracy",
        "maj_f1",
        "maj_precision",
        "maj_recall",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(
            {
                "subject": "MEAN+/-STD",
                "tn": "",
                "fp": "",
                "fn": "",
                "tp": "",
                "accuracy": f"{acc.mean():.6f}+/-{acc.std():.6f}",
                "f1": f"{f1.mean():.6f}+/-{f1.std():.6f}",
                "precision": f"{pr.mean():.6f}+/-{pr.std():.6f}",
                "recall": f"{rc.mean():.6f}+/-{rc.std():.6f}",
                "specificity": f"{spec.mean():.6f}+/-{spec.std():.6f}",
                "balanced_accuracy": f"{bacc.mean():.6f}+/-{bacc.std():.6f}",
                "pred_stress_pct": f"{psp.mean():.6f}+/-{psp.std():.6f}",
                "decision_threshold_used": f"{thr_u.mean():.6f}+/-{thr_u.std():.6f}",
                "n_samples": int(np.mean([r["n_samples"] for r in rows])) if rows else 0,
                "stress_pct": f"{np.mean([r['stress_pct'] for r in rows]):.3f}" if rows else "",
                "train_stress_pct": f"{np.mean([r['train_stress_pct'] for r in rows]):.3f}" if rows else "",
                "maj_label": "",
                "maj_accuracy": f"{maj_acc.mean():.6f}+/-{maj_acc.std():.6f}",
                "maj_f1": f"{maj_f1.mean():.6f}+/-{maj_f1.std():.6f}",
                "maj_precision": f"{maj_pr.mean():.6f}+/-{maj_pr.std():.6f}",
                "maj_recall": f"{maj_rc.mean():.6f}+/-{maj_rc.std():.6f}",
            }
        )

    print(f"mean_accuracy: {acc.mean():.3f} +/- {acc.std():.3f}")
    print(f"mean_f1: {f1.mean():.3f} +/- {f1.std():.3f}")
    print(f"mean_balanced_accuracy: {bacc.mean():.3f} +/- {bacc.std():.3f}")
    print(f"mean_majority_f1: {maj_f1.mean():.3f} +/- {maj_f1.std():.3f} (train-label majority baseline)")
    print(f"saved_csv: {out_csv}")

    pred_csv = (
        Path(args.out_predictions_csv)
        if args.out_predictions_csv is not None
        else out_csv.with_name(f"{out_csv.stem}_predictions{out_csv.suffix}")
    )
    pred_csv.parent.mkdir(parents=True, exist_ok=True)
    with pred_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject", "minute_index", "y_true", "y_pred", "p_stress", "decision_threshold"],
        )
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(f"saved_predictions_csv: {pred_csv}")


if __name__ == "__main__":
    main()
