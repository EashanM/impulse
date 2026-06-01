#!/usr/bin/env python3
"""
LOSO benchmarks on SWELL processed tensors (`data/processed_swell/S*.pt`).

- linear + --linear-head pytorch: one `nn.Linear` on `flatten(B, T, F)` trained with Adam.
- linear + --linear-head sklearn: tabular classifier on flattened windows (`--linear-sklearn-estimator logistic|rf|hgb|xgb`). Use `--linear-sklearn-penalty l1` for lasso (logistic only). Use `--linear-sklearn-scaler standard` for `Pipeline(StandardScaler, clf)` (scaler fit on train windows only). Use `--linear-sklearn-fit-on loso_train` to fit on all non-held-out subjects' windows. Default CSV tags include the estimator (e.g. `rf_sklearn_standard_loso_train_seq1_balanced_loso.csv`).
- cnn_gru: `Conv1d` along time then `GRU` + head.
  - `--cnn-gru-classifier pytorch` (default): `nn.Linear` head, trained end-to-end.
  - `--cnn-gru-classifier sklearn_lr`: same CNN+GRU training (PyTorch head for backprop), then **freeze** encoder and fit **sklearn `LogisticRegression`** on last GRU hidden states; reported metrics are for that LR head on the test fold.
- gru: GRU only on `(B, T, F)` (no conv), same classifier options via `--gru-classifier`.

Default output CSV name includes `_seq{N}` when `--seq-len` is not 10, and `_balanced` when `--class-weight balanced`, so runs land in distinct files without always passing `--out-csv`.

- `--feature-subset`: `all` (default, HR+RMSSD+SCL), pairwise `hr_rmssd` / `hr_scl` / `rmssd_scl`, or unimodal `hr` / `rmssd` / `scl` (same architectures; default CSV name includes the subset when not `all`).

Each fold also records a train-majority baseline (always predict the majority train-window label) in CSV columns `maj_*`.

Use `--save-confusion-for SUBJECT` to write `runs/swell_confusion_<arch>_S<SUBJECT>.png` for that held-out fold (model predictions vs true labels).

Same protocol as `benchmark_gru_cardiomind.py`: per-fold scaler on train subjects,
sliding windows, optional early stopping on a val subject split (PyTorch heads only).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.swell_baselines import SwellCnnGruClassifier, SwellFlattenLinearClassifier, SwellGruClassifier

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


def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) < seq_len:
        return np.zeros((0, seq_len, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    xs, ys = [], []
    for t in range(seq_len - 1, len(X)):
        xs.append(X[t - seq_len + 1 : t + 1])
        ys.append(y[t])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def _fit_scaler(train_subjects: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    Xcat = np.concatenate([x for x, _ in train_subjects], axis=0)
    mu = Xcat.mean(axis=0)
    sigma = Xcat.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma


def _apply_scaler(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def _sklearn_logistic_regression_kwargs(
    args: argparse.Namespace,
    *,
    class_weight: str | None,
) -> dict:
    """L2 (default) or L1 (lasso) binary logistic regression for sklearn heads."""
    common = dict(
        C=args.sklearn_C,
        max_iter=args.sklearn_max_iter,
        class_weight=class_weight,
        random_state=args.seed,
    )
    penalty = getattr(args, "linear_sklearn_penalty", "l2")
    if penalty == "l1":
        return {**common, "penalty": "l1", "solver": "saga"}
    return {**common, "penalty": "l2", "solver": "lbfgs"}


def _sklearn_tree_max_depth(args: argparse.Namespace) -> int | None:
    return None if args.sklearn_max_depth <= 0 else int(args.sklearn_max_depth)


def _build_sklearn_estimator(
    args: argparse.Namespace,
    *,
    class_weight: str | None,
    y_train: np.ndarray | None = None,
):
    """Binary sklearn classifier for minute/tabular features (linear sklearn head)."""
    est = args.linear_sklearn_estimator
    seed = args.seed
    max_depth = _sklearn_tree_max_depth(args)

    if est == "logistic":
        return LogisticRegression(**_sklearn_logistic_regression_kwargs(args, class_weight=class_weight))

    if est == "rf":
        return RandomForestClassifier(
            n_estimators=args.sklearn_n_estimators,
            max_depth=max_depth,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=-1,
        )

    if est == "hgb":
        kw: dict = dict(
            max_iter=args.sklearn_max_iter,
            random_state=seed,
        )
        if max_depth is not None:
            kw["max_depth"] = max_depth
        if class_weight is not None:
            kw["class_weight"] = class_weight
        return HistGradientBoostingClassifier(**kw)

    if est == "xgb":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError(
                "xgboost is required for --linear-sklearn-estimator xgb. "
                "Install with: pip install xgboost"
            ) from e
        scale_pos_weight = 1.0
        if class_weight == "balanced" and y_train is not None and len(y_train) > 0:
            n_pos = float(np.sum(y_train == 1))
            n_neg = float(np.sum(y_train == 0))
            scale_pos_weight = n_neg / max(n_pos, 1.0)
        xgb_depth = max_depth if max_depth is not None else 6
        return xgb.XGBClassifier(
            n_estimators=args.sklearn_n_estimators,
            max_depth=xgb_depth,
            learning_rate=args.sklearn_learning_rate,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            eval_metric="logloss",
            n_jobs=-1,
        )

    raise ValueError(f"Unknown linear-sklearn-estimator: {est}")


def _build_sklearn_pipeline(
    args: argparse.Namespace,
    *,
    class_weight: str | None,
    y_train: np.ndarray | None = None,
):
    """Optional StandardScaler + estimator (trees still work with scaling)."""
    estimator = _build_sklearn_estimator(args, class_weight=class_weight, y_train=y_train)
    if args.linear_sklearn_scaler == "standard":
        return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])
    return estimator


def _normalize_subject_neutral_baseline(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Per-subject z-score from non-stress minutes (y==0) before LOSO fold scaling.

    With ``--exclude-rest`` preprocessing, y==0 is neutral (N) only. Uses all rows
    if no neutral minutes exist. Not label leakage: baseline stats use only class 0.
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


def _metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float, float]:
    """Binary metrics; explicit ``labels`` avoids sklearn warnings when one class is absent in ``y_true``."""
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
    """specificity (TNR / recall of class 0), balanced accuracy, % windows predicted stress."""
    if len(y_true) == 0:
        return 0.0, 0.0, 0.0
    spec = float(
        recall_score(
            y_true, y_pred, labels=[0, 1], average="binary", pos_label=0, zero_division=0
        )
    )
    bacc = float(balanced_accuracy_score(y_true, y_pred))
    psp = float(np.mean(y_pred == 1) * 100.0)
    return spec, bacc, psp


def _best_threshold_f1(y_true: np.ndarray, probs: np.ndarray, n_steps: int = 49) -> float:
    """Threshold in (0,1) that maximizes F1 on validation labels and probs[:, positive]."""
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


def _collect_probs_torch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_list: list[int] = []
    p_list: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            pr = torch.softmax(logits, dim=1)[:, 1]
            y_list.extend(yb.cpu().numpy().tolist())
            p_list.extend(pr.cpu().numpy().astype(np.float64).tolist())
    return np.asarray(y_list, dtype=np.int64), np.asarray(p_list, dtype=np.float64)


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
    """Always predict majority class from train windows; score on test windows."""
    maj = _majority_train_label(y_tr)
    if len(y_te) == 0:
        return maj, 0.0, 0.0, 0.0, 0.0
    pred = np.full(len(y_te), maj, dtype=np.int64)
    acc, f1, prec, rec = _metrics_from_arrays(y_te, pred)
    return maj, acc, f1, prec, rec


def _collect_predictions_torch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    decision_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = (probs >= decision_threshold).long()
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
    return np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64)


def _collect_encoder_features(
    model: SwellCnnGruClassifier | SwellGruClassifier,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack (N, H) encoder outputs and labels for sklearn head fitting."""
    model.eval()
    h_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            h = model.encode(xb)
            h_list.append(h.cpu().numpy().astype(np.float64))
            y_list.append(yb.numpy().astype(np.int64))
    if not h_list:
        return np.zeros((0, model.encoding_dim), dtype=np.float64), np.zeros((0,), dtype=np.int64)
    return np.concatenate(h_list, axis=0), np.concatenate(y_list, axis=0)


def _arch_slug(args: argparse.Namespace) -> str:
    if args.architecture == "linear" and args.linear_head == "sklearn":
        slug = _sklearn_estimator_csv_tag(args)
        if args.linear_sklearn_scaler == "standard" and args.linear_sklearn_fit_on == "loso_train":
            slug = f"{slug}_standard_loso_train"
        elif args.linear_sklearn_scaler == "standard":
            slug = f"{slug}_standard"
        elif args.linear_sklearn_fit_on == "loso_train":
            slug = f"{slug}_loso_train"
    elif args.architecture == "linear":
        slug = "linear_pytorch"
    elif args.architecture == "cnn_gru" and args.cnn_gru_classifier == "sklearn_lr":
        slug = "cnn_gru_frozen_logreg"
    elif args.architecture == "cnn_gru":
        slug = "cnn_gru"
    elif args.architecture == "gru" and args.gru_classifier == "sklearn_lr":
        slug = "gru_frozen_logreg"
    elif args.architecture == "gru":
        slug = "gru"
    else:
        raise AssertionError(f"Unhandled architecture slug: {args.architecture}")
    if args.feature_subset != "all":
        slug = f"{slug}_{args.feature_subset}"
    return slug


def _save_confusion_png(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["pred 0", "pred 1"],
        yticklabels=["true 0", "true 1"],
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center", color="w" if cm[i, j] > thresh else "k")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    decision_threshold: float,
) -> tuple[float, float, float, float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    y_true, y_pred = [], []
    losses = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            losses.append(float(loss.item()))
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = (probs >= decision_threshold).long()
            y_true.extend(yb.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    acc, f1, prec, rec = _metrics_from_arrays(y_true_np, y_pred_np)
    return (
        float(np.mean(losses)) if losses else 0.0,
        acc,
        f1,
        prec,
        rec,
    )


def _parse_conv_channels(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("--conv-channels must be a comma-separated list, e.g. 32,32")
    return [int(p) for p in parts]


def _build_model(
    architecture: str,
    seq_len: int,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    conv_channels: list[int],
    conv_kernel: int,
) -> nn.Module:
    if architecture == "linear":
        return SwellFlattenLinearClassifier(seq_len=seq_len, input_dim=input_dim, num_classes=2)
    if architecture == "gru":
        return SwellGruClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=2,
        )
    if architecture == "cnn_gru":
        return SwellCnnGruClassifier(
            input_dim=input_dim,
            conv_channels=conv_channels,
            kernel_size=conv_kernel,
            gru_hidden=hidden_dim,
            gru_layers=num_layers,
            dropout=dropout,
            num_classes=2,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def _sklearn_estimator_csv_tag(args: argparse.Namespace) -> str:
    """CSV filename stem for --linear-head sklearn (logistic keeps legacy ``linear_sklearn`` prefix)."""
    est = args.linear_sklearn_estimator
    if est == "logistic":
        return "linear_sklearn"
    return f"{est}_sklearn"


def _default_swell_csv_path(args: argparse.Namespace) -> Path:
    """Distinct default filenames: base tag + `_seqN` if seq_len != 10 + `_balanced` if class-weight balanced."""
    if args.architecture == "linear" and args.linear_head == "sklearn":
        tag = _sklearn_estimator_csv_tag(args)
        if args.linear_sklearn_scaler == "standard" and args.linear_sklearn_fit_on == "loso_train":
            tag = f"{tag}_standard_loso_train"
        elif args.linear_sklearn_scaler == "standard":
            tag = f"{tag}_standard"
        elif args.linear_sklearn_fit_on == "loso_train":
            tag = f"{tag}_loso_train"
    elif args.architecture == "linear":
        tag = "linear_pytorch"
    elif args.architecture == "gru":
        tag = "gru_frozen_logreg" if args.gru_classifier == "sklearn_lr" else "gru"
    elif args.architecture == "cnn_gru" and args.cnn_gru_classifier == "sklearn_lr":
        tag = "cnn_gru_frozen_logreg"
    else:
        tag = "cnn_gru"
    extra: list[str] = []
    if args.feature_subset != "all":
        extra.append(args.feature_subset)
    if args.seq_len != 10:
        extra.append(f"seq{args.seq_len}")
    if args.class_weight == "balanced":
        extra.append("balanced")
    if getattr(args, "feature_scale", "raw") == "subject_neutral":
        extra.append("neutral_z")
    if getattr(args, "linear_sklearn_penalty", "l2") == "l1":
        extra.append("lasso")
    excl = _parse_subject_id_list(getattr(args, "exclude_subjects", "") or "")
    if excl:
        extra.append("excl" + "_".join(str(s) for s in excl))
    if extra:
        tag = f"{tag}_" + "_".join(extra)
    return Path("runs") / f"swell_{tag}_loso.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description="SWELL LOSO: linear, GRU-only, or CNN+GRU baseline")
    parser.add_argument("--architecture", choices=["linear", "gru", "cnn_gru"], required=True)
    parser.add_argument(
        "--linear-head",
        choices=["pytorch", "sklearn"],
        default="pytorch",
        help="For --architecture linear only: PyTorch Linear+CE or sklearn LogisticRegression on flattened windows.",
    )
    parser.add_argument(
        "--sklearn-C",
        type=float,
        default=1.0,
        help="LogisticRegression C for sklearn heads (linear sklearn, cnn_gru sklearn_lr, gru sklearn_lr).",
    )
    parser.add_argument(
        "--sklearn-max-iter",
        type=int,
        default=2000,
        help="Max iterations for LogisticRegression lbfgs (linear sklearn or neural sklearn_lr heads).",
    )
    parser.add_argument(
        "--linear-sklearn-estimator",
        choices=["logistic", "rf", "hgb", "xgb"],
        default="logistic",
        help="For --linear-head sklearn: logistic (default), random forest (rf), "
        "sklearn HistGradientBoosting (hgb), or XGBoost (xgb).",
    )
    parser.add_argument(
        "--linear-sklearn-penalty",
        choices=["l2", "l1"],
        default="l2",
        help="For --linear-sklearn-estimator logistic only: l2 or l1 (lasso, solver=saga).",
    )
    parser.add_argument(
        "--sklearn-n-estimators",
        type=int,
        default=200,
        help="n_estimators for rf / xgb (ignored for logistic / hgb).",
    )
    parser.add_argument(
        "--sklearn-max-depth",
        type=int,
        default=0,
        help="Max tree depth for rf / hgb / xgb; 0 = unlimited (rf) or default depth (hgb/xgb).",
    )
    parser.add_argument(
        "--sklearn-learning-rate",
        type=float,
        default=0.1,
        help="Learning rate for xgb only.",
    )
    parser.add_argument(
        "--linear-sklearn-scaler",
        choices=["none", "standard"],
        default="none",
        help="For --linear-head sklearn: 'standard' = WESAD-style Pipeline(StandardScaler, LogisticRegression); 'none' = LR only on fold-z-scored flattened windows.",
    )
    parser.add_argument(
        "--linear-sklearn-fit-on",
        choices=["inner_train", "loso_train"],
        default="inner_train",
        help="For --linear-head sklearn: which windows to fit LR on. loso_train = all subjects except held-out (like benchmark_linear_cardiomind.py); inner_train = same subset as PyTorch inner train split.",
    )
    parser.add_argument("--data-root", default="data/processed_swell")
    parser.add_argument(
        "--feature-scale",
        choices=["raw", "subject_neutral"],
        default="raw",
        help="raw: use stored features only. subject_neutral: z-score each subject using "
        "non-stress (y=0) minutes, then LOSO train-fold scaler as usual.",
    )
    parser.add_argument(
        "--feature-subset",
        choices=["all", "hr_rmssd", "hr_scl", "rmssd_scl", "hr", "rmssd", "scl"],
        default="all",
        help="Which cardiac_features columns: all, pairwise combos, or unimodal hr / rmssd / scl (column order HR, RMSSD, SCL).",
    )
    parser.add_argument("--seq-len", type=int, default=10, help="Minutes per window (use 1 for instantaneous features only)")
    parser.add_argument("--hidden-dim", type=int, default=32, help="GRU hidden size (gru and cnn_gru)")
    parser.add_argument("--num-layers", type=int, default=1, help="GRU layers (gru and cnn_gru)")
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--conv-channels", default="32,32", help="cnn_gru: comma-separated Conv1d widths")
    parser.add_argument("--conv-kernel", type=int, default=3, help="cnn_gru: odd kernel along time")
    parser.add_argument(
        "--cnn-gru-classifier",
        choices=["pytorch", "sklearn_lr"],
        default="pytorch",
        help="cnn_gru only: end-to-end nn.Linear head, or frozen encoder + sklearn LogisticRegression on GRU states.",
    )
    parser.add_argument(
        "--gru-classifier",
        choices=["pytorch", "sklearn_lr"],
        default="pytorch",
        help="gru only: same options as --cnn-gru-classifier (PyTorch head vs frozen GRU + sklearn LR).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument(
        "--exclude-subjects",
        default="",
        metavar="IDS",
        help="Comma-separated subject ids to omit from LOSO (e.g. 7,8,11,23).",
    )
    parser.add_argument("--out-csv", default=None)
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--decision-threshold-tune",
        choices=["none", "f1_val"],
        default="none",
        help="f1_val: choose probability threshold on inner val windows to maximize F1, then apply to test. "
        "Helps when scores are shifted so 0.5 always predicts stress; requires val split with both classes.",
    )
    parser.add_argument("--print-epoch-loss", action="store_true")
    parser.add_argument(
        "--save-confusion-for",
        type=int,
        default=None,
        metavar="SUBJECT",
        help="Subject id (same as S*.pt stem). Saves confusion matrix PNG for that LOSO held-out fold only.",
    )
    parser.add_argument(
        "--confusion-out-dir",
        default="runs",
        help="Output directory for confusion PNG (used with --save-confusion-for).",
    )
    args = parser.parse_args()

    if args.architecture != "linear" and args.linear_head != "pytorch":
        parser.error("--linear-head is only valid with --architecture linear")

    if args.architecture != "cnn_gru" and args.cnn_gru_classifier != "pytorch":
        parser.error("--cnn-gru-classifier is only valid with --architecture cnn_gru")

    if args.architecture != "gru" and args.gru_classifier != "pytorch":
        parser.error("--gru-classifier is only valid with --architecture gru")

    if args.architecture == "linear" and args.linear_head == "sklearn":
        if args.linear_sklearn_penalty != "l2" and args.linear_sklearn_estimator != "logistic":
            parser.error("--linear-sklearn-penalty l1 is only valid with --linear-sklearn-estimator logistic")
    elif args.linear_sklearn_estimator != "logistic":
        parser.error("--linear-sklearn-estimator requires --architecture linear --linear-head sklearn")
    elif args.linear_sklearn_penalty != "l2":
        parser.error("--linear-sklearn-penalty requires --architecture linear --linear-head sklearn")
    elif args.linear_sklearn_scaler != "none" or args.linear_sklearn_fit_on != "inner_train":
        parser.error(
            "--linear-sklearn-scaler / --linear-sklearn-fit-on require --architecture linear --linear-head sklearn"
        )

    if args.out_csv is None:
        args.out_csv = str(_default_swell_csv_path(args))

    if args.architecture == "cnn_gru":
        conv_channels = _parse_conv_channels(args.conv_channels)
    else:
        conv_channels = [32, 32]

    _seed_all(args.seed)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

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
            raise ValueError(
                f"Need at least 2 subjects after --exclude-subjects {exclude_ids}; got {subjects}"
            )
        print(f"exclude_subjects={exclude_ids} -> LOSO folds on {subjects}")

    if args.save_confusion_for is not None and args.save_confusion_for not in subjects:
        print(
            f"WARNING: --save-confusion-for {args.save_confusion_for} not in subject list {subjects}; "
            "no confusion figure will be written."
        )

    feat_idx = FEATURE_SUBSET_INDICES[args.feature_subset]
    subj_data = {sid: _load_subject_pt(p, feat_idx) for sid, p in zip(subjects, files)}
    if args.feature_scale == "subject_neutral":
        subj_data = {
            sid: (_normalize_subject_neutral_baseline(X, y), y) for sid, (X, y) in subj_data.items()
        }
    input_dim = next(iter(subj_data.values()))[0].shape[1]

    print(
        f"SWELL LOSO | arch={args.architecture}"
        + (f" | linear_head={args.linear_head}" if args.architecture == "linear" else "")
        + (
            f" | sklearn_est={args.linear_sklearn_estimator}"
            + (
                f" penalty={args.linear_sklearn_penalty}"
                if args.linear_sklearn_estimator == "logistic"
                else ""
            )
            if args.architecture == "linear" and args.linear_head == "sklearn"
            else ""
        )
        + f" | subjects={subjects} | data_root={data_root} | "
        f"feature_subset={args.feature_subset} | feature_scale={args.feature_scale} | "
        f"seq_len={args.seq_len} | F={input_dim} | class_weight={args.class_weight} | device={device}"
    )
    if args.architecture == "cnn_gru":
        print(f"  cnn: channels={conv_channels} kernel={args.conv_kernel} | gru: hidden={args.hidden_dim} layers={args.num_layers}")
        print(f"  cnn_gru classifier: {args.cnn_gru_classifier}")
    if args.architecture == "gru":
        print(f"  gru: hidden={args.hidden_dim} layers={args.num_layers}")
        print(f"  gru classifier: {args.gru_classifier}")

    rows = []
    loss_rows = []
    for held_out in subjects:
        train_ids_all = [sid for sid in subjects if sid != held_out]
        train_ids, val_ids = _split_train_val_subjects(train_ids_all, seed=args.seed + held_out)

        mu, sigma = _fit_scaler([subj_data[sid] for sid in train_ids_all])

        X_te, y_te = subj_data[held_out]
        X_te = _apply_scaler(X_te, mu, sigma)
        X_te_seq, y_te_seq = _build_sequences(X_te, y_te, seq_len=args.seq_len)

        tr_seq_list, tr_y_list = [], []
        for sid in train_ids:
            X, y = subj_data[sid]
            X = _apply_scaler(X, mu, sigma)
            sx, sy = _build_sequences(X, y, seq_len=args.seq_len)
            if len(sy) > 0:
                tr_seq_list.append(sx)
                tr_y_list.append(sy)
        X_tr_seq = np.concatenate(tr_seq_list, axis=0) if tr_seq_list else np.zeros((0, args.seq_len, input_dim), dtype=np.float32)
        y_tr_seq = np.concatenate(tr_y_list, axis=0) if tr_y_list else np.zeros((0,), dtype=np.int64)

        if len(val_ids) > 0:
            va_seq_list, va_y_list = [], []
            for sid in val_ids:
                X, y = subj_data[sid]
                X = _apply_scaler(X, mu, sigma)
                sx, sy = _build_sequences(X, y, seq_len=args.seq_len)
                if len(sy) > 0:
                    va_seq_list.append(sx)
                    va_y_list.append(sy)
            X_va_seq = (
                np.concatenate(va_seq_list, axis=0)
                if va_seq_list
                else np.zeros((0, args.seq_len, input_dim), dtype=np.float32)
            )
            y_va_seq = np.concatenate(va_y_list, axis=0) if va_y_list else np.zeros((0,), dtype=np.int64)
        else:
            X_va_seq = np.zeros((0, args.seq_len, input_dim), dtype=np.float32)
            y_va_seq = np.zeros((0,), dtype=np.int64)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_seq)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_te_seq), torch.from_numpy(y_te_seq)),
            batch_size=args.batch_size,
            shuffle=False,
        )
        val_loader = None
        if len(y_va_seq) > 0:
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_va_seq), torch.from_numpy(y_va_seq)),
                batch_size=args.batch_size,
                shuffle=False,
            )

        use_sklearn_linear = args.architecture == "linear" and args.linear_head == "sklearn"

        pred_model = np.zeros((0,), dtype=np.int64)
        eval_threshold = float(args.decision_threshold)

        if use_sklearn_linear:
            flat_dim = args.seq_len * input_dim
            if args.linear_sklearn_fit_on == "loso_train":
                tr_seq_list_lr, tr_y_list_lr = [], []
                for sid in train_ids_all:
                    X, y = subj_data[sid]
                    X = _apply_scaler(X, mu, sigma)
                    sx, sy = _build_sequences(X, y, seq_len=args.seq_len)
                    if len(sy) > 0:
                        tr_seq_list_lr.append(sx)
                        tr_y_list_lr.append(sy)
                X_tr_lr = (
                    np.concatenate(tr_seq_list_lr, axis=0)
                    if tr_seq_list_lr
                    else np.zeros((0, args.seq_len, input_dim), dtype=np.float32)
                )
                y_tr_lr = (
                    np.concatenate(tr_y_list_lr, axis=0)
                    if tr_y_list_lr
                    else np.zeros((0,), dtype=np.int64)
                )
            else:
                X_tr_lr, y_tr_lr = X_tr_seq, y_tr_seq

            X_tr_flat = X_tr_lr.reshape(len(y_tr_lr), flat_dim) if len(y_tr_lr) else np.zeros((0, flat_dim), dtype=np.float64)
            X_te_flat = X_te_seq.reshape(len(y_te_seq), flat_dim) if len(y_te_seq) else np.zeros((0, flat_dim), dtype=np.float64)

            sk_class_weight = "balanced" if args.class_weight == "balanced" else None
            if len(y_tr_lr) == 0:
                acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0
            else:
                clf = _build_sklearn_pipeline(
                    args, class_weight=sk_class_weight, y_train=y_tr_lr
                )
                clf.fit(X_tr_flat, y_tr_lr)
                if (
                    args.decision_threshold_tune == "f1_val"
                    and len(y_va_seq) >= 2
                    and len(np.unique(y_va_seq)) >= 2
                    and X_va_seq.shape[0] > 0
                ):
                    X_va_flat = X_va_seq.reshape(len(y_va_seq), flat_dim)
                    p_va = clf.predict_proba(X_va_flat)[:, 1].astype(np.float64)
                    eval_threshold = _best_threshold_f1(y_va_seq, p_va)
                if len(y_te_seq) == 0:
                    acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0
                else:
                    probs = clf.predict_proba(X_te_flat)[:, 1]
                    pred = (probs >= eval_threshold).astype(np.int64)
                    pred_model = pred
                    acc, f1, prec, rec = _metrics_from_arrays(y_te_seq, pred)
        else:
            model = _build_model(
                args.architecture,
                seq_len=args.seq_len,
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
                conv_channels=conv_channels,
                conv_kernel=args.conv_kernel,
            ).to(device)

            if args.class_weight == "balanced" and len(y_tr_seq) > 0:
                n_neg = float(np.sum(y_tr_seq == 0))
                n_pos = float(np.sum(y_tr_seq == 1))
                w0 = 0.5 * (n_neg + n_pos) / max(n_neg, 1.0)
                w1 = 0.5 * (n_neg + n_pos) / max(n_pos, 1.0)
                class_w = torch.tensor([w0, w1], dtype=torch.float32, device=device)
                criterion = nn.CrossEntropyLoss(weight=class_w)
            else:
                criterion = nn.CrossEntropyLoss()

            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            best_state = None
            best_val_loss = float("inf")
            bad_epochs = 0

            for _ in range(args.epochs):
                epoch_idx = _ + 1
                model.train()
                train_losses = []
                for xb, yb in train_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model(xb)
                    loss = criterion(logits, yb)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_losses.append(float(loss.item()))

                train_loss = float(np.mean(train_losses)) if train_losses else 0.0
                val_loss = float("nan")

                if val_loader is not None:
                    val_loss, _, _, _, _ = _evaluate(model, val_loader, device, args.decision_threshold)
                    if val_loss < best_val_loss - 1e-6:
                        best_val_loss = val_loss
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                        if bad_epochs >= args.patience:
                            loss_rows.append(
                                {
                                    "subject": held_out,
                                    "epoch": epoch_idx,
                                    "train_loss": train_loss,
                                    "val_loss": float(val_loss),
                                    "best_val_loss": float(best_val_loss),
                                    "bad_epochs": int(bad_epochs),
                                }
                            )
                            if args.print_epoch_loss:
                                print(
                                    f"    [S{held_out}] epoch={epoch_idx} train_loss={train_loss:.5f} "
                                    f"val_loss={val_loss:.5f} best_val={best_val_loss:.5f} bad_epochs={bad_epochs}"
                                )
                            break

                loss_rows.append(
                    {
                        "subject": held_out,
                        "epoch": epoch_idx,
                        "train_loss": train_loss,
                        "val_loss": float(val_loss) if val_loss == val_loss else "",
                        "best_val_loss": float(best_val_loss) if best_val_loss < float("inf") else "",
                        "bad_epochs": int(bad_epochs),
                    }
                )
                if args.print_epoch_loss:
                    val_str = f"{val_loss:.5f}" if val_loss == val_loss else "NA"
                    best_str = f"{best_val_loss:.5f}" if best_val_loss < float("inf") else "NA"
                    print(
                        f"    [S{held_out}] epoch={epoch_idx} train_loss={train_loss:.5f} "
                        f"val_loss={val_str} best_val={best_str} bad_epochs={bad_epochs}"
                    )

            if best_state is not None:
                model.load_state_dict(best_state)

            use_sklearn_neural_head = (
                (args.architecture == "cnn_gru" and args.cnn_gru_classifier == "sklearn_lr")
                or (args.architecture == "gru" and args.gru_classifier == "sklearn_lr")
            )
            if use_sklearn_neural_head:
                if not isinstance(model, (SwellCnnGruClassifier, SwellGruClassifier)):
                    raise TypeError("Expected SwellCnnGruClassifier or SwellGruClassifier for sklearn_lr head")
                model.eval()
                train_enc_loader = DataLoader(
                    TensorDataset(torch.from_numpy(X_tr_seq), torch.from_numpy(y_tr_seq)),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                Z_tr, y_enc_tr = _collect_encoder_features(model, train_enc_loader, device)
                Z_te, y_enc_te = _collect_encoder_features(model, test_loader, device)
                sk_class_weight = "balanced" if args.class_weight == "balanced" else None
                if len(y_enc_tr) == 0 or len(y_enc_te) == 0:
                    acc, f1, prec, rec = 0.0, 0.0, 0.0, 0.0
                    pred_model = np.zeros((0,), dtype=np.int64)
                else:
                    clf_head = LogisticRegression(
                        **_sklearn_logistic_regression_kwargs(args, class_weight=sk_class_weight)
                    )
                    clf_head.fit(Z_tr, y_enc_tr)
                    if (
                        args.decision_threshold_tune == "f1_val"
                        and val_loader is not None
                        and len(y_va_seq) >= 2
                        and len(np.unique(y_va_seq)) >= 2
                    ):
                        Z_va, y_va_enc = _collect_encoder_features(model, val_loader, device)
                        if len(y_va_enc) >= 2 and len(np.unique(y_va_enc)) >= 2:
                            p_va = clf_head.predict_proba(Z_va)[:, 1].astype(np.float64)
                            eval_threshold = _best_threshold_f1(y_va_enc, p_va)
                    probs = clf_head.predict_proba(Z_te)[:, 1]
                    pred_model = (probs >= eval_threshold).astype(np.int64)
                    acc, f1, prec, rec = _metrics_from_arrays(y_enc_te, pred_model)
            else:
                if (
                    args.decision_threshold_tune == "f1_val"
                    and val_loader is not None
                    and len(y_va_seq) >= 2
                    and len(np.unique(y_va_seq)) >= 2
                ):
                    y_val_np, p_val_np = _collect_probs_torch(model, val_loader, device)
                    if len(y_val_np) >= 2 and len(np.unique(y_val_np)) >= 2:
                        eval_threshold = _best_threshold_f1(y_val_np, p_val_np)
                _, acc, f1, prec, rec = _evaluate(model, test_loader, device, eval_threshold)
                if len(y_te_seq) > 0:
                    _, pred_model = _collect_predictions_torch(model, test_loader, device, eval_threshold)
                else:
                    pred_model = np.zeros((0,), dtype=np.int64)

        maj_label, maj_acc, maj_f1, maj_prec, maj_rec = _majority_baseline_metrics(y_te_seq, y_tr_seq)
        train_stress_pct = float(np.mean(y_tr_seq == 1) * 100.0) if len(y_tr_seq) else 0.0

        if len(y_te_seq) > 0 and len(pred_model) == len(y_te_seq):
            spec, bacc, pstress = _diagnostic_metrics(y_te_seq, pred_model)
        else:
            spec, bacc, pstress = 0.0, 0.0, 0.0

        if (
            args.save_confusion_for is not None
            and held_out == args.save_confusion_for
            and len(y_te_seq) > 0
            and len(pred_model) == len(y_te_seq)
        ):
            slug = _arch_slug(args)
            cpath = Path(args.confusion_out_dir) / f"swell_confusion_{slug}_S{held_out}.png"
            _save_confusion_png(
                y_te_seq,
                pred_model,
                cpath,
                title=f"SWELL LOSO held-out S{held_out} ({slug})",
            )
            print(f"    saved confusion: {cpath}")

        row = {
            "subject": held_out,
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "specificity": float(spec),
            "balanced_accuracy": float(bacc),
            "pred_stress_pct": float(pstress),
            "decision_threshold_used": float(eval_threshold),
            "n_samples": int(len(y_te_seq)),
            "stress_pct": float(np.mean(y_te_seq == 1) * 100.0) if len(y_te_seq) else 0.0,
            "train_stress_pct": train_stress_pct,
            "maj_label": maj_label,
            "maj_accuracy": float(maj_acc),
            "maj_f1": float(maj_f1),
            "maj_precision": float(maj_prec),
            "maj_recall": float(maj_rec),
        }
        rows.append(row)
        warn = ""
        if pstress >= 95.0 and rec >= 0.99:
            warn = " | WARN: near-all-stress predictions (recall≈1)"
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
                "subject": "MEAN±STD",
                "accuracy": f"{acc.mean():.6f}±{acc.std():.6f}",
                "f1": f"{f1.mean():.6f}±{f1.std():.6f}",
                "precision": f"{pr.mean():.6f}±{pr.std():.6f}",
                "recall": f"{rc.mean():.6f}±{rc.std():.6f}",
                "specificity": f"{spec.mean():.6f}±{spec.std():.6f}",
                "balanced_accuracy": f"{bacc.mean():.6f}±{bacc.std():.6f}",
                "pred_stress_pct": f"{psp.mean():.6f}±{psp.std():.6f}",
                "decision_threshold_used": f"{thr_u.mean():.6f}±{thr_u.std():.6f}",
                "n_samples": int(np.mean([r["n_samples"] for r in rows])) if rows else 0,
                "stress_pct": f"{np.mean([r['stress_pct'] for r in rows]):.3f}" if rows else "",
                "train_stress_pct": f"{np.mean([r['train_stress_pct'] for r in rows]):.3f}" if rows else "",
                "maj_label": "",
                "maj_accuracy": f"{maj_acc.mean():.6f}±{maj_acc.std():.6f}",
                "maj_f1": f"{maj_f1.mean():.6f}±{maj_f1.std():.6f}",
                "maj_precision": f"{maj_pr.mean():.6f}±{maj_pr.std():.6f}",
                "maj_recall": f"{maj_rc.mean():.6f}±{maj_rc.std():.6f}",
            }
        )

    print(f"mean_accuracy: {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"mean_f1: {f1.mean():.3f} ± {f1.std():.3f}")
    print(f"mean_balanced_accuracy: {bacc.mean():.3f} ± {bacc.std():.3f}")
    print(f"mean_majority_f1: {maj_f1.mean():.3f} ± {maj_f1.std():.3f} (train-label majority baseline)")
    print(f"saved_csv: {out_csv}")


if __name__ == "__main__":
    main()
