#!/usr/bin/env python3
"""
LOSO benchmarks on SWELL waveform ``S*.pt`` (from ``preprocess_swell_waveform.py``).

Trains **one modality at a time** (``eda`` or ``ecg``) so ECG and EDA are never mixed
in the encoder input. Architectures: ``linear``, ``gru``, ``cnn_gru`` (reuse
``src/models/swell_baselines.py`` with ``input_dim=1`` and one sample = one window).

Writes (each modality × architecture is **separate files**, no overwrites between runs):

- LOSO summary CSV: ``runs/swell_waveform_loso_{modality}_{architecture}.csv``
- Test embeddings (one file per held-out subject, unique name):
  ``{embeddings_dir}/S{subject}_{modality}_{architecture}_test.npz``
- Optional inner-train embeddings: ``..._inner_train.npz``
- Manifest listing those npz paths: ``{embeddings_dir}/manifest_{modality}_{architecture}.json``

Use ``--embeddings-nested`` to use the older layout
``{embeddings_dir}/{modality}_{architecture}/S{subject}_test.npz`` instead.

Use ``--run-all`` to sweep modalities × architectures (six CSVs + six manifests + all npzs).
Use ``--run-all-modality eda`` or ``ecg`` to run only three combos (parallel-friendly).
Use ``--skip-existing`` to skip a modality+architecture if its output CSV already exists and is non-empty.
Use ``--results-subdir NAME`` to write default CSV paths and the default embeddings directory under
``runs/NAME/`` so a new window length does not overwrite or falsely skip against an older ``runs/`` tree.

Example:

    python scripts/benchmark_swell_waveform_loso.py --architecture gru --modality eda \\
        --data-root data/processed_swell_waveform

    python scripts/benchmark_swell_waveform_loso.py --run-all --data-root data/processed_swell_waveform

Quick check before a long run::

    python scripts/benchmark_swell_waveform_loso.py --smoke-test --no-save-embeddings \\
        --modality eda --architecture linear

    bash scripts/smoke_swell_waveform_loso.sh
"""

from __future__ import annotations

import argparse
import copy
import json
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.swell_baselines import (
    SwellCnnGruClassifier,
    SwellFlattenLinearClassifier,
    SwellGruClassifier,
)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _best_threshold_balanced(
    y_true: np.ndarray, probs: np.ndarray, objective: str = "bacc", n_steps: int = 49
) -> float:
    """Threshold that maximizes balanced accuracy or Youden's J.

    Unlike F1, these objectives drop to ~0 when the model predicts a single class
    (specificity becomes 0), so the tuner cannot 'cheat' by calling everything stress.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return 0.5
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, n_steps):
        pred = (probs >= t).astype(np.int64)
        sens = float(
            recall_score(y_true, pred, labels=[0, 1], average="binary", pos_label=1, zero_division=0)
        )
        spec = float(
            recall_score(y_true, pred, labels=[0, 1], average="binary", pos_label=0, zero_division=0)
        )
        score = 0.5 * (sens + spec) if objective == "bacc" else (sens + spec - 1.0)
        if score > best_score or (score == best_score and abs(t - 0.5) < abs(best_t - 0.5)):
            best_score, best_t = score, float(t)
    return best_t


def _majority_train_label(y_tr: np.ndarray) -> int:
    if len(y_tr) == 0:
        return 0
    y_tr = y_tr.astype(np.int64)
    c0, c1 = int(np.sum(y_tr == 0)), int(np.sum(y_tr == 1))
    return 1 if c1 > c0 else 0


def _majority_baseline_metrics(y_te: np.ndarray, y_tr: np.ndarray) -> tuple[int, float, float, float, float]:
    maj = _majority_train_label(y_tr)
    if len(y_te) == 0:
        return maj, 0.0, 0.0, 0.0, 0.0
    pred = np.full(len(y_te), maj, dtype=np.int64)
    acc, f1, prec, rec = _metrics_from_arrays(y_te, pred)
    return maj, acc, f1, prec, rec


def _parse_conv_channels(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("--conv-channels must be a comma-separated list, e.g. 64,64")
    return [int(p) for p in parts]


def _parse_conv_kernels(s: str) -> list[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if not parts:
        raise ValueError("--conv-kernels must be a comma-separated list, e.g. 7,5")
    return [int(p) for p in parts]


def _downsample_windows(X: np.ndarray, orig_hz: float, target_hz: float) -> np.ndarray:
    """Anti-aliased resample of (N, T) windows from ``orig_hz`` to ``target_hz``.

    No-op when target is missing, non-positive, or not below the original rate.
    """
    if target_hz is None or target_hz <= 0 or orig_hz <= 0 or target_hz >= orig_hz:
        return X
    if X.shape[0] == 0 or X.shape[1] == 0:
        return X
    from scipy.signal import resample_poly

    up = int(round(target_hz))
    down = int(round(orig_hz))
    g = math.gcd(up, down) or 1
    up //= g
    down //= g
    Xd = resample_poly(X, up, down, axis=1)
    return np.ascontiguousarray(Xd, dtype=np.float32)


def _load_waveform_subject(
    path: Path, modality: str, downsample_hz: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    d = torch.load(path, weights_only=False, map_location="cpu")
    if d.get("kind") != "swell_waveform":
        raise ValueError(f"Expected kind swell_waveform in {path}, got {d.get('kind')!r}")
    if modality == "eda" and "eda_waveform_windows" in d:
        X = np.asarray(d["eda_waveform_windows"], dtype=np.float32)
    elif modality == "ecg" and "ecg_waveform_windows" in d:
        X = np.asarray(d["ecg_waveform_windows"], dtype=np.float32)
    else:
        w = np.asarray(d["waveform_windows"], dtype=np.float32)
        if w.ndim != 3 or w.shape[2] < 2:
            raise ValueError(f"waveform_windows must be (N,T,>=2), got {w.shape}")
        ch = 0 if modality == "eda" else 1
        X = w[:, :, ch]
    y = np.asarray(d["labels"], dtype=np.int64)
    valid = np.isfinite(X).all(axis=1)
    X, y = X[valid], y[valid]
    if downsample_hz is not None and len(X):
        orig_hz = float(d.get("sample_rate_hz") or 0.0)
        if orig_hz <= 0:
            wsec = float(d.get("win_sec") or 0.0)
            if wsec > 0:
                orig_hz = X.shape[1] / wsec
        X = _downsample_windows(X, orig_hz, float(downsample_hz))
    return X, y


def _fit_scaler(xs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    Xcat = np.concatenate(xs, axis=0)
    mu = Xcat.mean(axis=0)
    sigma = Xcat.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu.astype(np.float32), sigma.astype(np.float32)


def _apply_scaler(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((X - mu) / sigma).astype(np.float32)


def _build_model(
    architecture: str,
    win_samples: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    conv_channels: list[int],
    conv_kernel: int,
    *,
    conv_style: str = "physio",
    conv_kernels: list[int] | None = None,
    conv_pool: int = 4,
    pool: str = "last",
    input_norm: bool = True,
    head: str = "linear",
) -> nn.Module:
    def _supported_kwargs(cls: type[nn.Module], kwargs: dict) -> dict:
        params = inspect.signature(cls.__init__).parameters
        return {k: v for k, v in kwargs.items() if k in params}

    if architecture == "linear":
        return SwellFlattenLinearClassifier(seq_len=win_samples, input_dim=1, num_classes=2)
    if architecture == "gru":
        kwargs = _supported_kwargs(
            SwellGruClassifier,
            {
                "input_dim": 1,
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "dropout": dropout,
                "num_classes": 2,
                "pool": pool,
                "input_norm": input_norm,
                "head": head,
            },
        )
        return SwellGruClassifier(**kwargs)
    if architecture == "cnn_gru":
        kwargs = _supported_kwargs(
            SwellCnnGruClassifier,
            {
                "input_dim": 1,
                "conv_channels": conv_channels,
                "kernel_size": conv_kernel,
                "gru_hidden": hidden_dim,
                "gru_layers": num_layers,
                "dropout": dropout,
                "num_classes": 2,
                "conv_style": conv_style,
                "conv_kernels": conv_kernels,
                "conv_pool": conv_pool,
                "pool": pool,
                "input_norm": input_norm,
                "head": head,
            },
        )
        return SwellCnnGruClassifier(**kwargs)
    raise ValueError(f"Unknown architecture: {architecture}")


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    decision_threshold: float,
) -> tuple[float, float, float, float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    y_true, y_pred, losses = [], [], []
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
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    acc, f1, prec, rec = _metrics_from_arrays(y_true_np, y_pred_np)
    return float(np.mean(losses)) if losses else 0.0, acc, f1, prec, rec


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


def _collect_encoder(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    h_list: list[np.ndarray] = []
    y_list: list[np.ndarray] = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            h = model.encode(xb)
            h_list.append(h.cpu().numpy().astype(np.float32))
            y_list.append(yb.numpy().astype(np.int64))
    if not h_list:
        return np.zeros((0, 1), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(h_list, axis=0), np.concatenate(y_list, axis=0)


def _to_model_input(X: np.ndarray) -> np.ndarray:
    """(N, W) -> (N, W, 1) for sequence models."""
    return X[:, :, np.newaxis].astype(np.float32, copy=False)


_DEFAULT_WAVEFORM_EMBEDDINGS_DIR = Path("runs/swell_waveform_embeddings")


def _default_out_csv(modality: str, architecture: str, runs_parent: Path | None = None) -> Path:
    base = runs_parent if runs_parent is not None else Path("runs")
    return base / f"swell_waveform_loso_{modality}_{architecture}.csv"


def _default_embeddings_subdir(modality: str, architecture: str) -> str:
    return f"{modality}_{architecture}"


def _embedding_paths(
    emb_root: Path,
    held_out: int,
    modality: str,
    architecture: str,
    nested: bool,
    *,
    kind: str,
) -> Path:
    """Return path for one embedding npz (unique stem includes modality + architecture)."""
    stem = f"S{held_out}_{modality}_{architecture}_{kind}.npz"
    if nested:
        return emb_root / _default_embeddings_subdir(modality, architecture) / stem
    return emb_root / stem


def run_loso(args: argparse.Namespace) -> None:
    modality = args.modality
    architecture = args.architecture
    runs_parent: Path = getattr(args, "_runs_parent", Path("runs"))
    out_csv_path = Path(args.out_csv) if args.out_csv else _default_out_csv(modality, architecture, runs_parent)
    if getattr(args, "skip_existing", False) and out_csv_path.is_file() and out_csv_path.stat().st_size > 20:
        print(f"Skipping existing results (--skip-existing): {out_csv_path}")
        return

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
        raise FileNotFoundError(f"No S*.pt under {data_root}")

    subjects = [int(p.stem.lstrip("S")) for p in files]
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
        files = [data_root / f"S{sid}.pt" for sid in subjects]

    subj_data: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    win_ref: int | None = None
    downsample_hz = getattr(args, "downsample_hz", None)
    for sid, p in zip(subjects, files):
        X, y = _load_waveform_subject(p, modality, downsample_hz=downsample_hz)
        if win_ref is None:
            win_ref = int(X.shape[1])
        elif int(X.shape[1]) != win_ref:
            raise ValueError(f"Inconsistent win_samples: S{sid} has {X.shape[1]}, expected {win_ref}")
        subj_data[sid] = (X, y)

    conv_channels = _parse_conv_channels(args.conv_channels) if architecture == "cnn_gru" else [32, 32]
    conv_kernels = (
        _parse_conv_kernels(args.conv_kernels) if architecture == "cnn_gru" else None
    )
    if architecture == "cnn_gru" and getattr(args, "cnn_gru_style", "physio") == "physio":
        if conv_kernels is not None and len(conv_kernels) != len(conv_channels):
            raise ValueError(
                f"--conv-kernels ({len(conv_kernels)}) must match --conv-channels ({len(conv_channels)})"
            )

    ds_note = f" | downsample_hz={downsample_hz}" if downsample_hz else ""
    cnn_note = ""
    if architecture == "cnn_gru":
        cnn_note = (
            f" | cnn={getattr(args, 'cnn_gru_style', 'physio')}"
            f" ch={conv_channels} k={conv_kernels or args.conv_kernel}"
        )
    print(
        f"SWELL waveform LOSO | modality={modality} | arch={architecture} | "
        f"subjects={subjects} | W={win_ref}{ds_note}{cnn_note} | H={args.hidden_dim} | pool={args.pool} | "
        f"head={args.head} | device={device}"
    )
    if getattr(args, "smoke_test", False):
        print("  (smoke-test: short run for wiring checks; metrics are not meaningful)")

    rows: list[dict] = []
    prediction_rows: list[dict] = []
    embedding_files_written: list[str] = []
    for held_out in subjects:
        train_ids_all = [sid for sid in subjects if sid != held_out]
        train_ids, val_ids = _split_train_val_subjects(train_ids_all, seed=args.seed + held_out)

        mu, sigma = _fit_scaler([subj_data[sid][0] for sid in train_ids_all])

        X_te, y_te = subj_data[held_out]
        X_te_s = _apply_scaler(X_te, mu, sigma)

        tr_list, tr_y = [], []
        for sid in train_ids:
            X, y = subj_data[sid]
            Xs = _apply_scaler(X, mu, sigma)
            if len(y) > 0:
                tr_list.append(Xs)
                tr_y.append(y)
        X_tr = np.concatenate(tr_list, axis=0) if tr_list else np.zeros((0, win_ref), dtype=np.float32)
        y_tr = np.concatenate(tr_y, axis=0) if tr_y else np.zeros((0,), dtype=np.int64)

        va_list, va_y = [], []
        for sid in val_ids:
            X, y = subj_data[sid]
            Xs = _apply_scaler(X, mu, sigma)
            if len(y) > 0:
                va_list.append(Xs)
                va_y.append(y)
        X_va = np.concatenate(va_list, axis=0) if va_list else np.zeros((0, win_ref), dtype=np.float32)
        y_va = np.concatenate(va_y, axis=0) if va_y else np.zeros((0,), dtype=np.int64)

        X_tr_m = _to_model_input(X_tr)
        X_va_m = _to_model_input(X_va)
        X_te_m = _to_model_input(X_te_s)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr_m), torch.from_numpy(y_tr)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_te_m), torch.from_numpy(y_te)),
            batch_size=args.batch_size,
            shuffle=False,
        )
        val_loader = None
        if len(y_va) > 0:
            val_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_va_m), torch.from_numpy(y_va)),
                batch_size=args.batch_size,
                shuffle=False,
            )

        model = _build_model(
            architecture,
            win_ref,
            args.hidden_dim,
            args.num_layers,
            args.dropout,
            conv_channels,
            args.conv_kernel,
            conv_style=getattr(args, "cnn_gru_style", "physio"),
            conv_kernels=conv_kernels,
            conv_pool=getattr(args, "conv_pool", 4),
            pool=args.pool,
            input_norm=not args.no_input_norm,
            head=args.head,
        ).to(device)

        if args.class_weight == "balanced" and len(y_tr) > 0:
            n_neg = float(np.sum(y_tr == 0))
            n_pos = float(np.sum(y_tr == 1))
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
        eval_threshold = float(args.decision_threshold)

        for epoch in range(args.epochs):
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

            val_loss = float("nan")
            if val_loader is not None:
                val_loss, _, _, _, _ = _evaluate(model, val_loader, device, eval_threshold)
                if val_loss < best_val_loss - 1e-6:
                    best_val_loss = val_loss
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= args.patience:
                        break

        if best_state is not None:
            model.load_state_dict(best_state)

        if (
            args.decision_threshold_tune != "none"
            and val_loader is not None
            and len(y_va) >= 2
            and len(np.unique(y_va)) >= 2
        ):
            y_val_np, p_val_np = _collect_probs_torch(model, val_loader, device)
            if len(np.unique(y_val_np)) >= 2:
                if args.decision_threshold_tune == "f1_val":
                    eval_threshold = _best_threshold_f1(y_val_np, p_val_np)
                elif args.decision_threshold_tune == "bacc_val":
                    eval_threshold = _best_threshold_balanced(y_val_np, p_val_np, objective="bacc")
                elif args.decision_threshold_tune == "youden_val":
                    eval_threshold = _best_threshold_balanced(y_val_np, p_val_np, objective="youden")

        _, acc, f1, prec, rec = _evaluate(model, test_loader, device, eval_threshold)
        if len(y_te) > 0:
            _, pred_model = _collect_predictions_torch(model, test_loader, device, eval_threshold)
        else:
            pred_model = np.zeros((0,), dtype=np.int64)

        y_tr_inner = y_tr
        maj_label, maj_acc, maj_f1, maj_prec, maj_rec = _majority_baseline_metrics(y_te, y_tr_inner)
        train_stress_pct = float(np.mean(y_tr_inner == 1) * 100.0) if len(y_tr_inner) else 0.0

        if len(y_te) > 0 and len(pred_model) == len(y_te):
            spec, bacc, pstress = _diagnostic_metrics(y_te, pred_model)
        else:
            spec, bacc, pstress = 0.0, 0.0, 0.0

        Z_te, y_enc_te = _collect_encoder(model, test_loader, device)
        _, p_te = _collect_probs_torch(model, test_loader, device)
        if len(y_te) == len(pred_model) == len(p_te):
            for window_index, y_true, y_pred, p_stress in zip(
                np.arange(len(y_te), dtype=np.int64),
                y_te,
                pred_model,
                p_te,
                strict=True,
            ):
                prediction_rows.append(
                    {
                        "subject": int(held_out),
                        "modality": modality,
                        "architecture": architecture,
                        "window_index": int(window_index),
                        "y_true": int(y_true),
                        "y_pred": int(y_pred),
                        "p_stress": float(p_stress),
                        "decision_threshold": float(eval_threshold),
                        "win_samples": int(win_ref),
                    }
                )

        if args.save_embeddings:
            emb_root = Path(args.embeddings_dir)
            emb_root.mkdir(parents=True, exist_ok=True)
            nested = bool(args.embeddings_nested)
            test_path = _embedding_paths(emb_root, held_out, modality, architecture, nested, kind="test")
            if nested:
                test_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                test_path,
                Z_test=Z_te.astype(np.float32, copy=False),
                y_test=y_te.astype(np.int64, copy=False),
                p_test=p_te.astype(np.float32, copy=False),
                held_out_subject=np.int32(held_out),
                modality=np.array(modality),
                architecture=np.array(architecture),
                win_samples=np.int32(win_ref),
                embedding_dim=np.int32(Z_te.shape[1] if len(Z_te) else 0),
            )
            embedding_files_written.append(str(test_path.resolve()))
            if args.save_train_embeddings and len(y_tr) > 0:
                tr_loader = DataLoader(
                    TensorDataset(torch.from_numpy(X_tr_m), torch.from_numpy(y_tr)),
                    batch_size=args.batch_size,
                    shuffle=False,
                )
                Z_trn, _ = _collect_encoder(model, tr_loader, device)
                _, p_trn = _collect_probs_torch(model, tr_loader, device)
                train_path = _embedding_paths(emb_root, held_out, modality, architecture, nested, kind="inner_train")
                if nested:
                    train_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(
                    train_path,
                    Z_train=Z_trn.astype(np.float32, copy=False),
                    y_train=y_tr.astype(np.int64, copy=False),
                    p_train=p_trn.astype(np.float32, copy=False),
                    held_out_subject=np.int32(held_out),
                    modality=np.array(modality),
                    architecture=np.array(architecture),
                )
                embedding_files_written.append(str(train_path.resolve()))

        rows.append(
            {
                "subject": held_out,
                "modality": modality,
                "architecture": architecture,
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
                "win_samples": int(win_ref),
            }
        )
        print(
            f"  S{held_out}: acc={acc:.3f} f1={f1:.3f} prec={prec:.3f} rec={rec:.3f} "
            f"spec={spec:.3f} bacc={bacc:.3f} pred_stress%={pstress:.1f} "
            f"(n={len(y_te)}) | maj f1={maj_f1:.3f}"
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    pred_csv = out_csv.with_name(f"{out_csv.stem}_predictions{out_csv.suffix}")
    pd.DataFrame(prediction_rows).to_csv(pred_csv, index=False)
    print(f"Wrote {pred_csv}")

    if args.save_embeddings:
        emb_root = Path(args.embeddings_dir)
        manifest = {
            "out_csv": str(out_csv.resolve()),
            "data_root": str(data_root.resolve()),
            "modality": modality,
            "architecture": architecture,
            "embeddings_nested": bool(args.embeddings_nested),
            "embeddings_dir": str(emb_root.resolve()),
            "embedding_files": sorted(embedding_files_written),
            "keys_per_fold_npz": [
                "Z_test",
                "y_test",
                "p_test",
                "held_out_subject",
                "modality",
                "architecture",
                "win_samples",
                "embedding_dim",
            ],
        }
        man_name = f"manifest_{modality}_{architecture}.json"
        man_path = emb_root / man_name
        man_path.parent.mkdir(parents=True, exist_ok=True)
        man_path.write_text(json.dumps(manifest, indent=2))
        print(f"Wrote {man_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="SWELL waveform LOSO (EDA or ECG separately)")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed_swell_waveform"))
    parser.add_argument("--modality", choices=["eda", "ecg"], default="eda", help="Ignored when --run-all")
    parser.add_argument("--architecture", choices=["linear", "gru", "cnn_gru"], default="gru", help="Ignored when --run-all")
    parser.add_argument("--run-all", action="store_true", help="Run eda+ecg × linear+gru+cnn_gru (six CSVs)")
    parser.add_argument(
        "--run-all-modality",
        choices=["both", "eda", "ecg"],
        default="both",
        help="With --run-all: sweep both modalities (default) or only eda / only ecg (three runs). Ignored without --run-all.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="If the output CSV for this modality+architecture already exists and is non-empty, skip that run.",
    )
    parser.add_argument("--hidden-dim", type=int, default=64, help="GRU hidden size (64 for physio CNN-GRU)")
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument(
        "--cnn-gru-style",
        choices=["legacy", "physio"],
        default="physio",
        help="cnn_gru trunk: physio = two Conv-BN-ReLU-MaxPool blocks (paper); legacy = length-preserving conv stack.",
    )
    parser.add_argument(
        "--conv-channels",
        default="64,64",
        help="cnn_gru: comma-separated Conv1d output widths per block (default 64,64 for physio).",
    )
    parser.add_argument(
        "--conv-kernels",
        default="7,5",
        help="cnn_gru physio: comma-separated kernel sizes per block (default 7,5). Ignored for legacy style.",
    )
    parser.add_argument(
        "--conv-pool",
        type=int,
        default=4,
        help="cnn_gru physio: MaxPool1d factor after each block (default 4 → 16× total length reduction).",
    )
    parser.add_argument("--conv-kernel", type=int, default=3, help="cnn_gru legacy: shared kernel size for all conv layers.")
    parser.add_argument(
        "--pool",
        choices=["attn", "last"],
        default="last",
        help="Temporal pooling after GRU: last = final timestep (default, simpler); attn = learned weighted mean.",
    )
    parser.add_argument(
        "--head",
        choices=["linear", "mlp"],
        default="linear",
        help="Classification head after pooling: linear (default) or two-layer MLP.",
    )
    parser.add_argument(
        "--no-input-norm",
        action="store_true",
        help="Disable LayerNorm on (B, T, 1) inputs before conv/GRU.",
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
        "--downsample-hz",
        type=float,
        default=None,
        help="Anti-aliased downsample of raw waveform windows to this rate (e.g. 128) before "
        "training. Shortens long high-rate ECG sequences so the GRU is learnable. No-op if >= original rate.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Quick sanity check: epochs=1, patience=1, cap to first 4 subjects (unless --max-subjects set). "
        "Combine with --no-save-embeddings for fastest run. When --out-csv omitted, writes under runs/_smoke_waveform_loso_*.csv",
    )
    parser.add_argument("--out-csv", type=str, default=None)
    parser.add_argument(
        "--results-subdir",
        default=None,
        metavar="NAME",
        help="Write default out CSVs (and default --embeddings-dir) under runs/NAME/ to avoid clobbering other preprocess variants.",
    )
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--decision-threshold-tune",
        choices=["none", "f1_val", "bacc_val", "youden_val"],
        default="none",
        help="Pick the decision threshold on the validation split. f1_val maximizes F1 (can "
        "collapse to all-positive when classes are balanced); bacc_val/youden_val maximize "
        "balanced accuracy / Youden's J, which penalize predicting a single class.",
    )
    parser.add_argument("--no-save-embeddings", action="store_true", help="Skip writing per-fold npz files")
    parser.add_argument("--save-train-embeddings", action="store_true", help="Also save inner-train Z per fold (large)")
    parser.add_argument("--embeddings-dir", type=Path, default=_DEFAULT_WAVEFORM_EMBEDDINGS_DIR)
    parser.add_argument(
        "--embeddings-nested",
        action="store_true",
        help="Put npz files under {embeddings_dir}/{modality}_{architecture}/ instead of flat unique names",
    )
    args = parser.parse_args()

    if args.results_subdir:
        args._runs_parent = Path("runs") / args.results_subdir
        args._runs_parent.mkdir(parents=True, exist_ok=True)
        if args.embeddings_dir == _DEFAULT_WAVEFORM_EMBEDDINGS_DIR:
            args.embeddings_dir = args._runs_parent / "swell_waveform_embeddings"
    else:
        args._runs_parent = Path("runs")

    if args.smoke_test:
        args.epochs = 1
        args.patience = 1
        if args.max_subjects is None:
            args.max_subjects = 4

    args.save_embeddings = not args.no_save_embeddings

    if args.smoke_test and not args.out_csv:
        if args.run_all:
            pass  # set per combo in loop below
        else:
            args.out_csv = str(args._runs_parent / f"_smoke_waveform_loso_{args.modality}_{args.architecture}.csv")

    if args.run_all:
        if args.run_all_modality == "both":
            modalities: tuple[str, ...] = ("eda", "ecg")
        else:
            modalities = (args.run_all_modality,)
        combos = [(m, a) for m in modalities for a in ("linear", "gru", "cnn_gru")]
        for modality, architecture in combos:
            a = copy.deepcopy(args)
            a.modality = modality
            a.architecture = architecture
            if a.smoke_test and not a.out_csv:
                a.out_csv = str(a._runs_parent / f"_smoke_waveform_loso_{modality}_{architecture}.csv")
            else:
                a.out_csv = str(_default_out_csv(modality, architecture, a._runs_parent))
            a.run_all = False
            print("===", modality, architecture, "->", a.out_csv, "===")
            run_loso(a)
        return

    if args.out_csv is None:
        args.out_csv = str(_default_out_csv(args.modality, args.architecture, args._runs_parent))
    run_loso(args)


if __name__ == "__main__":
    main()
