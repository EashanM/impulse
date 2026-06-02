"""LOSO fold helpers for Albaladejo-style SWELL HRV sequence GRU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from src.data.swell_hrv_albaladejo import DEFAULT_EXCLUDE_SUBJECTS
from src.models.gru_classifier import GRUClassifier, LSTMClassifier

DEFAULT_EXCLUDE = list(DEFAULT_EXCLUDE_SUBJECTS)


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_subject_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = np.load(path)
    x = d["X"].astype(np.float32)
    y = d["y"].astype(np.int64)
    is_neutral = d["is_neutral"].astype(bool)
    block_num = d["block_num"].astype(np.int64)
    timestamp_sec = d["timestamp_sec"].astype(np.float64)
    valid = np.isfinite(x).any(axis=1)
    return x[valid], y[valid], is_neutral[valid], block_num[valid], timestamp_sec[valid]


def baseline_norm_subject(
    x: np.ndarray,
    neutral_mask: np.ndarray,
    baseline_frac: float = 0.5,
) -> np.ndarray:
    neutral_x = x[neutral_mask]
    if len(neutral_x) == 0:
        return x.copy()

    n_base = max(1, int(len(neutral_x) * baseline_frac))
    baseline = neutral_x[:n_base]

    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        feat_min = np.nanmin(baseline, axis=0)
        feat_max = np.nanmax(baseline, axis=0)

    bad_col = ~np.isfinite(feat_min) | ~np.isfinite(feat_max)
    feat_min = np.where(bad_col, 0.0, feat_min)
    feat_max = np.where(bad_col, 1.0, feat_max)

    rng = feat_max - feat_min
    rng[rng < 1e-8] = 1.0

    x_norm = (x - feat_min) / rng
    x_norm = np.clip(x_norm, -0.5, 2.0)
    if bad_col.any():
        x_norm[:, bad_col] = x[:, bad_col]
    return x_norm.astype(np.float32)


def fit_train_medians(train_arrays: list[np.ndarray]) -> np.ndarray:
    xcat = np.concatenate(train_arrays, axis=0)
    import warnings as _w

    with _w.catch_warnings():
        _w.simplefilter("ignore", RuntimeWarning)
        medians = np.nanmedian(xcat, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0)
    return medians.astype(np.float32)


def apply_impute(x: np.ndarray, medians: np.ndarray) -> np.ndarray:
    out = x.copy()
    for col in range(out.shape[1]):
        mask = ~np.isfinite(out[:, col])
        if mask.any():
            out[mask, col] = medians[col]
    return out.astype(np.float32)


def split_train_val_subjects(subjects: list[str], seed: int) -> tuple[list[str], list[str]]:
    if len(subjects) <= 2:
        return subjects, []
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(0.2 * len(shuffled))))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def contiguous_segment_starts(block_num: np.ndarray, timestamp_sec: np.ndarray) -> list[tuple[int, int]]:
    if len(block_num) == 0:
        return []

    starts = [0]
    for i in range(1, len(block_num)):
        if block_num[i] != block_num[i - 1] or timestamp_sec[i] <= timestamp_sec[i - 1]:
            starts.append(i)

    segments: list[tuple[int, int]] = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(block_num)
        segments.append((start, end))
    return segments


def build_sequences(
    x: np.ndarray,
    y: np.ndarray,
    block_num: np.ndarray,
    timestamp_sec: np.ndarray,
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x) < seq_len:
        return np.zeros((0, seq_len, x.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    xs, ys = [], []
    for start, end in contiguous_segment_starts(block_num, timestamp_sec):
        seg_x = x[start:end]
        seg_y = y[start:end]
        if len(seg_x) < seq_len:
            continue
        for t in range(seq_len - 1, len(seg_x)):
            xs.append(seg_x[t - seq_len + 1 : t + 1])
            ys.append(seg_y[t])

    if not xs:
        return np.zeros((0, seq_len, x.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def evaluate_gru(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    decision_threshold: float,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    y_true, y_pred, y_prob, losses = [], [], [], []
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
            y_prob.extend(probs.cpu().numpy().tolist())

    yt = np.asarray(y_true, dtype=np.int64)
    yp = np.asarray(y_pred, dtype=np.int64)
    prob = np.asarray(y_prob, dtype=np.float32)
    per_f1 = f1_score(yt, yp, labels=[0, 1], average=None, zero_division=0)
    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(yt, yp, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(yt, yp, average="macro", zero_division=0)),
        "f1_nostress": float(per_f1[0]) if len(per_f1) > 0 else 0.0,
        "f1_stress": float(per_f1[1]) if len(per_f1) > 1 else 0.0,
    }
    return metrics, yt, yp, prob


def best_threshold_f1(y_true: np.ndarray, probs: np.ndarray, n_steps: int = 49) -> float:
    """Choose a threshold on validation probabilities to maximize positive-class F1."""
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if len(y_true) < 2 or len(np.unique(y_true)) < 2:
        return 0.5
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.02, 0.98, n_steps):
        pred = (probs >= t).astype(np.int64)
        f1v = float(f1_score(y_true, pred, labels=[0, 1], average="binary", pos_label=1, zero_division=0))
        if f1v > best_f1 or (f1v == best_f1 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_f1, best_t = f1v, float(t)
    return best_t


def train_gru_one_loso_fold(
    subj_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    subjects: list[str],
    held_out: str,
    *,
    seq_len: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    rnn_type: str,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    class_weight: str,
    decision_threshold: float,
    decision_threshold_tune: str,
    device: torch.device,
    split_seed: int,
) -> tuple[dict[str, float | int | str], int, dict[str, np.ndarray]]:
    train_ids_all = [sid for sid in subjects if sid != held_out]
    train_ids, val_ids = split_train_val_subjects(train_ids_all, seed=split_seed)

    medians = fit_train_medians([subj_data[s][0] for s in train_ids_all])

    x_te, y_te, _, block_te, t_te = subj_data[held_out]
    x_te = apply_impute(x_te, medians)
    x_te_seq, y_te_seq = build_sequences(x_te, y_te, block_te, t_te, seq_len=seq_len)

    tr_seq_list, tr_y_list = [], []
    for sid in train_ids:
        x, y, _, block_num, timestamp_sec = subj_data[sid]
        x = apply_impute(x, medians)
        sx, sy = build_sequences(x, y, block_num, timestamp_sec, seq_len=seq_len)
        if len(sy) > 0:
            tr_seq_list.append(sx)
            tr_y_list.append(sy)
    if not tr_seq_list:
        raise RuntimeError(f"No training sequences produced for held_out={held_out}")
    x_tr_seq = np.concatenate(tr_seq_list, axis=0)
    y_tr_seq = np.concatenate(tr_y_list, axis=0)

    input_dim = x_tr_seq.shape[2]

    if len(val_ids) > 0:
        va_seq_list, va_y_list = [], []
        for sid in val_ids:
            x, y, _, block_num, timestamp_sec = subj_data[sid]
            x = apply_impute(x, medians)
            sx, sy = build_sequences(x, y, block_num, timestamp_sec, seq_len=seq_len)
            if len(sy) > 0:
                va_seq_list.append(sx)
                va_y_list.append(sy)
        x_va_seq = (
            np.concatenate(va_seq_list, axis=0)
            if va_seq_list
            else np.zeros((0, seq_len, input_dim), dtype=np.float32)
        )
        y_va_seq = (
            np.concatenate(va_y_list, axis=0)
            if va_y_list
            else np.zeros((0,), dtype=np.int64)
        )
    else:
        x_va_seq = np.zeros((0, seq_len, input_dim), dtype=np.float32)
        y_va_seq = np.zeros((0,), dtype=np.int64)

    if len(y_te_seq) == 0:
        raise RuntimeError(f"No test sequences produced for held_out={held_out}")

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_tr_seq), torch.from_numpy(y_tr_seq)),
        batch_size=min(batch_size, len(y_tr_seq)),
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_te_seq), torch.from_numpy(y_te_seq)),
        batch_size=min(batch_size, max(1, len(y_te_seq))),
        shuffle=False,
    )
    val_loader = None
    if len(y_va_seq) > 0:
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(x_va_seq), torch.from_numpy(y_va_seq)),
            batch_size=min(batch_size, len(y_va_seq)),
            shuffle=False,
        )

    model_cls = LSTMClassifier if rnn_type == "lstm" else GRUClassifier
    model = model_cls(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=2,
    ).to(device)

    if class_weight == "balanced":
        n_neg = float(np.sum(y_tr_seq == 0))
        n_pos = float(np.sum(y_tr_seq == 1))
        w0 = 0.5 * (n_neg + n_pos) / max(n_neg, 1.0)
        w1 = 0.5 * (n_neg + n_pos) / max(n_pos, 1.0)
        class_w = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=class_w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_f1 = -1.0
    bad_epochs = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if val_loader is not None:
            val_metrics, _, _, _ = evaluate_gru(model, val_loader, device, decision_threshold)
            val_f1 = val_metrics["macro_f1"]
            if val_f1 > best_val_f1 + 1e-6:
                best_val_f1 = val_f1
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    eval_threshold = float(decision_threshold)
    if decision_threshold_tune == "f1_val" and val_loader is not None:
        _, y_val_true, _, y_val_prob = evaluate_gru(model, val_loader, device, decision_threshold)
        eval_threshold = best_threshold_f1(y_val_true, y_val_prob)

    metrics, y_true, y_pred, y_prob = evaluate_gru(model, test_loader, device, eval_threshold)
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    row: dict[str, float | int | str] = {
        "subject": held_out,
        "n_test": int(len(y_te_seq)),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "decision_threshold_used": float(eval_threshold),
        **{k: v for k, v in metrics.items() if k != "loss"},
    }
    predictions = {
        "sequence_index": np.arange(len(y_true), dtype=np.int64),
        "y_true": y_true,
        "y_pred": y_pred,
        "p_stress": y_prob,
        "decision_threshold_used": np.full(len(y_true), eval_threshold, dtype=np.float32),
    }
    return row, int(len(y_te_seq)), predictions


def load_subjects_data(
    subjects: list[str],
    npz_by_subject: dict[str, Path],
    use_baseline_norm: bool,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    subj_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {
        s: load_subject_npz(npz_by_subject[s]) for s in subjects
    }
    if use_baseline_norm:
        subj_data = {
            s: (baseline_norm_subject(x, is_neutral), y, is_neutral, block_num, timestamp_sec)
            for s, (x, y, is_neutral, block_num, timestamp_sec) in subj_data.items()
        }
    return subj_data
