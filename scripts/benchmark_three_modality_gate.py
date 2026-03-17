#!/usr/bin/env python3
"""Three-modality dynamic gate benchmark: ECG + EDA + BVP (LOSO).

Pipeline per fold:
1) Train ECG GRU on train subjects (CardioMind features).
2) Train EDA GRU on train subjects (strict EDA features).
3) Train BVP CNN on train subjects (raw BVP waveforms).
4) Freeze all three models.
5) Extract stress probabilities from each: p_ecg, p_eda, p_bvp.
6) Fuse with one of these strategies:
    - mean:        simple average of three probabilities
    - weighted:    softmax-weighted average (learned on val set)
    - gate_3way:   learned 3-way attention gate (MLP on entropies + activities)
    - best_two:    use the two most confident modalities per sample
    - majority:    majority vote across three binary predictions

Requires:
    data/processed_cardiomind_strict_ratio/  (ECG)
    data/processed_eda_strict_ratio_aligned_to_ecg/  (EDA)
    data/processed_bvp_raw_aligned_to_ecg/  (BVP)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.bvp_cnn_classifier import BVPCNNClassifier
from src.models.gru_classifier import GRUClassifier


# ---------------------------------------------------------------------------
# 3-way modality attention gate
# ---------------------------------------------------------------------------

class ModalityAttention3(nn.Module):
    """Learned 3-way gate: produces softmax weights over ECG, EDA, BVP."""

    def __init__(self, hidden_dim: int = 16) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(6, hidden_dim),  # [h_c, a_c, h_e, a_e, h_b, a_b]
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        h_c: torch.Tensor, a_c: torch.Tensor,
        h_e: torch.Tensor, a_e: torch.Tensor,
        h_b: torch.Tensor, a_b: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack([h_c, a_c, h_e, a_e, h_b, a_b], dim=1)  # (B, 6)
        return torch.softmax(self.mlp(x), dim=1)  # (B, 3)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_ecg_subject(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = torch.load(path, weights_only=False)
    X = np.asarray(d["cardiac_features"], dtype=np.float32)
    y = np.asarray(d["labels"], dtype=np.int64)
    t = np.asarray(d["timestamps_sec"], dtype=np.float64)
    valid = np.isfinite(X).all(axis=1)
    return X[valid], y[valid], t[valid]


def _load_eda_subject(path: Path, stress_label: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = torch.load(path, weights_only=False)
    X = np.asarray(d["cardiac_features"], dtype=np.float32)
    y_raw = np.asarray(d["labels"], dtype=np.int64)
    y = (y_raw == int(stress_label)).astype(np.int64)
    t = np.asarray(d["timestamps_sec"], dtype=np.float64)
    valid = np.isfinite(X).all(axis=1)
    return X[valid], y[valid], t[valid]


def _load_bvp_subject(path: Path, stress_label: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = torch.load(path, weights_only=False)
    X = np.asarray(d["bvp_windows"], dtype=np.float32)   # (N, 1280)
    y_raw = np.asarray(d["labels"], dtype=np.int64)
    y = (y_raw == int(stress_label)).astype(np.int64)
    t = np.asarray(d["timestamps_sec"], dtype=np.float64)
    valid = np.isfinite(X).all(axis=1)
    return X[valid], y[valid], t[valid]


def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) < seq_len:
        return np.zeros((0, seq_len, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    xs, ys = [], []
    for t in range(seq_len - 1, len(X)):
        xs.append(X[t - seq_len + 1 : t + 1])
        ys.append(y[t])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def _split_train_val(subjects: list[int], seed: int) -> tuple[list[int], list[int]]:
    if len(subjects) <= 1:
        return subjects, []
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(0.2 * len(shuffled))))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def _fit_scaler(subjects: list[tuple[np.ndarray, ...]]) -> tuple[np.ndarray, np.ndarray]:
    Xcat = np.concatenate([s[0] for s in subjects], axis=0)
    mu = Xcat.mean(axis=0)
    sigma = Xcat.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma


def _fit_scaler_1d(subjects: list[tuple[np.ndarray, ...]]) -> tuple[float, float]:
    """Fit global scalar mean/std for raw BVP windows."""
    Xcat = np.concatenate([s[0] for s in subjects], axis=0)
    return float(np.mean(Xcat)), max(float(np.std(Xcat)), 1e-8)


def _apply_scaler(X: np.ndarray, mu, sigma) -> np.ndarray:
    return ((X - mu) / sigma).astype(np.float32)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _train_gru(
    train_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    train_ids: list[int], val_ids: list[int],
    seq_len: int, input_dim: int, hidden_dim: int,
    epochs: int, patience: int, batch_size: int,
    lr: float, class_weight: str, threshold: float,
    device: torch.device, dropout: float = 0.2,
) -> nn.Module:
    tr_x, tr_y = [], []
    for sid in train_ids:
        X, y, _ = train_data[sid]
        sx, sy = _build_sequences(X, y, seq_len)
        if len(sy) > 0:
            tr_x.append(sx); tr_y.append(sy)
    X_tr = np.concatenate(tr_x, axis=0)
    y_tr = np.concatenate(tr_y, axis=0)

    va_x, va_y = [], []
    for sid in val_ids:
        X, y, _ = train_data[sid]
        sx, sy = _build_sequences(X, y, seq_len)
        if len(sy) > 0:
            va_x.append(sx); va_y.append(sy)
    X_va = np.concatenate(va_x) if va_x else np.zeros((0, seq_len, input_dim), dtype=np.float32)
    y_va = np.concatenate(va_y) if va_y else np.zeros((0,), dtype=np.int64)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)), batch_size=batch_size) if len(y_va) > 0 else None

    model = GRUClassifier(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device)

    if class_weight == "balanced":
        n0 = float(np.sum(y_tr == 0)); n1 = float(np.sum(y_tr == 1))
        cw = torch.tensor([0.5*(n0+n1)/max(n0,1), 0.5*(n0+n1)/max(n1,1)], device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state, best_val, bad = None, float("-inf"), 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if val_loader is not None:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    p = torch.softmax(model(xb.to(device)), dim=1)[:, 1]
                    preds.extend((p >= threshold).long().cpu().tolist())
                    trues.extend(yb.tolist())
            f1 = f1_score(trues, preds, zero_division=0)
            if f1 > best_val + 1e-6:
                best_val = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state:
        model.load_state_dict(best_state)
    return model


def _train_bvp_cnn(
    train_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    train_ids: list[int], val_ids: list[int],
    input_len: int, gru_hidden: int,
    epochs: int, patience: int, batch_size: int,
    lr: float, class_weight: str, threshold: float,
    device: torch.device, dropout: float = 0.3,
) -> nn.Module:
    """Train BVPCNNClassifier (no sequencing — each window is independent)."""
    X_tr = np.concatenate([train_data[sid][0] for sid in train_ids], axis=0)
    y_tr = np.concatenate([train_data[sid][1] for sid in train_ids], axis=0)
    if val_ids:
        X_va = np.concatenate([train_data[sid][0] for sid in val_ids], axis=0)
        y_va = np.concatenate([train_data[sid][1] for sid in val_ids], axis=0)
    else:
        X_va, y_va = np.empty((0, input_len), np.float32), np.empty((0,), np.int64)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)), batch_size=batch_size) if len(y_va) > 0 else None

    model = BVPCNNClassifier(input_len=input_len, gru_hidden=gru_hidden, dropout=dropout).to(device)

    if class_weight == "balanced":
        n0 = float(np.sum(y_tr == 0)); n1 = float(np.sum(y_tr == 1))
        cw = torch.tensor([0.5*(n0+n1)/max(n0,1), 0.5*(n0+n1)/max(n1,1)], device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    best_state, best_val, bad = None, float("-inf"), 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        if val_loader is not None:
            model.eval()
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    p = torch.softmax(model(xb.to(device)), dim=1)[:, 1]
                    preds.extend((p >= threshold).long().cpu().tolist())
                    trues.extend(yb.tolist())
            f1 = f1_score(trues, preds, zero_division=0)
            if f1 > best_val + 1e-6:
                best_val = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Probability extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _extract_gru_probs(model: GRUClassifier, X_seq: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probs, entropies, activity_norms) for sequence model."""
    loader = DataLoader(TensorDataset(torch.from_numpy(X_seq)), batch_size=batch_size, shuffle=False)
    probs, ents, acts = [], [], []
    for (xb,) in loader:
        xb = xb.to(device)
        _, p_seq, _ = model.forward_with_state(xb)
        p = p_seq[:, -1]
        eps = 1e-8
        ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
        act = torch.linalg.vector_norm(xb, ord=2, dim=(1, 2))
        probs.append(p.cpu().numpy())
        ents.append(ent.cpu().numpy())
        acts.append(act.cpu().numpy())
    return np.concatenate(probs).astype(np.float32), np.concatenate(ents).astype(np.float32), np.concatenate(acts).astype(np.float32)


@torch.no_grad()
def _extract_bvp_probs(model: BVPCNNClassifier, X_bvp: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (probs, entropies, activity_norms) for BVP CNN model."""
    loader = DataLoader(TensorDataset(torch.from_numpy(X_bvp)), batch_size=batch_size, shuffle=False)
    probs, ents, acts = [], [], []
    for (xb,) in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.softmax(logits, dim=1)[:, 1]
        eps = 1e-8
        ent = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
        act = torch.linalg.vector_norm(xb, ord=2, dim=1)
        probs.append(p.cpu().numpy())
        ents.append(ent.cpu().numpy())
        acts.append(act.cpu().numpy())
    return np.concatenate(probs).astype(np.float32), np.concatenate(ents).astype(np.float32), np.concatenate(acts).astype(np.float32)


# ---------------------------------------------------------------------------
# Fusion strategies
# ---------------------------------------------------------------------------

def _fuse_mean(p_c: np.ndarray, p_e: np.ndarray, p_b: np.ndarray, threshold: float) -> np.ndarray:
    p = (p_c + p_e + p_b) / 3.0
    return (p >= threshold).astype(np.int64)


def _fuse_majority(p_c: np.ndarray, p_e: np.ndarray, p_b: np.ndarray,
                   th_c: float, th_e: float, th_b: float) -> np.ndarray:
    votes = ((p_c >= th_c).astype(int) + (p_e >= th_e).astype(int) + (p_b >= th_b).astype(int))
    return (votes >= 2).astype(np.int64)


def _fuse_best_two(p_c: np.ndarray, p_e: np.ndarray, p_b: np.ndarray, threshold: float) -> np.ndarray:
    """Average the two most confident modalities (furthest from 0.5) per sample."""
    conf_c = np.abs(p_c - 0.5)
    conf_e = np.abs(p_e - 0.5)
    conf_b = np.abs(p_b - 0.5)
    stack = np.stack([conf_c, conf_e, conf_b], axis=1)
    probs = np.stack([p_c, p_e, p_b], axis=1)
    worst_idx = np.argmin(stack, axis=1)
    mask = np.ones_like(probs)
    mask[np.arange(len(mask)), worst_idx] = 0.0
    p_fused = (probs * mask).sum(axis=1) / mask.sum(axis=1)
    return (p_fused >= threshold).astype(np.int64)


def _train_and_apply_gate_3way(
    p_c_tr: np.ndarray, h_c_tr: np.ndarray, a_c_tr: np.ndarray,
    p_e_tr: np.ndarray, h_e_tr: np.ndarray, a_e_tr: np.ndarray,
    p_b_tr: np.ndarray, h_b_tr: np.ndarray, a_b_tr: np.ndarray,
    y_tr: np.ndarray,
    p_c_va: np.ndarray, h_c_va: np.ndarray, a_c_va: np.ndarray,
    p_e_va: np.ndarray, h_e_va: np.ndarray, a_e_va: np.ndarray,
    p_b_va: np.ndarray, h_b_va: np.ndarray, a_b_va: np.ndarray,
    y_va: np.ndarray,
    p_c_te: np.ndarray, h_c_te: np.ndarray, a_c_te: np.ndarray,
    p_e_te: np.ndarray, h_e_te: np.ndarray, a_e_te: np.ndarray,
    p_b_te: np.ndarray, h_b_te: np.ndarray, a_b_te: np.ndarray,
    epochs: int, patience: int, batch_size: int, lr: float,
    device: torch.device, threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Train 3-way gating MLP and return (fused_probs, predictions) on test set."""
    gate = ModalityAttention3(hidden_dim=16).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)

    tr_loader = DataLoader(TensorDataset(
        torch.from_numpy(h_c_tr), torch.from_numpy(a_c_tr), torch.from_numpy(p_c_tr),
        torch.from_numpy(h_e_tr), torch.from_numpy(a_e_tr), torch.from_numpy(p_e_tr),
        torch.from_numpy(h_b_tr), torch.from_numpy(a_b_tr), torch.from_numpy(p_b_tr),
        torch.from_numpy(y_tr.astype(np.float32)),
    ), batch_size=batch_size, shuffle=True)

    best_state, best_val, bad = None, float("-inf"), 0
    for _ in range(epochs):
        gate.train()
        for hc, ac, pc, he, ae, pe, hb, ab, pb, yt in tr_loader:
            hc,ac,pc = hc.to(device), ac.to(device), pc.to(device)
            he,ae,pe = he.to(device), ae.to(device), pe.to(device)
            hb,ab,pb = hb.to(device), ab.to(device), pb.to(device)
            yt = yt.to(device)
            alpha = gate(hc, ac, he, ae, hb, ab)  # (B, 3)
            p_fused = alpha[:, 0] * pc + alpha[:, 1] * pe + alpha[:, 2] * pb
            p_fused = torch.clamp(p_fused, 1e-6, 1.0 - 1e-6)
            loss = nn.functional.binary_cross_entropy(p_fused, yt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            opt.step()

        if len(y_va) > 0:
            gate.eval()
            with torch.no_grad():
                alpha = gate(
                    torch.from_numpy(h_c_va).to(device), torch.from_numpy(a_c_va).to(device),
                    torch.from_numpy(h_e_va).to(device), torch.from_numpy(a_e_va).to(device),
                    torch.from_numpy(h_b_va).to(device), torch.from_numpy(a_b_va).to(device),
                )
                p_f = alpha[:, 0] * torch.from_numpy(p_c_va).to(device) + \
                      alpha[:, 1] * torch.from_numpy(p_e_va).to(device) + \
                      alpha[:, 2] * torch.from_numpy(p_b_va).to(device)
                pred = (p_f >= threshold).long().cpu().numpy()
                f1 = f1_score(y_va, pred, zero_division=0)
            if f1 > best_val + 1e-6:
                best_val = f1
                best_state = {k: v.cpu().clone() for k, v in gate.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state:
        gate.load_state_dict(best_state)
    gate.eval()
    with torch.no_grad():
        alpha = gate(
            torch.from_numpy(h_c_te).to(device), torch.from_numpy(a_c_te).to(device),
            torch.from_numpy(h_e_te).to(device), torch.from_numpy(a_e_te).to(device),
            torch.from_numpy(h_b_te).to(device), torch.from_numpy(a_b_te).to(device),
        )
        p_fused = (alpha[:, 0] * torch.from_numpy(p_c_te).to(device) +
                   alpha[:, 1] * torch.from_numpy(p_e_te).to(device) +
                   alpha[:, 2] * torch.from_numpy(p_b_te).to(device))
        p_fused_np = p_fused.cpu().numpy().astype(np.float32)
    return p_fused_np, (p_fused_np >= threshold).astype(np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Three-modality ECG+EDA+BVP dynamic gate benchmark")

    # Data roots
    parser.add_argument("--ecg-root", default="data/processed_cardiomind_strict_ratio")
    parser.add_argument("--eda-root", default="data/processed_eda_strict_ratio_aligned_to_ecg")
    parser.add_argument("--bvp-root", default="data/processed_bvp_raw_aligned_to_ecg")

    # Fusion
    parser.add_argument("--fusion-strategy", default="mean",
                        choices=["mean", "weighted", "gate_3way", "best_two", "majority"])

    # Thresholds
    parser.add_argument("--decision-threshold-ecg", type=float, default=0.4)
    parser.add_argument("--decision-threshold-eda", type=float, default=0.5)
    parser.add_argument("--decision-threshold-bvp", type=float, default=0.5)
    parser.add_argument("--decision-threshold-fused", type=float, default=0.5)

    # GRU settings (ECG + EDA)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--hidden-dim-ecg", type=int, default=64)
    parser.add_argument("--hidden-dim-eda", type=int, default=32)
    parser.add_argument("--epochs-gru", type=int, default=60)

    # BVP CNN settings
    parser.add_argument("--gru-hidden-bvp", type=int, default=64)
    parser.add_argument("--epochs-bvp", type=int, default=60)

    # Common
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-gru", type=float, default=1e-3)
    parser.add_argument("--lr-bvp", type=float, default=1e-3)
    parser.add_argument("--lr-gate", type=float, default=1e-3)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--out-csv", default="runs/three_modality_gate.csv")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    args = parser.parse_args()

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

    # Load data
    ecg_root, eda_root, bvp_root = Path(args.ecg_root), Path(args.eda_root), Path(args.bvp_root)
    ecg_files = {p.stem: p for p in ecg_root.glob("S*.pt")}
    eda_files = {p.stem: p for p in eda_root.glob("S*.pt")}
    bvp_files = {p.stem: p for p in bvp_root.glob("S*.pt")}
    subjects = sorted(
        set(ecg_files.keys()) & set(eda_files.keys()) & set(bvp_files.keys()),
        key=lambda s: int(s[1:]),
    )
    if not subjects:
        raise FileNotFoundError("No overlapping subjects across ECG, EDA, and BVP roots")

    subject_ids = [int(s[1:]) for s in subjects]
    if args.max_subjects:
        subject_ids = subject_ids[: args.max_subjects]
        subjects = [f"S{sid}" for sid in subject_ids]

    print(f"Three-modality gate | strategy={args.fusion_strategy} | subjects={subject_ids} | device={device}")

    ecg_data = {int(s[1:]): _load_ecg_subject(ecg_files[s]) for s in subjects}
    eda_data = {int(s[1:]): _load_eda_subject(eda_files[s]) for s in subjects}
    bvp_data = {int(s[1:]): _load_bvp_subject(bvp_files[s]) for s in subjects}

    bvp_input_len = bvp_data[subject_ids[0]][0].shape[1]
    print(f"  BVP input_len={bvp_input_len}")

    rows = []
    for held_out in subject_ids:
        _seed_all(args.seed)
        train_all = [sid for sid in subject_ids if sid != held_out]
        train_ids, val_ids = _split_train_val(train_all, seed=args.seed)
        if not train_ids:
            train_ids = train_all.copy()

        # Scale ECG + EDA features
        mu_c, sg_c = _fit_scaler([ecg_data[sid] for sid in train_ids])
        mu_e, sg_e = _fit_scaler([eda_data[sid] for sid in train_ids])
        mu_b, sg_b = _fit_scaler_1d([bvp_data[sid] for sid in train_ids])

        ecg_scaled = {sid: (_apply_scaler(ecg_data[sid][0], mu_c, sg_c), ecg_data[sid][1], ecg_data[sid][2]) for sid in subject_ids}
        eda_scaled = {sid: (_apply_scaler(eda_data[sid][0], mu_e, sg_e), eda_data[sid][1], eda_data[sid][2]) for sid in subject_ids}
        bvp_scaled = {sid: (_apply_scaler(bvp_data[sid][0], mu_b, sg_b), bvp_data[sid][1], bvp_data[sid][2]) for sid in subject_ids}

        # Train ECG GRU
        ecg_dim = ecg_scaled[held_out][0].shape[1]
        ecg_model = _train_gru(ecg_scaled, train_ids, val_ids, args.seq_len, ecg_dim,
                               args.hidden_dim_ecg, args.epochs_gru, args.patience,
                               args.batch_size, args.lr_gru, args.class_weight,
                               args.decision_threshold_ecg, device, args.dropout)

        # Train EDA GRU
        eda_dim = eda_scaled[held_out][0].shape[1]
        eda_model = _train_gru(eda_scaled, train_ids, val_ids, args.seq_len, eda_dim,
                               args.hidden_dim_eda, args.epochs_gru, args.patience,
                               args.batch_size, args.lr_gru, args.class_weight,
                               args.decision_threshold_eda, device, args.dropout)

        # Train BVP CNN
        bvp_model = _train_bvp_cnn(bvp_scaled, train_ids, val_ids, bvp_input_len,
                                   args.gru_hidden_bvp, args.epochs_bvp, args.patience,
                                   args.batch_size, args.lr_bvp, args.class_weight,
                                   args.decision_threshold_bvp, device)

        # Freeze all
        for m in [ecg_model, eda_model, bvp_model]:
            m.eval()
            for p in m.parameters():
                p.requires_grad = False

        # Build test sequences for ECG/EDA
        def _build_pair_seqs(ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            xs_c, xs_e, xs_b, ys = [], [], [], []
            for sid in ids:
                Xc, yc, _ = ecg_scaled[sid]
                Xe, ye, _ = eda_scaled[sid]
                Xb, yb, _ = bvp_scaled[sid]
                sc, syc = _build_sequences(Xc, yc, args.seq_len)
                se, sye = _build_sequences(Xe, ye, args.seq_len)
                if len(syc) == 0:
                    continue
                n = min(len(syc), len(sye), len(yb) - args.seq_len + 1)
                if n <= 0:
                    continue
                xs_c.append(sc[:n])
                xs_e.append(se[:n])
                xs_b.append(Xb[args.seq_len - 1:args.seq_len - 1 + n])
                ys.append(syc[:n])
            if not xs_c:
                return (np.zeros((0, args.seq_len, 1), np.float32),
                        np.zeros((0, args.seq_len, 1), np.float32),
                        np.zeros((0, bvp_input_len), np.float32),
                        np.zeros((0,), np.int64))
            return np.concatenate(xs_c), np.concatenate(xs_e), np.concatenate(xs_b), np.concatenate(ys)

        Xc_tr, Xe_tr, Xb_tr, y_tr = _build_pair_seqs(train_ids)
        Xc_va, Xe_va, Xb_va, y_va = _build_pair_seqs(val_ids)
        Xc_te, Xe_te, Xb_te, y_te = _build_pair_seqs([held_out])

        if len(y_te) == 0:
            print(f"  S{held_out}: SKIP (no test samples)")
            continue

        # Extract probabilities
        p_c_tr, h_c_tr, a_c_tr = _extract_gru_probs(ecg_model, Xc_tr, args.batch_size, device)
        p_e_tr, h_e_tr, a_e_tr = _extract_gru_probs(eda_model, Xe_tr, args.batch_size, device)
        p_b_tr, h_b_tr, a_b_tr = _extract_bvp_probs(bvp_model, Xb_tr, args.batch_size, device)

        if len(y_va) > 0:
            p_c_va, h_c_va, a_c_va = _extract_gru_probs(ecg_model, Xc_va, args.batch_size, device)
            p_e_va, h_e_va, a_e_va = _extract_gru_probs(eda_model, Xe_va, args.batch_size, device)
            p_b_va, h_b_va, a_b_va = _extract_bvp_probs(bvp_model, Xb_va, args.batch_size, device)
        else:
            z0 = np.zeros((0,), np.float32)
            p_c_va, h_c_va, a_c_va = z0, z0, z0
            p_e_va, h_e_va, a_e_va = z0, z0, z0
            p_b_va, h_b_va, a_b_va = z0, z0, z0

        p_c_te, h_c_te, a_c_te = _extract_gru_probs(ecg_model, Xc_te, args.batch_size, device)
        p_e_te, h_e_te, a_e_te = _extract_gru_probs(eda_model, Xe_te, args.batch_size, device)
        p_b_te, h_b_te, a_b_te = _extract_bvp_probs(bvp_model, Xb_te, args.batch_size, device)

        # Individual modality predictions
        pred_ecg = (p_c_te >= args.decision_threshold_ecg).astype(np.int64)
        pred_eda = (p_e_te >= args.decision_threshold_eda).astype(np.int64)
        pred_bvp = (p_b_te >= args.decision_threshold_bvp).astype(np.int64)

        # Fused predictions
        if args.fusion_strategy == "mean":
            pred_fused = _fuse_mean(p_c_te, p_e_te, p_b_te, args.decision_threshold_fused)
        elif args.fusion_strategy == "majority":
            pred_fused = _fuse_majority(p_c_te, p_e_te, p_b_te,
                                        args.decision_threshold_ecg, args.decision_threshold_eda,
                                        args.decision_threshold_bvp)
        elif args.fusion_strategy == "best_two":
            pred_fused = _fuse_best_two(p_c_te, p_e_te, p_b_te, args.decision_threshold_fused)
        elif args.fusion_strategy == "gate_3way":
            _, pred_fused = _train_and_apply_gate_3way(
                p_c_tr, h_c_tr, a_c_tr, p_e_tr, h_e_tr, a_e_tr, p_b_tr, h_b_tr, a_b_tr, y_tr,
                p_c_va, h_c_va, a_c_va, p_e_va, h_e_va, a_e_va, p_b_va, h_b_va, a_b_va, y_va,
                p_c_te, h_c_te, a_c_te, p_e_te, h_e_te, a_e_te, p_b_te, h_b_te, a_b_te,
                epochs=30, patience=8, batch_size=args.batch_size, lr=args.lr_gate,
                device=device, threshold=args.decision_threshold_fused,
            )
        else:
            # "weighted" — use validation to find best static weights
            best_f1, best_w = 0.0, (1/3, 1/3, 1/3)
            for wc in np.arange(0, 1.05, 0.1):
                for we in np.arange(0, 1.05 - wc, 0.1):
                    wb = 1.0 - wc - we
                    if wb < -0.01:
                        continue
                    p = wc * p_c_va + we * p_e_va + wb * p_b_va
                    pred = (p >= args.decision_threshold_fused).astype(np.int64)
                    f = f1_score(y_va, pred, zero_division=0) if len(y_va) > 0 else 0.0
                    if f > best_f1:
                        best_f1 = f
                        best_w = (wc, we, wb)
            wc, we, wb = best_w
            p_fused = wc * p_c_te + we * p_e_te + wb * p_b_te
            pred_fused = (p_fused >= args.decision_threshold_fused).astype(np.int64)

        # Compute metrics
        m_ecg = _metrics(y_te, pred_ecg)
        m_eda = _metrics(y_te, pred_eda)
        m_bvp = _metrics(y_te, pred_bvp)
        m_fused = _metrics(y_te, pred_fused)

        row = {
            "subject": held_out,
            "ecg_f1": m_ecg["f1"], "ecg_acc": m_ecg["accuracy"],
            "eda_f1": m_eda["f1"], "eda_acc": m_eda["accuracy"],
            "bvp_f1": m_bvp["f1"], "bvp_acc": m_bvp["accuracy"],
            "fused_f1": m_fused["f1"], "fused_acc": m_fused["accuracy"],
            "fused_prec": m_fused["precision"], "fused_rec": m_fused["recall"],
            "n_samples": int(len(y_te)),
        }
        rows.append(row)

        print(
            f"  S{held_out}: ECG_f1={m_ecg['f1']:.3f} EDA_f1={m_eda['f1']:.3f} "
            f"BVP_f1={m_bvp['f1']:.3f} FUSED_f1={m_fused['f1']:.3f} (n={len(y_te)})"
        )

    if not rows:
        print("No results.")
        return

    # Write CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        summary = {"subject": "MEAN±STD"}
        for col in ["ecg_f1", "eda_f1", "bvp_f1", "fused_f1", "ecg_acc", "eda_acc", "bvp_acc", "fused_acc"]:
            vals = np.array([r[col] for r in rows], dtype=np.float64)
            summary[col] = f"{vals.mean():.6f}±{vals.std():.6f}"
        writer.writerow(summary)

    fused_f1 = np.array([r["fused_f1"] for r in rows], dtype=np.float64)
    ecg_f1 = np.array([r["ecg_f1"] for r in rows], dtype=np.float64)
    eda_f1 = np.array([r["eda_f1"] for r in rows], dtype=np.float64)
    bvp_f1 = np.array([r["bvp_f1"] for r in rows], dtype=np.float64)

    print(f"\n{'='*60}")
    print(f"mean ECG  F1: {ecg_f1.mean():.3f} ± {ecg_f1.std():.3f}")
    print(f"mean EDA  F1: {eda_f1.mean():.3f} ± {eda_f1.std():.3f}")
    print(f"mean BVP  F1: {bvp_f1.mean():.3f} ± {bvp_f1.std():.3f}")
    print(f"mean FUSED F1: {fused_f1.mean():.3f} ± {fused_f1.std():.3f}")
    print(f"strategy: {args.fusion_strategy}")
    print(f"saved_csv: {out_csv}")


if __name__ == "__main__":
    main()
