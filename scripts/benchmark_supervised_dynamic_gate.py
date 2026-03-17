#!/usr/bin/env python3
"""Supervised dynamic modality gating benchmark (LOSO).

Pipeline per fold:
1) Train ECG GRU on train subjects.
2) Train EDA GRU on train subjects.
3) Freeze both GRUs.
4) Train uncertainty-based gate (entropy + activity) to fuse modality probabilities.
5) Evaluate ECG-only, EDA-only, and gated fusion on held-out subject.

Requires aligned EDA windows to ECG timeline.
"""

from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.models.gru_classifier import GRUClassifier


class ModalityAttention(nn.Module):
    def __init__(self, hidden_dim: int = 16):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, h_c: torch.Tensor, a_c: torch.Tensor, h_e: torch.Tensor, a_e: torch.Tensor) -> torch.Tensor:
        x = torch.stack([h_c, a_c, h_e, a_e], dim=1)  # (B,4)
        logits = self.mlp(x)
        return torch.softmax(logits, dim=1)  # (B,2)


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


def _build_sequences(X: np.ndarray, y: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    if len(X) < seq_len:
        return np.zeros((0, seq_len, X.shape[1]), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    xs, ys = [], []
    for t in range(seq_len - 1, len(X)):
        xs.append(X[t - seq_len + 1 : t + 1])
        ys.append(y[t])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64)


def _split_train_val_subjects(subjects: list[int], seed: int) -> tuple[list[int], list[int]]:
    if len(subjects) <= 1:
        return subjects, []
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(0.2 * len(shuffled))))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def _fit_scaler(train_subjects: list[tuple[np.ndarray, np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    Xcat = np.concatenate([x for x, _, _ in train_subjects], axis=0)
    mu = Xcat.mean(axis=0)
    sigma = Xcat.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma


def _apply_scaler(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return (X - mu) / sigma


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def _f1_from_probs(y_true: np.ndarray, p: np.ndarray, threshold: float) -> float:
    if len(y_true) == 0:
        return 0.0
    pred = (p >= threshold).astype(np.int64)
    return float(f1_score(y_true, pred, zero_division=0))


def _evaluate_gru(model: nn.Module, loader: DataLoader, device: torch.device, threshold: float) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = (probs >= threshold).long().cpu().numpy()
            y_pred.extend(pred.tolist())
            y_true.extend(yb.numpy().tolist())
    y_true_np = np.asarray(y_true, dtype=np.int64)
    y_pred_np = np.asarray(y_pred, dtype=np.int64)
    return float(np.mean(y_pred_np == y_true_np)), y_true_np, y_pred_np


def _train_gru(
    train_subjects: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    train_ids: list[int],
    val_ids: list[int],
    seq_len: int,
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    class_weight: str,
    threshold: float,
    device: torch.device,
) -> nn.Module:
    # Build train seq
    tr_x, tr_y = [], []
    for sid in train_ids:
        X, y, _ = train_subjects[sid]
        sx, sy = _build_sequences(X, y, seq_len)
        if len(sy) > 0:
            tr_x.append(sx)
            tr_y.append(sy)
    X_tr = np.concatenate(tr_x, axis=0)
    y_tr = np.concatenate(tr_y, axis=0)

    # Build val seq
    va_x, va_y = [], []
    for sid in val_ids:
        X, y, _ = train_subjects[sid]
        sx, sy = _build_sequences(X, y, seq_len)
        if len(sy) > 0:
            va_x.append(sx)
            va_y.append(sy)
    if va_x:
        X_va = np.concatenate(va_x, axis=0)
        y_va = np.concatenate(va_y, axis=0)
    else:
        X_va = np.zeros((0, seq_len, input_dim), dtype=np.float32)
        y_va = np.zeros((0,), dtype=np.int64)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)), batch_size=batch_size, shuffle=False) if len(y_va) > 0 else None

    model = GRUClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        num_classes=2,
    ).to(device)

    if class_weight == "balanced":
        n_neg = float(np.sum(y_tr == 0))
        n_pos = float(np.sum(y_tr == 1))
        w0 = 0.5 * (n_neg + n_pos) / max(n_neg, 1.0)
        w1 = 0.5 * (n_neg + n_pos) / max(n_pos, 1.0)
        cw = torch.tensor([w0, w1], dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=cw)
    else:
        criterion = nn.CrossEntropyLoss()

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = float("-inf")
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        if val_loader is not None:
            _, yv_true, yv_pred = _evaluate_gru(model, val_loader, device, threshold=threshold)
            val_f1 = f1_score(yv_true, yv_pred, zero_division=0)
            if val_f1 > best_val + 1e-6:
                best_val = float(val_f1)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def _extract_modality_outputs(model: GRUClassifier, X_seq: np.ndarray, batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(TensorDataset(torch.from_numpy(X_seq)), batch_size=batch_size, shuffle=False)
    probs_all, ent_all, act_all = [], [], []

    for (xb,) in loader:
        xb = xb.to(device)
        logits_seq, p_seq, _ = model.forward_with_state(xb)  # p_seq: (B,T)
        p = p_seq[:, -1]  # last-step prob
        eps = 1e-8
        ent = -(p * torch.log(p + eps) + (1.0 - p) * torch.log(1.0 - p + eps))
        act = torch.linalg.vector_norm(xb, ord=2, dim=(1, 2))

        probs_all.append(p.cpu().numpy())
        ent_all.append(ent.cpu().numpy())
        act_all.append(act.cpu().numpy())

    return (
        np.concatenate(probs_all, axis=0).astype(np.float32),
        np.concatenate(ent_all, axis=0).astype(np.float32),
        np.concatenate(act_all, axis=0).astype(np.float32),
    )


def _train_gate(
    p_c_tr: np.ndarray,
    h_c_tr: np.ndarray,
    a_c_tr: np.ndarray,
    p_e_tr: np.ndarray,
    h_e_tr: np.ndarray,
    a_e_tr: np.ndarray,
    y_tr: np.ndarray,
    p_c_va: np.ndarray,
    h_c_va: np.ndarray,
    a_c_va: np.ndarray,
    p_e_va: np.ndarray,
    h_e_va: np.ndarray,
    a_e_va: np.ndarray,
    y_va: np.ndarray,
    epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> ModalityAttention:
    gate = ModalityAttention(hidden_dim=16).to(device)
    opt = torch.optim.Adam(gate.parameters(), lr=lr)

    tr_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(h_c_tr),
            torch.from_numpy(a_c_tr),
            torch.from_numpy(p_c_tr),
            torch.from_numpy(h_e_tr),
            torch.from_numpy(a_e_tr),
            torch.from_numpy(p_e_tr),
            torch.from_numpy(y_tr.astype(np.float32)),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    best_state = None
    best_val = float("-inf")
    bad = 0

    for _ in range(epochs):
        gate.train()
        for hct, act, pct, het, aet, pet, yt in tr_loader:
            hct = hct.to(device)
            act = act.to(device)
            pct = pct.to(device)
            het = het.to(device)
            aet = aet.to(device)
            pet = pet.to(device)
            yt = yt.to(device)

            alpha = gate(hct, act, het, aet)
            p_fused = alpha[:, 0] * pct + alpha[:, 1] * pet
            p_fused = torch.clamp(p_fused, 1e-6, 1.0 - 1e-6)
            loss = nn.functional.binary_cross_entropy(p_fused, yt)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(gate.parameters(), 1.0)
            opt.step()

        # val f1
        if len(y_va) > 0:
            gate.eval()
            with torch.no_grad():
                hct = torch.from_numpy(h_c_va).to(device)
                act = torch.from_numpy(a_c_va).to(device)
                pct = torch.from_numpy(p_c_va).to(device)
                het = torch.from_numpy(h_e_va).to(device)
                aet = torch.from_numpy(a_e_va).to(device)
                pet = torch.from_numpy(p_e_va).to(device)
                alpha = gate(hct, act, het, aet)
                p_fused = alpha[:, 0] * pct + alpha[:, 1] * pet
                pred = (p_fused >= 0.5).long().cpu().numpy()
                f1 = f1_score(y_va, pred, zero_division=0)

            if f1 > best_val + 1e-6:
                best_val = float(f1)
                best_state = {k: v.detach().cpu().clone() for k, v in gate.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

    if best_state is not None:
        gate.load_state_dict(best_state)

    return gate


def _parse_float_grid(spec: str | None, default_value: float) -> list[float]:
    if spec is None or spec.strip() == "":
        return [float(default_value)]
    vals = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        vals.append(float(token))
    if not vals:
        return [float(default_value)]
    return sorted(set(vals))


def _binary_entropy_np(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(np.float64), 1e-8, 1.0 - 1e-8)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)).astype(np.float64)


def _sequence_snr_proxy(X_seq: np.ndarray) -> np.ndarray:
    if len(X_seq) == 0:
        return np.zeros((0,), dtype=np.float64)
    signal_power = np.mean(np.square(X_seq), axis=(1, 2))
    diff = np.diff(X_seq, axis=1)
    noise_power = np.mean(np.square(diff), axis=(1, 2)) + 1e-8
    return (signal_power / noise_power).astype(np.float64)


def _sequence_variance_proxy(X_seq: np.ndarray) -> np.ndarray:
    if len(X_seq) == 0:
        return np.zeros((0,), dtype=np.float64)
    return np.var(X_seq, axis=(1, 2)).astype(np.float64)


def _sequence_relative_variation_proxy(X_seq: np.ndarray, eps: float = 1e-3, clip_max: float = 50.0) -> np.ndarray:
    if len(X_seq) == 0:
        return np.zeros((0,), dtype=np.float64)
    mu = np.mean(X_seq, axis=1)
    sd = np.std(X_seq, axis=1)
    rel = sd / (np.abs(mu) + float(eps))
    rel = np.clip(rel, 0.0, float(clip_max))
    return np.mean(rel, axis=1).astype(np.float64)


def _zscore_with_train_stats(x: np.ndarray, mu: float, sd: float) -> np.ndarray:
    sd = max(float(sd), 1e-8)
    return ((x.astype(np.float64) - float(mu)) / sd).astype(np.float64)


def _proxy_fuse_probs(
    p_c: np.ndarray,
    p_e: np.ndarray,
    snr_c_z: np.ndarray,
    snr_e_z: np.ndarray,
    var_c_z: np.ndarray,
    var_e_z: np.ndarray,
    w_quality: float,
    w_margin: float,
    w_entropy: float,
    w_snr: float,
    w_var: float,
    quality_hard_gap: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_c = p_c.astype(np.float64)
    p_e = p_e.astype(np.float64)

    margin_c = np.abs(p_c - 0.5)
    margin_e = np.abs(p_e - 0.5)
    ent_c = _binary_entropy_np(p_c)
    ent_e = _binary_entropy_np(p_e)

    quality_c = w_snr * snr_c_z - w_var * var_c_z
    quality_e = w_snr * snr_e_z - w_var * var_e_z

    score_c = w_quality * quality_c + w_margin * margin_c - w_entropy * ent_c
    score_e = w_quality * quality_e + w_margin * margin_e - w_entropy * ent_e

    quality_delta = quality_c - quality_e
    if quality_hard_gap > 0:
        force_c = quality_delta >= quality_hard_gap
        force_e = quality_delta <= -quality_hard_gap
        score_c = np.where(force_c, 40.0, score_c)
        score_e = np.where(force_c, -40.0, score_e)
        score_c = np.where(force_e, -40.0, score_c)
        score_e = np.where(force_e, 40.0, score_e)

    m = np.maximum(score_c, score_e)
    exp_c = np.exp(score_c - m)
    exp_e = np.exp(score_e - m)
    denom = exp_c + exp_e + 1e-12
    alpha_c = exp_c / denom
    alpha_e = exp_e / denom

    p_fused = alpha_c * p_c + alpha_e * p_e
    return p_fused.astype(np.float32), alpha_c.astype(np.float32), alpha_e.astype(np.float32)


def _proxy_hard_select_probs(
    p_c: np.ndarray,
    p_e: np.ndarray,
    snr_c_z: np.ndarray,
    snr_e_z: np.ndarray,
    var_c_z: np.ndarray,
    var_e_z: np.ndarray,
    w_snr: float,
    w_var: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_c = p_c.astype(np.float64)
    p_e = p_e.astype(np.float64)
    quality_c = w_snr * snr_c_z - w_var * var_c_z
    quality_e = w_snr * snr_e_z - w_var * var_e_z
    choose_c = quality_c >= quality_e
    alpha_c = choose_c.astype(np.float32)
    alpha_e = (~choose_c).astype(np.float32)
    p_sel = np.where(choose_c, p_c, p_e)
    return p_sel.astype(np.float32), alpha_c, alpha_e


def _gate_proxy_override_predictions(
    p_gate: np.ndarray,
    p_c: np.ndarray,
    p_e: np.ndarray,
    alpha_proxy_hard_c: np.ndarray,
    threshold_gate: float,
    threshold_ecg: float,
    threshold_eda: float,
    confidence_margin_threshold: float,
    confidence_entropy_threshold: float,
    low_conf_force_modality: str = "proxy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p_gate = p_gate.astype(np.float64)
    p_c = p_c.astype(np.float64)
    p_e = p_e.astype(np.float64)
    alpha_proxy_hard_c = alpha_proxy_hard_c.astype(np.float64)

    gate_entropy = _binary_entropy_np(p_gate)
    low_conf = (np.abs(p_gate - 0.5) < float(confidence_margin_threshold)) | (gate_entropy > float(confidence_entropy_threshold))

    pred_gate = (p_gate >= float(threshold_gate))
    if low_conf_force_modality == "ecg":
        pred_proxy = p_c >= float(threshold_ecg)
    elif low_conf_force_modality == "eda":
        pred_proxy = p_e >= float(threshold_eda)
    else:
        pred_proxy = np.where(alpha_proxy_hard_c >= 0.5, p_c >= float(threshold_ecg), p_e >= float(threshold_eda))
    pred = np.where(low_conf, pred_proxy, pred_gate).astype(np.int64)

    if low_conf_force_modality == "ecg":
        p_low = p_c
    elif low_conf_force_modality == "eda":
        p_low = p_e
    else:
        p_low = np.where(alpha_proxy_hard_c >= 0.5, p_c, p_e)
    p_eff = np.where(low_conf, p_low, p_gate).astype(np.float32)
    return pred, p_eff, low_conf.astype(np.int64)


def _gate_proxy_hybrid_relvar_predictions(
    p_gate: np.ndarray,
    p_c: np.ndarray,
    p_e: np.ndarray,
    alpha_proxy_hard_c: np.ndarray,
    relvar_delta: np.ndarray,
    threshold_gate: float,
    threshold_ecg: float,
    threshold_eda: float,
    confidence_margin_threshold: float,
    confidence_entropy_threshold: float,
    confident_neg_threshold: float,
    relvar_gap_threshold: float,
    low_conf_force_modality: str = "proxy",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pred_base, p_eff_base, low_conf = _gate_proxy_override_predictions(
        p_gate=p_gate,
        p_c=p_c,
        p_e=p_e,
        alpha_proxy_hard_c=alpha_proxy_hard_c,
        threshold_gate=threshold_gate,
        threshold_ecg=threshold_ecg,
        threshold_eda=threshold_eda,
        confidence_margin_threshold=confidence_margin_threshold,
        confidence_entropy_threshold=confidence_entropy_threshold,
        low_conf_force_modality=low_conf_force_modality,
    )

    p_gate = p_gate.astype(np.float64)
    p_e = p_e.astype(np.float64)
    relvar_delta = relvar_delta.astype(np.float64)
    rescue = (
        (p_gate < float(confident_neg_threshold))
        & (p_e >= float(threshold_eda))
        & (relvar_delta >= float(relvar_gap_threshold))
    )

    pred = np.where(rescue, (p_e >= float(threshold_eda)).astype(np.int64), pred_base).astype(np.int64)
    p_eff = np.where(rescue, p_e.astype(np.float32), p_eff_base).astype(np.float32)
    return pred, p_eff, low_conf.astype(np.int64), rescue.astype(np.int64)


def _build_reliability_features(p: np.ndarray, snr_z: np.ndarray, var_z: np.ndarray) -> np.ndarray:
    p = p.astype(np.float64)
    margin = np.abs(p - 0.5)
    ent = _binary_entropy_np(p)
    return np.stack([margin, ent, snr_z, var_z, np.abs(var_z)], axis=1).astype(np.float32)


def _fit_reliability_and_predict(X_tr: np.ndarray, y_tr: np.ndarray, X_te: np.ndarray) -> np.ndarray:
    if len(X_te) == 0:
        return np.zeros((0,), dtype=np.float32)
    if len(X_tr) == 0:
        return np.full((len(X_te),), 0.5, dtype=np.float32)

    y_tr = y_tr.astype(np.int64)
    if np.unique(y_tr).size < 2:
        return np.full((len(X_te),), float(y_tr[0]), dtype=np.float32)

    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_tr)
    Xte = scaler.transform(X_te)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs")
    clf.fit(Xtr, y_tr)
    probs = clf.predict_proba(Xte)[:, 1]
    return probs.astype(np.float32)


def _build_bandit_state_features(
    p_gate: np.ndarray,
    p_c: np.ndarray,
    p_e: np.ndarray,
    p_hybrid: np.ndarray,
    alpha_gate_c: np.ndarray,
    relvar_delta: np.ndarray,
    low_conf: np.ndarray,
    rescue: np.ndarray,
) -> np.ndarray:
    p_gate = p_gate.astype(np.float64)
    p_c = p_c.astype(np.float64)
    p_e = p_e.astype(np.float64)
    p_hybrid = p_hybrid.astype(np.float64)
    alpha_gate_c = alpha_gate_c.astype(np.float64)
    relvar_delta = relvar_delta.astype(np.float64)
    low_conf = low_conf.astype(np.float64)
    rescue = rescue.astype(np.float64)

    m_gate = np.abs(p_gate - 0.5)
    m_c = np.abs(p_c - 0.5)
    m_e = np.abs(p_e - 0.5)
    m_h = np.abs(p_hybrid - 0.5)
    e_gate = _binary_entropy_np(p_gate)
    e_c = _binary_entropy_np(p_c)
    e_e = _binary_entropy_np(p_e)
    e_h = _binary_entropy_np(p_hybrid)

    X = np.stack(
        [
            p_gate,
            p_c,
            p_e,
            p_hybrid,
            p_e - p_c,
            p_hybrid - p_gate,
            m_gate,
            m_c,
            m_e,
            m_h,
            e_gate,
            e_c,
            e_e,
            e_h,
            alpha_gate_c,
            relvar_delta,
            low_conf,
            rescue,
        ],
        axis=1,
    )
    return X.astype(np.float32)


def _train_contextual_bandit_router(
    X_tr: np.ndarray,
    labels_tr: np.ndarray,
    C: float,
    max_iter: int,
) -> dict[str, object]:
    if len(X_tr) == 0 or len(labels_tr) == 0:
        return {"constant_action": 0}

    labels_tr = labels_tr.astype(np.int64)
    uniq = np.unique(labels_tr)
    if uniq.size < 2:
        return {"constant_action": int(uniq[0])}

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_tr)
    clf = LogisticRegression(
        C=float(C),
        max_iter=int(max_iter),
        solver="lbfgs",
    )
    clf.fit(Xs, labels_tr)
    return {"scaler": scaler, "model": clf, "constant_action": None}


def _predict_contextual_bandit_actions(router: dict[str, object], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(X) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    constant_action = router.get("constant_action", None)
    if constant_action is not None:
        act = np.full((len(X),), int(constant_action), dtype=np.int64)
        conf = np.ones((len(X),), dtype=np.float32)
        return act, conf

    scaler = router["scaler"]
    model = router["model"]
    Xs = scaler.transform(X)
    act = model.predict(Xs).astype(np.int64)
    proba = model.predict_proba(Xs)
    conf = np.max(proba, axis=1).astype(np.float32)
    return act, conf


def _select_mode_from_validation(
    fusion_strategy: str,
    val_gate_f1: float,
    val_ecg_f1: float,
    val_eda_f1: float,
    max_allowed_degradation: float,
    min_gate_advantage: float,
) -> tuple[str, float, bool]:
    if np.isnan(val_gate_f1) or np.isnan(val_ecg_f1) or np.isnan(val_eda_f1):
        return "gate", np.nan, False

    if fusion_strategy == "val_guarded":
        if val_ecg_f1 >= val_gate_f1 and val_ecg_f1 >= val_eda_f1:
            chosen_mode = "ecg"
        elif val_eda_f1 >= val_gate_f1 and val_eda_f1 >= val_ecg_f1:
            chosen_mode = "eda"
        else:
            chosen_mode = "gate"
    else:
        chosen_mode = "gate"

    gate_delta = float(val_gate_f1 - max(val_ecg_f1, val_eda_f1))
    degradation_triggered = False
    if chosen_mode == "gate" and gate_delta < float(min_gate_advantage):
        degradation_triggered = True
        chosen_mode = "ecg" if val_ecg_f1 >= val_eda_f1 else "eda"
    elif chosen_mode == "gate" and gate_delta < -float(max_allowed_degradation):
        degradation_triggered = True
        chosen_mode = "ecg" if val_ecg_f1 >= val_eda_f1 else "eda"

    return chosen_mode, gate_delta, degradation_triggered


def _run_fold_predictions(
    held_out: int,
    fold_subjects: list[int],
    ecg_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    eda_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
    threshold_ecg: float,
    threshold_eda: float,
    split_seed: int,
) -> dict[str, np.ndarray | float | list[int]]:
    train_ids_all = [sid for sid in fold_subjects if sid != held_out]
    train_ids, val_ids = _split_train_val_subjects(train_ids_all, seed=split_seed)
    if not train_ids:
        train_ids = train_ids_all.copy()

    mu_c, sg_c = _fit_scaler([ecg_data[sid] for sid in train_ids])
    mu_e, sg_e = _fit_scaler([eda_data[sid] for sid in train_ids])

    ecg_fold: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    eda_fold: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for sid in fold_subjects:
        Xc, y, t = ecg_data[sid]
        Xe, y2, t2 = eda_data[sid]
        ecg_fold[sid] = (_apply_scaler(Xc, mu_c, sg_c), y, t)
        eda_fold[sid] = (_apply_scaler(Xe, mu_e, sg_e), y2, t2)

    ecg_model = _train_gru(
        train_subjects=ecg_fold,
        train_ids=train_ids,
        val_ids=val_ids,
        seq_len=args.seq_len,
        input_dim=ecg_fold[held_out][0].shape[1],
        hidden_dim=args.hidden_dim_ecg,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs_gru,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr_gru,
        weight_decay=args.weight_decay,
        class_weight=args.class_weight,
        threshold=threshold_ecg,
        device=device,
    )
    eda_model = _train_gru(
        train_subjects=eda_fold,
        train_ids=train_ids,
        val_ids=val_ids,
        seq_len=args.seq_len,
        input_dim=eda_fold[held_out][0].shape[1],
        hidden_dim=args.hidden_dim_eda,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs_gru,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr_gru,
        weight_decay=args.weight_decay,
        class_weight=args.class_weight,
        threshold=threshold_eda,
        device=device,
    )

    ecg_model.eval()
    eda_model.eval()
    for p in ecg_model.parameters():
        p.requires_grad = False
    for p in eda_model.parameters():
        p.requires_grad = False

    def build_for_ids(ids: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs_c, xs_e, ys = [], [], []
        for sid in ids:
            Xc, y, _ = ecg_fold[sid]
            Xe, y2, _ = eda_fold[sid]
            sc, yc = _build_sequences(Xc, y, args.seq_len)
            se, ye = _build_sequences(Xe, y2, args.seq_len)
            if len(yc) == 0:
                continue
            if len(yc) != len(ye):
                raise RuntimeError(f"S{sid}: sequence count mismatch after build")
            if not np.array_equal(yc, ye):
                raise RuntimeError(f"S{sid}: sequence labels mismatch after build")
            xs_c.append(sc)
            xs_e.append(se)
            ys.append(yc)
        if not xs_c:
            return (
                np.zeros((0, args.seq_len, 1), dtype=np.float32),
                np.zeros((0, args.seq_len, 1), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )
        return np.concatenate(xs_c, axis=0), np.concatenate(xs_e, axis=0), np.concatenate(ys, axis=0)

    Xc_tr, Xe_tr, y_tr = build_for_ids(train_ids)
    Xc_va, Xe_va, y_va = build_for_ids(val_ids)
    Xc_te, Xe_te, y_te = build_for_ids([held_out])

    snr_c_tr = _sequence_snr_proxy(Xc_tr)
    snr_e_tr = _sequence_snr_proxy(Xe_tr)
    var_c_tr = _sequence_variance_proxy(Xc_tr)
    var_e_tr = _sequence_variance_proxy(Xe_tr)
    relvar_c_tr = _sequence_relative_variation_proxy(Xc_tr)
    relvar_e_tr = _sequence_relative_variation_proxy(Xe_tr)

    snr_c_mu, snr_c_sd = float(np.mean(snr_c_tr)), float(np.std(snr_c_tr) + 1e-8)
    snr_e_mu, snr_e_sd = float(np.mean(snr_e_tr)), float(np.std(snr_e_tr) + 1e-8)
    var_c_mu, var_c_sd = float(np.mean(var_c_tr)), float(np.std(var_c_tr) + 1e-8)
    var_e_mu, var_e_sd = float(np.mean(var_e_tr)), float(np.std(var_e_tr) + 1e-8)

    p_c_tr, h_c_tr, a_c_tr = _extract_modality_outputs(ecg_model, Xc_tr, args.batch_size, device)
    p_e_tr, h_e_tr, a_e_tr = _extract_modality_outputs(eda_model, Xe_tr, args.batch_size, device)

    snr_c_tr_z = _zscore_with_train_stats(snr_c_tr, snr_c_mu, snr_c_sd)
    snr_e_tr_z = _zscore_with_train_stats(snr_e_tr, snr_e_mu, snr_e_sd)
    var_c_tr_z = _zscore_with_train_stats(var_c_tr, var_c_mu, var_c_sd)
    var_e_tr_z = _zscore_with_train_stats(var_e_tr, var_e_mu, var_e_sd)

    relX_c_tr = _build_reliability_features(p_c_tr, snr_c_tr_z, var_c_tr_z)
    relX_e_tr = _build_reliability_features(p_e_tr, snr_e_tr_z, var_e_tr_z)
    relY_c_tr = ((p_c_tr >= threshold_ecg).astype(np.int64) == y_tr.astype(np.int64)).astype(np.int64)
    relY_e_tr = ((p_e_tr >= threshold_eda).astype(np.int64) == y_tr.astype(np.int64)).astype(np.int64)
    if len(y_va) > 0:
        p_c_va, h_c_va, a_c_va = _extract_modality_outputs(ecg_model, Xc_va, args.batch_size, device)
        p_e_va, h_e_va, a_e_va = _extract_modality_outputs(eda_model, Xe_va, args.batch_size, device)
    else:
        p_c_va = np.zeros((0,), np.float32)
        h_c_va = np.zeros((0,), np.float32)
        a_c_va = np.zeros((0,), np.float32)
        p_e_va = np.zeros((0,), np.float32)
        h_e_va = np.zeros((0,), np.float32)
        a_e_va = np.zeros((0,), np.float32)
    p_c_te, h_c_te, a_c_te = _extract_modality_outputs(ecg_model, Xc_te, args.batch_size, device)
    p_e_te, h_e_te, a_e_te = _extract_modality_outputs(eda_model, Xe_te, args.batch_size, device)

    snr_c_te_z = _zscore_with_train_stats(_sequence_snr_proxy(Xc_te), snr_c_mu, snr_c_sd)
    snr_e_te_z = _zscore_with_train_stats(_sequence_snr_proxy(Xe_te), snr_e_mu, snr_e_sd)
    var_c_te_z = _zscore_with_train_stats(_sequence_variance_proxy(Xc_te), var_c_mu, var_c_sd)
    var_e_te_z = _zscore_with_train_stats(_sequence_variance_proxy(Xe_te), var_e_mu, var_e_sd)

    p_proxy_te, alpha_proxy_c_te, alpha_proxy_e_te = _proxy_fuse_probs(
        p_c=p_c_te,
        p_e=p_e_te,
        snr_c_z=snr_c_te_z,
        snr_e_z=snr_e_te_z,
        var_c_z=var_c_te_z,
        var_e_z=var_e_te_z,
        w_quality=args.proxy_w_quality,
        w_margin=args.proxy_w_margin,
        w_entropy=args.proxy_w_entropy,
        w_snr=args.proxy_w_snr,
        w_var=args.proxy_w_var,
        quality_hard_gap=args.proxy_quality_hard_gap,
    )
    p_proxy_hard_te, alpha_proxy_hard_c_te, alpha_proxy_hard_e_te = _proxy_hard_select_probs(
        p_c=p_c_te,
        p_e=p_e_te,
        snr_c_z=snr_c_te_z,
        snr_e_z=snr_e_te_z,
        var_c_z=var_c_te_z,
        var_e_z=var_e_te_z,
        w_snr=args.proxy_w_snr,
        w_var=args.proxy_w_var,
    )
    p_proxy_hard_tr, alpha_proxy_hard_c_tr, _ = _proxy_hard_select_probs(
        p_c=p_c_tr,
        p_e=p_e_tr,
        snr_c_z=snr_c_tr_z,
        snr_e_z=snr_e_tr_z,
        var_c_z=var_c_tr_z,
        var_e_z=var_e_tr_z,
        w_snr=args.proxy_w_snr,
        w_var=args.proxy_w_var,
    )

    relvar_c_te_z = _zscore_with_train_stats(
        _sequence_relative_variation_proxy(Xc_te),
        float(np.mean(relvar_c_tr)),
        float(np.std(relvar_c_tr) + 1e-8),
    )
    relvar_e_te_z = _zscore_with_train_stats(
        _sequence_relative_variation_proxy(Xe_te),
        float(np.mean(relvar_e_tr)),
        float(np.std(relvar_e_tr) + 1e-8),
    )
    relvar_delta_te = (relvar_c_te_z - relvar_e_te_z).astype(np.float32)
    relvar_c_tr_z = _zscore_with_train_stats(relvar_c_tr, float(np.mean(relvar_c_tr)), float(np.std(relvar_c_tr) + 1e-8))
    relvar_e_tr_z = _zscore_with_train_stats(relvar_e_tr, float(np.mean(relvar_e_tr)), float(np.std(relvar_e_tr) + 1e-8))
    relvar_delta_tr = (relvar_c_tr_z - relvar_e_tr_z).astype(np.float32)

    if len(y_va) > 0:
        snr_c_va_z = _zscore_with_train_stats(_sequence_snr_proxy(Xc_va), snr_c_mu, snr_c_sd)
        snr_e_va_z = _zscore_with_train_stats(_sequence_snr_proxy(Xe_va), snr_e_mu, snr_e_sd)
        var_c_va_z = _zscore_with_train_stats(_sequence_variance_proxy(Xc_va), var_c_mu, var_c_sd)
        var_e_va_z = _zscore_with_train_stats(_sequence_variance_proxy(Xe_va), var_e_mu, var_e_sd)
        p_proxy_va, alpha_proxy_c_va, alpha_proxy_e_va = _proxy_fuse_probs(
            p_c=p_c_va,
            p_e=p_e_va,
            snr_c_z=snr_c_va_z,
            snr_e_z=snr_e_va_z,
            var_c_z=var_c_va_z,
            var_e_z=var_e_va_z,
            w_quality=args.proxy_w_quality,
            w_margin=args.proxy_w_margin,
            w_entropy=args.proxy_w_entropy,
            w_snr=args.proxy_w_snr,
            w_var=args.proxy_w_var,
            quality_hard_gap=args.proxy_quality_hard_gap,
        )
        p_proxy_hard_va, alpha_proxy_hard_c_va, alpha_proxy_hard_e_va = _proxy_hard_select_probs(
            p_c=p_c_va,
            p_e=p_e_va,
            snr_c_z=snr_c_va_z,
            snr_e_z=snr_e_va_z,
            var_c_z=var_c_va_z,
            var_e_z=var_e_va_z,
            w_snr=args.proxy_w_snr,
            w_var=args.proxy_w_var,
        )

        relX_c_va = _build_reliability_features(p_c_va, snr_c_va_z, var_c_va_z)
        relX_e_va = _build_reliability_features(p_e_va, snr_e_va_z, var_e_va_z)
        relY_c_va = ((p_c_va >= threshold_ecg).astype(np.int64) == y_va.astype(np.int64)).astype(np.int64)
        relY_e_va = ((p_e_va >= threshold_eda).astype(np.int64) == y_va.astype(np.int64)).astype(np.int64)
        relvar_c_va_z = _zscore_with_train_stats(
            _sequence_relative_variation_proxy(Xc_va),
            float(np.mean(relvar_c_tr)),
            float(np.std(relvar_c_tr) + 1e-8),
        )
        relvar_e_va_z = _zscore_with_train_stats(
            _sequence_relative_variation_proxy(Xe_va),
            float(np.mean(relvar_e_tr)),
            float(np.std(relvar_e_tr) + 1e-8),
        )
        relvar_delta_va = (relvar_c_va_z - relvar_e_va_z).astype(np.float32)
        p_proxy_hard_tr = p_proxy_hard_tr
    else:
        p_proxy_va = np.zeros((0,), np.float32)
        alpha_proxy_c_va = np.zeros((0,), np.float32)
        alpha_proxy_e_va = np.zeros((0,), np.float32)
        p_proxy_hard_va = np.zeros((0,), np.float32)
        alpha_proxy_hard_c_va = np.zeros((0,), np.float32)
        alpha_proxy_hard_e_va = np.zeros((0,), np.float32)
        relX_c_va = np.zeros((0, 5), np.float32)
        relX_e_va = np.zeros((0, 5), np.float32)
        relY_c_va = np.zeros((0,), np.int64)
        relY_e_va = np.zeros((0,), np.int64)
        relvar_delta_va = np.zeros((0,), np.float32)

    gate = _train_gate(
        p_c_tr=p_c_tr,
        h_c_tr=h_c_tr,
        a_c_tr=a_c_tr,
        p_e_tr=p_e_tr,
        h_e_tr=h_e_tr,
        a_e_tr=a_e_tr,
        y_tr=y_tr,
        p_c_va=p_c_va,
        h_c_va=h_c_va,
        a_c_va=a_c_va,
        p_e_va=p_e_va,
        h_e_va=h_e_va,
        a_e_va=a_e_va,
        y_va=y_va,
        epochs=args.epochs_gate,
        patience=args.patience,
        batch_size=args.batch_size,
        lr=args.lr_gate,
        device=device,
    )

    gate.eval()
    with torch.no_grad():
        act = torch.from_numpy(a_c_tr).to(device)
        hct = torch.from_numpy(h_c_tr).to(device)
        aet = torch.from_numpy(a_e_tr).to(device)
        het = torch.from_numpy(h_e_tr).to(device)
        pct = torch.from_numpy(p_c_tr).to(device)
        pet = torch.from_numpy(p_e_tr).to(device)
        alpha_tr = gate(hct, act, het, aet)
        p_gate_tr = (alpha_tr[:, 0] * pct + alpha_tr[:, 1] * pet).cpu().numpy()
        alpha_gate_c_tr = alpha_tr[:, 0].cpu().numpy()

        ac = torch.from_numpy(a_c_te).to(device)
        hc = torch.from_numpy(h_c_te).to(device)
        ae = torch.from_numpy(a_e_te).to(device)
        he = torch.from_numpy(h_e_te).to(device)
        pc = torch.from_numpy(p_c_te).to(device)
        pe = torch.from_numpy(p_e_te).to(device)
        alpha_te = gate(hc, ac, he, ae)
        p_gate_te = (alpha_te[:, 0] * pc + alpha_te[:, 1] * pe).cpu().numpy()
        alpha_gate_c_te = alpha_te[:, 0].cpu().numpy()
        a_c_gate_mean = float(alpha_te[:, 0].mean().item())
        a_e_gate_mean = float(alpha_te[:, 1].mean().item())

    if len(y_va) > 0:
        with torch.no_grad():
            acv = torch.from_numpy(a_c_va).to(device)
            hcv = torch.from_numpy(h_c_va).to(device)
            aev = torch.from_numpy(a_e_va).to(device)
            hev = torch.from_numpy(h_e_va).to(device)
            pcv = torch.from_numpy(p_c_va).to(device)
            pev = torch.from_numpy(p_e_va).to(device)
            alpha_va = gate(hcv, acv, hev, aev)
            p_gate_va = (alpha_va[:, 0] * pcv + alpha_va[:, 1] * pev).cpu().numpy()
            alpha_gate_c_va = alpha_va[:, 0].cpu().numpy()
    else:
        p_gate_va = np.zeros((0,), np.float32)
        alpha_gate_c_va = np.zeros((0,), np.float32)

    pred_hybrid_tr, p_hybrid_tr, low_conf_tr, rescue_tr = _gate_proxy_hybrid_relvar_predictions(
        p_gate=p_gate_tr,
        p_c=p_c_tr,
        p_e=p_e_tr,
        alpha_proxy_hard_c=alpha_proxy_hard_c_tr,
        relvar_delta=relvar_delta_tr,
        threshold_gate=0.5,
        threshold_ecg=threshold_ecg,
        threshold_eda=threshold_eda,
        confidence_margin_threshold=float(args.override_margin_threshold),
        confidence_entropy_threshold=float(args.override_entropy_threshold),
        confident_neg_threshold=float(args.hybrid_confident_neg_threshold),
        relvar_gap_threshold=float(args.hybrid_relvar_gap_threshold),
        low_conf_force_modality="proxy",
    )
    if len(y_va) > 0:
        pred_hybrid_va, p_hybrid_va, low_conf_va_tmp, rescue_va = _gate_proxy_hybrid_relvar_predictions(
            p_gate=p_gate_va,
            p_c=p_c_va,
            p_e=p_e_va,
            alpha_proxy_hard_c=alpha_proxy_hard_c_va,
            relvar_delta=relvar_delta_va,
            threshold_gate=0.5,
            threshold_ecg=threshold_ecg,
            threshold_eda=threshold_eda,
            confidence_margin_threshold=float(args.override_margin_threshold),
            confidence_entropy_threshold=float(args.override_entropy_threshold),
            confident_neg_threshold=float(args.hybrid_confident_neg_threshold),
            relvar_gap_threshold=float(args.hybrid_relvar_gap_threshold),
            low_conf_force_modality="proxy",
        )
    else:
        p_hybrid_va = np.zeros((0,), dtype=np.float32)
        low_conf_va_tmp = np.zeros((0,), dtype=np.int64)
        rescue_va = np.zeros((0,), dtype=np.int64)
    pred_hybrid_te, p_hybrid_te, low_conf_te_tmp, rescue_te = _gate_proxy_hybrid_relvar_predictions(
        p_gate=p_gate_te,
        p_c=p_c_te,
        p_e=p_e_te,
        alpha_proxy_hard_c=alpha_proxy_hard_c_te,
        relvar_delta=relvar_delta_te,
        threshold_gate=0.5,
        threshold_ecg=threshold_ecg,
        threshold_eda=threshold_eda,
        confidence_margin_threshold=float(args.override_margin_threshold),
        confidence_entropy_threshold=float(args.override_entropy_threshold),
        confident_neg_threshold=float(args.hybrid_confident_neg_threshold),
        relvar_gap_threshold=float(args.hybrid_relvar_gap_threshold),
        low_conf_force_modality="proxy",
    )

    bw = [float(tok.strip()) for tok in str(args.bandit_soft_hybrid_weights).split(",") if tok.strip() != ""]
    if not bw:
        bw = [0.0, 0.5, 1.0]
    bandit_weights = np.asarray(sorted(set(bw)), dtype=np.float32)

    p_actions_tr = (1.0 - bandit_weights[None, :]) * p_gate_tr[:, None] + bandit_weights[None, :] * p_hybrid_tr[:, None]
    reward_tr = (p_actions_tr >= 0.5) == y_tr[:, None]
    reward_tr = reward_tr.astype(np.float32) - float(args.bandit_hybrid_weight_penalty) * bandit_weights[None, :]
    labels_bandit_tr = np.argmax(reward_tr, axis=1).astype(np.int64)

    X_bandit_tr = _build_bandit_state_features(
        p_gate=p_gate_tr,
        p_c=p_c_tr,
        p_e=p_e_tr,
        p_hybrid=p_hybrid_tr,
        alpha_gate_c=alpha_gate_c_tr,
        relvar_delta=relvar_delta_tr,
        low_conf=low_conf_tr,
        rescue=rescue_tr,
    )
    router = _train_contextual_bandit_router(
        X_tr=X_bandit_tr,
        labels_tr=labels_bandit_tr,
        C=float(args.bandit_C),
        max_iter=int(args.bandit_max_iter),
    )

    def _apply_bandit(p_gate_x, p_hybrid_x, p_c_x, p_e_x, alpha_gate_c_x, relvar_delta_x, low_conf_x, rescue_x):
        if len(p_gate_x) == 0:
            return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        Xb = _build_bandit_state_features(
            p_gate=p_gate_x,
            p_c=p_c_x,
            p_e=p_e_x,
            p_hybrid=p_hybrid_x,
            alpha_gate_c=alpha_gate_c_x,
            relvar_delta=relvar_delta_x,
            low_conf=low_conf_x,
            rescue=rescue_x,
        )
        act_idx, conf = _predict_contextual_bandit_actions(router, Xb)
        w = bandit_weights[act_idx]
        p = ((1.0 - w) * p_gate_x + w * p_hybrid_x).astype(np.float32)
        return p, w.astype(np.float32), conf.astype(np.float32)

    p_bandit_va, bandit_w_va, bandit_conf_va = _apply_bandit(
        p_gate_va,
        p_hybrid_va,
        p_c_va,
        p_e_va,
        alpha_gate_c_va,
        relvar_delta_va,
        low_conf_va_tmp,
        rescue_va,
    )
    p_bandit_te, bandit_w_te, bandit_conf_te = _apply_bandit(
        p_gate_te,
        p_hybrid_te,
        p_c_te,
        p_e_te,
        alpha_gate_c_te,
        relvar_delta_te,
        low_conf_te_tmp,
        rescue_te,
    )

    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "y_te": y_te,
        "y_va": y_va,
        "p_c_te": p_c_te,
        "p_e_te": p_e_te,
        "p_gate_te": p_gate_te,
        "p_c_va": p_c_va,
        "p_e_va": p_e_va,
        "p_gate_va": p_gate_va,
        "alpha_gate_c_te": alpha_gate_c_te,
        "alpha_gate_c_va": alpha_gate_c_va,
        "p_proxy_te": p_proxy_te,
        "p_proxy_va": p_proxy_va,
        "p_bandit_te": p_bandit_te,
        "p_bandit_va": p_bandit_va,
        "bandit_w_te": bandit_w_te,
        "bandit_w_va": bandit_w_va,
        "bandit_conf_te": bandit_conf_te,
        "bandit_conf_va": bandit_conf_va,
        "p_proxy_hard_te": p_proxy_hard_te,
        "p_proxy_hard_va": p_proxy_hard_va,
        "alpha_proxy_hard_c_te": alpha_proxy_hard_c_te,
        "alpha_proxy_hard_e_te": alpha_proxy_hard_e_te,
        "alpha_proxy_hard_c_va": alpha_proxy_hard_c_va,
        "relX_c_tr": relX_c_tr,
        "relX_e_tr": relX_e_tr,
        "relY_c_tr": relY_c_tr,
        "relY_e_tr": relY_e_tr,
        "relX_c_va": relX_c_va,
        "relX_e_va": relX_e_va,
        "relY_c_va": relY_c_va,
        "relY_e_va": relY_e_va,
        "relX_c_te": _build_reliability_features(p_c_te, snr_c_te_z, var_c_te_z),
        "relX_e_te": _build_reliability_features(p_e_te, snr_e_te_z, var_e_te_z),
        "relvar_delta_te": relvar_delta_te,
        "relvar_delta_va": relvar_delta_va,
        "p_hybrid_te": p_hybrid_te,
        "p_hybrid_va": p_hybrid_va,
        "a_c_gate_mean": a_c_gate_mean,
        "a_e_gate_mean": a_e_gate_mean,
        "a_c_proxy_mean": float(alpha_proxy_c_te.mean()) if len(alpha_proxy_c_te) > 0 else np.nan,
        "a_e_proxy_mean": float(alpha_proxy_e_te.mean()) if len(alpha_proxy_e_te) > 0 else np.nan,
        "a_c_proxy_hard_mean": float(alpha_proxy_hard_c_te.mean()) if len(alpha_proxy_hard_c_te) > 0 else np.nan,
        "a_e_proxy_hard_mean": float(alpha_proxy_hard_e_te.mean()) if len(alpha_proxy_hard_e_te) > 0 else np.nan,
        "proxy_snr_c_mu": snr_c_mu,
        "proxy_snr_e_mu": snr_e_mu,
        "proxy_var_c_mu": var_c_mu,
        "proxy_var_e_mu": var_e_mu,
    }


def _select_nested_policy(
    outer_train_subjects: list[int],
    ecg_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    eda_data: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    args: argparse.Namespace,
    device: torch.device,
    default_thr_ecg: float,
    default_thr_eda: float,
    default_thr_fused: float,
) -> dict[str, float]:
    inner_subjects = outer_train_subjects
    if args.nested_max_inner_subjects is not None and args.nested_max_inner_subjects > 0:
        k = min(len(outer_train_subjects), int(args.nested_max_inner_subjects))
        if k < len(outer_train_subjects):
            rng = np.random.default_rng(args.seed + 777)
            sampled = rng.choice(np.asarray(outer_train_subjects, dtype=np.int64), size=k, replace=False)
            inner_subjects = sorted(int(x) for x in sampled.tolist())

    grid_ecg = _parse_float_grid(args.nested_grid_ecg, default_thr_ecg)
    grid_eda = _parse_float_grid(args.nested_grid_eda, default_thr_eda)
    grid_fused = _parse_float_grid(args.nested_grid_fused, default_thr_fused)
    grid_deg = _parse_float_grid(args.nested_grid_max_degradation, args.max_allowed_degradation)
    grid_adv = _parse_float_grid(args.nested_grid_min_gate_advantage, args.min_gate_advantage)

    inner_cache = []
    for inner_held_out in inner_subjects:
        fold_pred = _run_fold_predictions(
            held_out=inner_held_out,
            fold_subjects=outer_train_subjects,
            ecg_data=ecg_data,
            eda_data=eda_data,
            args=args,
            device=device,
            threshold_ecg=default_thr_ecg,
            threshold_eda=default_thr_eda,
            split_seed=args.seed + 1000 + inner_held_out,
        )
        inner_cache.append(fold_pred)

    best = {
        "score": float("-inf"),
        "delta": float("-inf"),
        "thr_ecg": default_thr_ecg,
        "thr_eda": default_thr_eda,
        "thr_fused": default_thr_fused,
        "max_deg": args.max_allowed_degradation,
        "min_adv": args.min_gate_advantage,
    }

    for thr_ecg, thr_eda, thr_fused, max_deg, min_adv in itertools.product(grid_ecg, grid_eda, grid_fused, grid_deg, grid_adv):
        fold_f1 = []
        fold_delta = []

        for rec in inner_cache:
            y_va = rec["y_va"]
            y_te = rec["y_te"]
            p_c_va = rec["p_c_va"]
            p_e_va = rec["p_e_va"]
            p_gate_va = rec["p_gate_va"]
            p_c_te = rec["p_c_te"]
            p_e_te = rec["p_e_te"]
            p_gate_te = rec["p_gate_te"]
            p_proxy_va = rec["p_proxy_va"]
            p_proxy_te = rec["p_proxy_te"]
            p_bandit_va = rec["p_bandit_va"]
            p_bandit_te = rec["p_bandit_te"]
            p_proxy_hard_va = rec["p_proxy_hard_va"]
            p_proxy_hard_te = rec["p_proxy_hard_te"]
            alpha_proxy_hard_c_va = rec["alpha_proxy_hard_c_va"]
            alpha_proxy_hard_c_te = rec["alpha_proxy_hard_c_te"]
            relX_c_va = rec["relX_c_va"]
            relX_e_va = rec["relX_e_va"]
            relX_c_te = rec["relX_c_te"]
            relX_e_te = rec["relX_e_te"]
            relvar_delta_va = rec["relvar_delta_va"]
            relvar_delta_te = rec["relvar_delta_te"]

            if args.fusion_strategy in {"proxy_weighted", "proxy_val_guarded"}:
                fused_va, fused_te = p_proxy_va, p_proxy_te
            elif args.fusion_strategy == "proxy_hard_select":
                fused_va, fused_te = p_proxy_hard_va, p_proxy_hard_te
            elif args.fusion_strategy == "contextual_bandit_soft_hybrid":
                fused_va, fused_te = p_bandit_va, p_bandit_te
            else:
                fused_va, fused_te = p_gate_va, p_gate_te

            if len(y_va) > 0:
                if args.fusion_strategy in {"gate_proxy_override", "gate_proxy_hybrid_relvar"}:
                    prior_mode = "proxy"
                    margin_thr = args.override_margin_threshold
                    entropy_thr = args.override_entropy_threshold
                    if args.override_use_subject_prior and len(relX_c_va) > 0 and len(relX_e_va) > 0:
                        q_c = float(np.mean(relX_c_va[:, 2] - relX_c_va[:, 3]))
                        q_e = float(np.mean(relX_e_va[:, 2] - relX_e_va[:, 3]))
                        if (q_e - q_c) >= float(args.override_prior_quality_gap):
                            prior_mode = "eda"
                            margin_thr = margin_thr + float(args.override_prior_margin_boost)
                            entropy_thr = max(0.50, entropy_thr - float(args.override_prior_entropy_drop))
                        elif (q_c - q_e) >= float(args.override_prior_quality_gap):
                            prior_mode = "ecg"
                            margin_thr = margin_thr + float(args.override_prior_margin_boost)
                            entropy_thr = max(0.50, entropy_thr - float(args.override_prior_entropy_drop))
                    if args.fusion_strategy == "gate_proxy_hybrid_relvar":
                        pred_va, _, _, _ = _gate_proxy_hybrid_relvar_predictions(
                            p_gate=p_gate_va,
                            p_c=p_c_va,
                            p_e=p_e_va,
                            alpha_proxy_hard_c=alpha_proxy_hard_c_va,
                            relvar_delta=relvar_delta_va,
                            threshold_gate=thr_fused,
                            threshold_ecg=thr_ecg,
                            threshold_eda=thr_eda,
                            confidence_margin_threshold=margin_thr,
                            confidence_entropy_threshold=entropy_thr,
                            confident_neg_threshold=float(args.hybrid_confident_neg_threshold),
                            relvar_gap_threshold=float(args.hybrid_relvar_gap_threshold),
                            low_conf_force_modality=prior_mode,
                        )
                    else:
                        pred_va, _, _ = _gate_proxy_override_predictions(
                            p_gate=p_gate_va,
                            p_c=p_c_va,
                            p_e=p_e_va,
                            alpha_proxy_hard_c=alpha_proxy_hard_c_va,
                            threshold_gate=thr_fused,
                            threshold_ecg=thr_ecg,
                            threshold_eda=thr_eda,
                            confidence_margin_threshold=margin_thr,
                            confidence_entropy_threshold=entropy_thr,
                            low_conf_force_modality=prior_mode,
                        )
                    val_gate_f1 = float(f1_score(y_va, pred_va, zero_division=0))
                else:
                    val_gate_f1 = _f1_from_probs(y_va, fused_va, thr_fused)
                val_ecg_f1 = _f1_from_probs(y_va, p_c_va, thr_ecg)
                val_eda_f1 = _f1_from_probs(y_va, p_e_va, thr_eda)
            else:
                val_gate_f1 = np.nan
                val_ecg_f1 = np.nan
                val_eda_f1 = np.nan

            selector_strategy = "val_guarded" if args.fusion_strategy in {"val_guarded", "proxy_val_guarded", "gate_proxy_override", "gate_proxy_hybrid_relvar"} else "gate"
            chosen_mode, _, _ = _select_mode_from_validation(
                fusion_strategy=selector_strategy,
                val_gate_f1=val_gate_f1,
                val_ecg_f1=val_ecg_f1,
                val_eda_f1=val_eda_f1,
                max_allowed_degradation=max_deg,
                min_gate_advantage=min_adv,
            )

            if chosen_mode == "ecg":
                f1_fused = _f1_from_probs(y_te, p_c_te, thr_ecg)
            elif chosen_mode == "eda":
                f1_fused = _f1_from_probs(y_te, p_e_te, thr_eda)
            else:
                if args.fusion_strategy in {"gate_proxy_override", "gate_proxy_hybrid_relvar"}:
                    prior_mode = "proxy"
                    margin_thr = args.override_margin_threshold
                    entropy_thr = args.override_entropy_threshold
                    if args.override_use_subject_prior and len(relX_c_te) > 0 and len(relX_e_te) > 0:
                        q_c = float(np.mean(relX_c_te[:, 2] - relX_c_te[:, 3]))
                        q_e = float(np.mean(relX_e_te[:, 2] - relX_e_te[:, 3]))
                        if (q_e - q_c) >= float(args.override_prior_quality_gap):
                            prior_mode = "eda"
                            margin_thr = margin_thr + float(args.override_prior_margin_boost)
                            entropy_thr = max(0.50, entropy_thr - float(args.override_prior_entropy_drop))
                        elif (q_c - q_e) >= float(args.override_prior_quality_gap):
                            prior_mode = "ecg"
                            margin_thr = margin_thr + float(args.override_prior_margin_boost)
                            entropy_thr = max(0.50, entropy_thr - float(args.override_prior_entropy_drop))
                    if args.fusion_strategy == "gate_proxy_hybrid_relvar":
                        pred_te, _, _, _ = _gate_proxy_hybrid_relvar_predictions(
                            p_gate=p_gate_te,
                            p_c=p_c_te,
                            p_e=p_e_te,
                            alpha_proxy_hard_c=alpha_proxy_hard_c_te,
                            relvar_delta=relvar_delta_te,
                            threshold_gate=thr_fused,
                            threshold_ecg=thr_ecg,
                            threshold_eda=thr_eda,
                            confidence_margin_threshold=margin_thr,
                            confidence_entropy_threshold=entropy_thr,
                            confident_neg_threshold=float(args.hybrid_confident_neg_threshold),
                            relvar_gap_threshold=float(args.hybrid_relvar_gap_threshold),
                            low_conf_force_modality=prior_mode,
                        )
                    else:
                        pred_te, _, _ = _gate_proxy_override_predictions(
                            p_gate=p_gate_te,
                            p_c=p_c_te,
                            p_e=p_e_te,
                            alpha_proxy_hard_c=alpha_proxy_hard_c_te,
                            threshold_gate=thr_fused,
                            threshold_ecg=thr_ecg,
                            threshold_eda=thr_eda,
                            confidence_margin_threshold=margin_thr,
                            confidence_entropy_threshold=entropy_thr,
                            low_conf_force_modality=prior_mode,
                        )
                    f1_fused = float(f1_score(y_te, pred_te, zero_division=0))
                else:
                    f1_fused = _f1_from_probs(y_te, fused_te, thr_fused)

            f1_ecg = _f1_from_probs(y_te, p_c_te, thr_ecg)
            f1_eda = _f1_from_probs(y_te, p_e_te, thr_eda)
            fold_f1.append(f1_fused)
            fold_delta.append(f1_fused - max(f1_ecg, f1_eda))

        mean_f1 = float(np.mean(fold_f1)) if fold_f1 else float("-inf")
        mean_delta = float(np.mean(fold_delta)) if fold_delta else float("-inf")

        if mean_f1 > best["score"] + 1e-9 or (abs(mean_f1 - best["score"]) <= 1e-9 and mean_delta > best["delta"]):
            best.update(
                {
                    "score": mean_f1,
                    "delta": mean_delta,
                    "thr_ecg": float(thr_ecg),
                    "thr_eda": float(thr_eda),
                    "thr_fused": float(thr_fused),
                    "max_deg": float(max_deg),
                    "min_adv": float(min_adv),
                }
            )

    return {
        "threshold_ecg": best["thr_ecg"],
        "threshold_eda": best["thr_eda"],
        "threshold_fused": best["thr_fused"],
        "max_allowed_degradation": best["max_deg"],
        "min_gate_advantage": best["min_adv"],
        "nested_mean_fused_f1": best["score"],
        "nested_mean_delta_vs_best": best["delta"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised dynamic gating (ECG+EDA) LOSO benchmark")
    parser.add_argument("--ecg-root", default="data/processed_cardiomind_strict_ratio")
    parser.add_argument("--eda-root", default="data/processed_eda_strict_ratio_aligned_to_ecg")
    parser.add_argument("--stress-label", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--hidden-dim-ecg", type=int, default=64)
    parser.add_argument("--hidden-dim-eda", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs-gru", type=int, default=20)
    parser.add_argument("--epochs-gate", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr-gru", type=float, default=1e-3)
    parser.add_argument("--lr-gate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--decision-threshold", type=float, default=0.4)
    parser.add_argument("--decision-threshold-ecg", type=float, default=None)
    parser.add_argument("--decision-threshold-eda", type=float, default=None)
    parser.add_argument("--decision-threshold-fused", type=float, default=None)
    parser.add_argument("--fusion-strategy", choices=["gate", "val_guarded", "proxy_weighted", "proxy_val_guarded", "proxy_hard_select", "reliability_router", "gate_proxy_override", "gate_proxy_hybrid_relvar", "contextual_bandit_soft_hybrid"], default="val_guarded")
    parser.add_argument("--reliability-margin", type=float, default=0.0, help="Hard routing margin for reliability_router")
    parser.add_argument("--max-allowed-degradation", type=float, default=0.05)
    parser.add_argument("--min-gate-advantage", type=float, default=0.0, help="Require gate val F1 to exceed best single modality by this margin")
    parser.add_argument("--nested-subject-cv", action="store_true")
    parser.add_argument("--nested-max-inner-subjects", type=int, default=None, help="Limit inner held-out subjects for faster nested CV")
    parser.add_argument("--nested-grid-ecg", default=None, help="Comma-separated ECG thresholds for inner CV")
    parser.add_argument("--nested-grid-eda", default=None, help="Comma-separated EDA thresholds for inner CV")
    parser.add_argument("--nested-grid-fused", default=None, help="Comma-separated fused thresholds for inner CV")
    parser.add_argument("--nested-grid-max-degradation", default=None, help="Comma-separated degradation tolerances for inner CV")
    parser.add_argument("--nested-grid-min-gate-advantage", default=None, help="Comma-separated minimum gate advantage values for inner CV")
    parser.add_argument("--proxy-w-quality", type=float, default=2.0)
    parser.add_argument("--proxy-w-margin", type=float, default=1.0)
    parser.add_argument("--proxy-w-entropy", type=float, default=1.0)
    parser.add_argument("--proxy-w-snr", type=float, default=0.5)
    parser.add_argument("--proxy-w-var", type=float, default=0.5)
    parser.add_argument("--proxy-quality-hard-gap", type=float, default=0.75, help="If quality score gap exceeds this, force hard modality selection")
    parser.add_argument("--override-margin-threshold", type=float, default=0.10)
    parser.add_argument("--override-entropy-threshold", type=float, default=0.65)
    parser.add_argument("--hybrid-confident-neg-threshold", type=float, default=0.15)
    parser.add_argument("--hybrid-relvar-gap-threshold", type=float, default=0.10)
    parser.add_argument("--bandit-soft-hybrid-weights", default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--bandit-C", type=float, default=1.0)
    parser.add_argument("--bandit-max-iter", type=int, default=1000)
    parser.add_argument("--bandit-hybrid-weight-penalty", type=float, default=0.02)
    parser.add_argument("--override-use-subject-prior", action="store_true")
    parser.add_argument("--override-prior-quality-gap", type=float, default=0.5)
    parser.add_argument("--override-prior-margin-boost", type=float, default=0.10)
    parser.add_argument("--override-prior-entropy-drop", type=float, default=0.05)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--out-csv", default="runs/supervised_dynamic_gate_loso.csv")
    args = parser.parse_args()

    thr_ecg = args.decision_threshold if args.decision_threshold_ecg is None else float(args.decision_threshold_ecg)
    thr_eda = args.decision_threshold if args.decision_threshold_eda is None else float(args.decision_threshold_eda)
    thr_fused = args.decision_threshold if args.decision_threshold_fused is None else float(args.decision_threshold_fused)

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

    ecg_root = Path(args.ecg_root)
    eda_root = Path(args.eda_root)

    ecg_files = sorted(ecg_root.glob("S*.pt"), key=lambda p: int(p.stem.lstrip("S")))
    if not ecg_files:
        raise FileNotFoundError(f"No ECG files found in {ecg_root}")

    subjects = [int(p.stem.lstrip("S")) for p in ecg_files if (eda_root / p.name).exists()]
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]

    # Load aligned modalities
    ecg_data = {}
    eda_data = {}
    for sid in subjects:
        Xc, yc, tc = _load_ecg_subject(ecg_root / f"S{sid}.pt")
        Xe, ye, te = _load_eda_subject(eda_root / f"S{sid}.pt", stress_label=args.stress_label)

        if len(Xc) != len(Xe) or not np.array_equal(tc, te):
            raise RuntimeError(f"S{sid}: ECG/EDA not aligned; run alignment script first")
        if not np.array_equal(yc.astype(np.int64), ye.astype(np.int64)):
            raise RuntimeError(f"S{sid}: binary labels mismatch between modalities")

        ecg_data[sid] = (Xc, yc.astype(np.int64), tc)
        eda_data[sid] = (Xe, ye.astype(np.int64), te)

    print(f"Dynamic Gate LOSO | subjects={subjects} | seq_len={args.seq_len} | device={device}")

    rows = []

    for held_out in subjects:
        train_ids_all = [sid for sid in subjects if sid != held_out]
        fold_thr_ecg = float(thr_ecg)
        fold_thr_eda = float(thr_eda)
        fold_thr_fused = float(thr_fused)
        fold_max_deg = float(args.max_allowed_degradation)
        fold_min_gate_adv = float(args.min_gate_advantage)
        nested_mean_fused_f1 = np.nan
        nested_mean_delta = np.nan

        if args.nested_subject_cv and len(train_ids_all) >= 4:
            nested = _select_nested_policy(
                outer_train_subjects=train_ids_all,
                ecg_data=ecg_data,
                eda_data=eda_data,
                args=args,
                device=device,
                default_thr_ecg=fold_thr_ecg,
                default_thr_eda=fold_thr_eda,
                default_thr_fused=fold_thr_fused,
            )
            fold_thr_ecg = float(nested["threshold_ecg"])
            fold_thr_eda = float(nested["threshold_eda"])
            fold_thr_fused = float(nested["threshold_fused"])
            fold_max_deg = float(nested["max_allowed_degradation"])
            fold_min_gate_adv = float(nested["min_gate_advantage"])
            nested_mean_fused_f1 = float(nested["nested_mean_fused_f1"])
            nested_mean_delta = float(nested["nested_mean_delta_vs_best"])

        rec = _run_fold_predictions(
            held_out=held_out,
            fold_subjects=subjects,
            ecg_data=ecg_data,
            eda_data=eda_data,
            args=args,
            device=device,
            threshold_ecg=fold_thr_ecg,
            threshold_eda=fold_thr_eda,
            split_seed=args.seed + held_out,
        )

        y_te = rec["y_te"]
        y_va = rec["y_va"]
        p_c_te = rec["p_c_te"]
        p_e_te = rec["p_e_te"]
        p_gate_te = rec["p_gate_te"]
        alpha_gate_c_te = rec["alpha_gate_c_te"]
        p_proxy_te = rec["p_proxy_te"]
        p_proxy_hard_te = rec["p_proxy_hard_te"]
        p_c_va = rec["p_c_va"]
        p_e_va = rec["p_e_va"]
        p_gate_va = rec["p_gate_va"]
        alpha_gate_c_va = rec["alpha_gate_c_va"]
        p_proxy_va = rec["p_proxy_va"]
        p_bandit_te = rec["p_bandit_te"]
        p_bandit_va = rec["p_bandit_va"]
        bandit_w_te = rec["bandit_w_te"]
        bandit_conf_te = rec["bandit_conf_te"]
        p_proxy_hard_va = rec["p_proxy_hard_va"]
        a_c_gate_mean = float(rec["a_c_gate_mean"])
        a_e_gate_mean = float(rec["a_e_gate_mean"])
        a_c_proxy_mean = float(rec["a_c_proxy_mean"])
        a_e_proxy_mean = float(rec["a_e_proxy_mean"])
        a_c_proxy_hard_mean = float(rec["a_c_proxy_hard_mean"])
        a_e_proxy_hard_mean = float(rec["a_e_proxy_hard_mean"])
        alpha_proxy_hard_c_te = rec["alpha_proxy_hard_c_te"]
        alpha_proxy_hard_c_va = rec["alpha_proxy_hard_c_va"]
        relX_c_tr = rec["relX_c_tr"]
        relX_e_tr = rec["relX_e_tr"]
        relY_c_tr = rec["relY_c_tr"]
        relY_e_tr = rec["relY_e_tr"]
        relX_c_va = rec["relX_c_va"]
        relX_e_va = rec["relX_e_va"]
        relY_c_va = rec["relY_c_va"]
        relY_e_va = rec["relY_e_va"]
        relX_c_te = rec["relX_c_te"]
        relX_e_te = rec["relX_e_te"]
        relvar_delta_te = rec["relvar_delta_te"]
        relvar_delta_va = rec["relvar_delta_va"]
        override_prior_mode = "proxy"
        override_margin_eff = float(args.override_margin_threshold)
        override_entropy_eff = float(args.override_entropy_threshold)
        low_conf_mask = np.zeros((len(y_te),), dtype=np.int64)
        hybrid_rescue_mask = np.zeros((len(y_te),), dtype=np.int64)

        pred_c = (p_c_te >= fold_thr_ecg).astype(np.int64)
        pred_e = (p_e_te >= fold_thr_eda).astype(np.int64)
        m_c = _metrics(y_te, pred_c)
        m_e = _metrics(y_te, pred_e)

        if len(y_va) > 0:
            if args.fusion_strategy in {"proxy_weighted", "proxy_val_guarded"}:
                fused_va = p_proxy_va
            elif args.fusion_strategy == "proxy_hard_select":
                fused_va = p_proxy_hard_va
            elif args.fusion_strategy == "contextual_bandit_soft_hybrid":
                fused_va = p_bandit_va
            else:
                fused_va = p_gate_va
            if args.fusion_strategy in {"gate_proxy_override", "gate_proxy_hybrid_relvar"}:
                va_prior_mode = "proxy"
                va_margin_eff = float(args.override_margin_threshold)
                va_entropy_eff = float(args.override_entropy_threshold)
                if args.override_use_subject_prior and len(relX_c_va) > 0 and len(relX_e_va) > 0:
                    q_c_va = float(np.mean(relX_c_va[:, 2] - relX_c_va[:, 3]))
                    q_e_va = float(np.mean(relX_e_va[:, 2] - relX_e_va[:, 3]))
                    if (q_e_va - q_c_va) >= float(args.override_prior_quality_gap):
                        va_prior_mode = "eda"
                        va_margin_eff = va_margin_eff + float(args.override_prior_margin_boost)
                        va_entropy_eff = max(0.50, va_entropy_eff - float(args.override_prior_entropy_drop))
                    elif (q_c_va - q_e_va) >= float(args.override_prior_quality_gap):
                        va_prior_mode = "ecg"
                        va_margin_eff = va_margin_eff + float(args.override_prior_margin_boost)
                        va_entropy_eff = max(0.50, va_entropy_eff - float(args.override_prior_entropy_drop))
                if args.fusion_strategy == "gate_proxy_hybrid_relvar":
                    pred_va_override, _, _, _ = _gate_proxy_hybrid_relvar_predictions(
                        p_gate=p_gate_va,
                        p_c=p_c_va,
                        p_e=p_e_va,
                        alpha_proxy_hard_c=alpha_proxy_hard_c_va,
                        relvar_delta=relvar_delta_va,
                        threshold_gate=fold_thr_fused,
                        threshold_ecg=fold_thr_ecg,
                        threshold_eda=fold_thr_eda,
                        confidence_margin_threshold=va_margin_eff,
                        confidence_entropy_threshold=va_entropy_eff,
                        confident_neg_threshold=float(args.hybrid_confident_neg_threshold),
                        relvar_gap_threshold=float(args.hybrid_relvar_gap_threshold),
                        low_conf_force_modality=va_prior_mode,
                    )
                else:
                    pred_va_override, _, _ = _gate_proxy_override_predictions(
                        p_gate=p_gate_va,
                        p_c=p_c_va,
                        p_e=p_e_va,
                        alpha_proxy_hard_c=alpha_proxy_hard_c_va,
                        threshold_gate=fold_thr_fused,
                        threshold_ecg=fold_thr_ecg,
                        threshold_eda=fold_thr_eda,
                        confidence_margin_threshold=va_margin_eff,
                        confidence_entropy_threshold=va_entropy_eff,
                        low_conf_force_modality=va_prior_mode,
                    )
                val_gate_f1 = float(f1_score(y_va, pred_va_override, zero_division=0))
            else:
                val_gate_f1 = _f1_from_probs(y_va, fused_va, fold_thr_fused)
            val_ecg_f1 = _f1_from_probs(y_va, p_c_va, fold_thr_ecg)
            val_eda_f1 = _f1_from_probs(y_va, p_e_va, fold_thr_eda)
        else:
            val_gate_f1 = np.nan
            val_ecg_f1 = np.nan
            val_eda_f1 = np.nan

        selector_strategy = "val_guarded" if args.fusion_strategy in {"val_guarded", "proxy_val_guarded", "gate_proxy_override", "gate_proxy_hybrid_relvar", "contextual_bandit_soft_hybrid"} else "gate"
        chosen_mode, val_gate_delta, degradation_triggered = _select_mode_from_validation(
            fusion_strategy=selector_strategy,
            val_gate_f1=val_gate_f1,
            val_ecg_f1=val_ecg_f1,
            val_eda_f1=val_eda_f1,
            max_allowed_degradation=fold_max_deg,
            min_gate_advantage=fold_min_gate_adv,
        )

        if args.fusion_strategy == "proxy_weighted":
            chosen_mode = "proxy"
            degradation_triggered = False
        elif args.fusion_strategy == "proxy_hard_select":
            chosen_mode = "proxy_hard"
            degradation_triggered = False
        elif args.fusion_strategy == "reliability_router":
            chosen_mode = "reliability_hard"
            degradation_triggered = False
        elif args.fusion_strategy == "gate_proxy_override":
            chosen_mode = "gate_proxy_override"
            degradation_triggered = False
        elif args.fusion_strategy == "gate_proxy_hybrid_relvar":
            chosen_mode = "gate_proxy_hybrid_relvar"
            degradation_triggered = False
        elif args.fusion_strategy == "contextual_bandit_soft_hybrid":
            chosen_mode = "bandit_soft_hybrid"
            degradation_triggered = False

        if chosen_mode == "ecg":
            p_fused = p_c_te
            a_c_mean, a_e_mean = 1.0, 0.0
            fused_threshold = fold_thr_ecg
        elif chosen_mode == "eda":
            p_fused = p_e_te
            a_c_mean, a_e_mean = 0.0, 1.0
            fused_threshold = fold_thr_eda
        elif chosen_mode == "proxy":
            p_fused = p_proxy_te
            a_c_mean, a_e_mean = a_c_proxy_mean, a_e_proxy_mean
            fused_threshold = fold_thr_fused
        elif chosen_mode == "proxy_hard":
            p_fused = p_proxy_hard_te
            a_c_mean, a_e_mean = a_c_proxy_hard_mean, a_e_proxy_hard_mean
            fused_threshold = fold_thr_fused
            pred_f = np.where(alpha_proxy_hard_c_te >= 0.5, p_c_te >= fold_thr_ecg, p_e_te >= fold_thr_eda).astype(np.int64)
        elif chosen_mode == "reliability_hard":
            relX_c_fit = np.concatenate([relX_c_tr, relX_c_va], axis=0)
            relY_c_fit = np.concatenate([relY_c_tr, relY_c_va], axis=0)
            relX_e_fit = np.concatenate([relX_e_tr, relX_e_va], axis=0)
            relY_e_fit = np.concatenate([relY_e_tr, relY_e_va], axis=0)
            r_c = _fit_reliability_and_predict(relX_c_fit, relY_c_fit, relX_c_te)
            r_e = _fit_reliability_and_predict(relX_e_fit, relY_e_fit, relX_e_te)
            choose_c = r_c >= (r_e + float(args.reliability_margin))
            p_fused = np.where(choose_c, p_c_te, p_e_te).astype(np.float32)
            a_c_mean, a_e_mean = float(np.mean(choose_c.astype(np.float32))), float(np.mean((~choose_c).astype(np.float32)))
            fused_threshold = fold_thr_fused
            pred_f = np.where(choose_c, p_c_te >= fold_thr_ecg, p_e_te >= fold_thr_eda).astype(np.int64)
        elif chosen_mode == "gate_proxy_override":
            if args.override_use_subject_prior and len(relX_c_te) > 0 and len(relX_e_te) > 0:
                q_c_te = float(np.mean(relX_c_te[:, 2] - relX_c_te[:, 3]))
                q_e_te = float(np.mean(relX_e_te[:, 2] - relX_e_te[:, 3]))
                if (q_e_te - q_c_te) >= float(args.override_prior_quality_gap):
                    override_prior_mode = "eda"
                    override_margin_eff = override_margin_eff + float(args.override_prior_margin_boost)
                    override_entropy_eff = max(0.50, override_entropy_eff - float(args.override_prior_entropy_drop))
                elif (q_c_te - q_e_te) >= float(args.override_prior_quality_gap):
                    override_prior_mode = "ecg"
                    override_margin_eff = override_margin_eff + float(args.override_prior_margin_boost)
                    override_entropy_eff = max(0.50, override_entropy_eff - float(args.override_prior_entropy_drop))
            pred_f, p_fused, low_conf_mask = _gate_proxy_override_predictions(
                p_gate=p_gate_te,
                p_c=p_c_te,
                p_e=p_e_te,
                alpha_proxy_hard_c=alpha_proxy_hard_c_te,
                threshold_gate=fold_thr_fused,
                threshold_ecg=fold_thr_ecg,
                threshold_eda=fold_thr_eda,
                confidence_margin_threshold=override_margin_eff,
                confidence_entropy_threshold=override_entropy_eff,
                low_conf_force_modality=override_prior_mode,
            )
            eff_alpha_c = np.where(low_conf_mask >= 1, alpha_proxy_hard_c_te, alpha_gate_c_te)
            a_c_mean, a_e_mean = float(np.mean(eff_alpha_c)), float(np.mean(1.0 - eff_alpha_c))
            fused_threshold = fold_thr_fused
        elif chosen_mode == "gate_proxy_hybrid_relvar":
            if args.override_use_subject_prior and len(relX_c_te) > 0 and len(relX_e_te) > 0:
                q_c_te = float(np.mean(relX_c_te[:, 2] - relX_c_te[:, 3]))
                q_e_te = float(np.mean(relX_e_te[:, 2] - relX_e_te[:, 3]))
                if (q_e_te - q_c_te) >= float(args.override_prior_quality_gap):
                    override_prior_mode = "eda"
                    override_margin_eff = override_margin_eff + float(args.override_prior_margin_boost)
                    override_entropy_eff = max(0.50, override_entropy_eff - float(args.override_prior_entropy_drop))
                elif (q_c_te - q_e_te) >= float(args.override_prior_quality_gap):
                    override_prior_mode = "ecg"
                    override_margin_eff = override_margin_eff + float(args.override_prior_margin_boost)
                    override_entropy_eff = max(0.50, override_entropy_eff - float(args.override_prior_entropy_drop))
            pred_f, p_fused, low_conf_mask, hybrid_rescue_mask = _gate_proxy_hybrid_relvar_predictions(
                p_gate=p_gate_te,
                p_c=p_c_te,
                p_e=p_e_te,
                alpha_proxy_hard_c=alpha_proxy_hard_c_te,
                relvar_delta=relvar_delta_te,
                threshold_gate=fold_thr_fused,
                threshold_ecg=fold_thr_ecg,
                threshold_eda=fold_thr_eda,
                confidence_margin_threshold=override_margin_eff,
                confidence_entropy_threshold=override_entropy_eff,
                confident_neg_threshold=float(args.hybrid_confident_neg_threshold),
                relvar_gap_threshold=float(args.hybrid_relvar_gap_threshold),
                low_conf_force_modality=override_prior_mode,
            )
            eff_alpha_c = np.where(
                hybrid_rescue_mask >= 1,
                0.0,
                np.where(low_conf_mask >= 1, alpha_proxy_hard_c_te, alpha_gate_c_te),
            )
            a_c_mean, a_e_mean = float(np.mean(eff_alpha_c)), float(np.mean(1.0 - eff_alpha_c))
            fused_threshold = fold_thr_fused
        elif chosen_mode == "bandit_soft_hybrid":
            p_fused = p_bandit_te
            a_c_mean = float(np.mean(1.0 - bandit_w_te)) if len(bandit_w_te) > 0 else np.nan
            a_e_mean = float(np.mean(bandit_w_te)) if len(bandit_w_te) > 0 else np.nan
            fused_threshold = fold_thr_fused
            pred_f = (p_fused >= fused_threshold).astype(np.int64)
        else:
            p_fused = p_gate_te
            a_c_mean, a_e_mean = a_c_gate_mean, a_e_gate_mean
            fused_threshold = fold_thr_fused

        if chosen_mode not in {"proxy_hard", "reliability_hard", "gate_proxy_override", "gate_proxy_hybrid_relvar", "bandit_soft_hybrid"}:
            pred_f = (p_fused >= fused_threshold).astype(np.int64)
        m_f = _metrics(y_te, pred_f)

        row = {
            "subject": held_out,
            "ecg_f1": m_c["f1"],
            "eda_f1": m_e["f1"],
            "fused_f1": m_f["f1"],
            "ecg_acc": m_c["accuracy"],
            "eda_acc": m_e["accuracy"],
            "fused_acc": m_f["accuracy"],
            "fused_precision": m_f["precision"],
            "fused_recall": m_f["recall"],
            "alpha_ecg_mean": a_c_mean,
            "alpha_eda_mean": a_e_mean,
            "fusion_mode": chosen_mode,
            "degradation_triggered": int(degradation_triggered),
            "val_gate_delta_vs_best_single": val_gate_delta,
            "val_f1_gate": val_gate_f1,
            "val_f1_ecg": val_ecg_f1,
            "val_f1_eda": val_eda_f1,
            "threshold_ecg": fold_thr_ecg,
            "threshold_eda": fold_thr_eda,
            "threshold_fused": fold_thr_fused,
            "max_allowed_degradation": fold_max_deg,
            "min_gate_advantage": fold_min_gate_adv,
            "nested_cv_enabled": int(args.nested_subject_cv),
            "nested_mean_fused_f1": nested_mean_fused_f1,
            "nested_mean_delta_vs_best": nested_mean_delta,
            "proxy_label_free_runtime": 1,
            "proxy_train_stats_only": 1,
            "proxy_w_margin": float(args.proxy_w_margin),
            "proxy_w_entropy": float(args.proxy_w_entropy),
            "proxy_w_snr": float(args.proxy_w_snr),
            "proxy_w_var": float(args.proxy_w_var),
            "proxy_w_quality": float(args.proxy_w_quality),
            "proxy_quality_hard_gap": float(args.proxy_quality_hard_gap),
            "reliability_margin": float(args.reliability_margin),
            "proxy_snr_c_mu_train": float(rec["proxy_snr_c_mu"]),
            "proxy_snr_e_mu_train": float(rec["proxy_snr_e_mu"]),
            "proxy_var_c_mu_train": float(rec["proxy_var_c_mu"]),
            "proxy_var_e_mu_train": float(rec["proxy_var_e_mu"]),
            "override_margin_threshold": float(args.override_margin_threshold),
            "override_entropy_threshold": float(args.override_entropy_threshold),
            "override_margin_effective": override_margin_eff if chosen_mode == "gate_proxy_override" else np.nan,
            "override_entropy_effective": override_entropy_eff if chosen_mode == "gate_proxy_override" else np.nan,
            "override_prior_mode": override_prior_mode if chosen_mode == "gate_proxy_override" else "none",
            "override_low_conf_rate": float(np.mean(low_conf_mask)) if chosen_mode == "gate_proxy_override" else np.nan,
            "hybrid_confident_neg_threshold": float(args.hybrid_confident_neg_threshold),
            "hybrid_relvar_gap_threshold": float(args.hybrid_relvar_gap_threshold),
            "hybrid_rescue_rate": float(np.mean(hybrid_rescue_mask)) if chosen_mode == "gate_proxy_hybrid_relvar" else np.nan,
            "override_prior_mode_hybrid": override_prior_mode if chosen_mode == "gate_proxy_hybrid_relvar" else "none",
            "override_low_conf_rate_hybrid": float(np.mean(low_conf_mask)) if chosen_mode == "gate_proxy_hybrid_relvar" else np.nan,
            "bandit_weight_mean": float(np.mean(bandit_w_te)) if chosen_mode == "bandit_soft_hybrid" and len(bandit_w_te) > 0 else np.nan,
            "bandit_conf_mean": float(np.mean(bandit_conf_te)) if chosen_mode == "bandit_soft_hybrid" and len(bandit_conf_te) > 0 else np.nan,
            "n_samples": int(len(y_te)),
            "stress_pct": float(np.mean(y_te == 1) * 100.0),
        }
        rows.append(row)

        print(
            f"S{held_out}: ECG f1={row['ecg_f1']:.3f} | EDA f1={row['eda_f1']:.3f} | "
            f"FUSED f1={row['fused_f1']:.3f} [{chosen_mode}] (a_ecg={a_c_mean:.2f}, a_eda={a_e_mean:.2f})"
        )

    if not rows:
        print("No LOSO results produced")
        return

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
        summary = {"subject": "MEAN±STD"}
        for k in fieldnames:
            if k == "subject":
                continue
            try:
                vals = np.asarray([r[k] for r in rows], dtype=np.float64)
                summary[k] = f"{vals.mean():.6f}±{vals.std():.6f}"
            except (TypeError, ValueError):
                summary[k] = "NA"
        w.writerow(summary)

    fused_f1 = np.asarray([r["fused_f1"] for r in rows], dtype=np.float64)
    ecg_f1 = np.asarray([r["ecg_f1"] for r in rows], dtype=np.float64)
    eda_f1 = np.asarray([r["eda_f1"] for r in rows], dtype=np.float64)

    print(f"mean_ecg_f1: {ecg_f1.mean():.3f} ± {ecg_f1.std():.3f}")
    print(f"mean_eda_f1: {eda_f1.mean():.3f} ± {eda_f1.std():.3f}")
    print(f"mean_fused_f1: {fused_f1.mean():.3f} ± {fused_f1.std():.3f}")
    print(f"saved_csv: {out}")


if __name__ == "__main__":
    main()
