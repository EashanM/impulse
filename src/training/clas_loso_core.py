"""Shared LOSO pipeline: encoder pretrain + embedding logistic regression (CLAS)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.clas_dataset import collect_windows_for_participant, discover_participant_ids
from src.data.clas_feature_extract import load_preprocessed_participants
from src.models.clas_encoders import (
    CLASCnnGruEncoder,
    CLASGruEncoder,
    CLASLinearEncoder,
    EncoderWithHead,
)


def seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def impute_feature_tensor(
    X: np.ndarray,
    fill: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaNs in (N, C, L). Returns imputed X and per-channel fill values."""
    out = np.asarray(X, dtype=np.float32).copy()
    n, c, l = out.shape
    flat = out.transpose(0, 2, 1).reshape(-1, c)
    if fill is None:
        col_fill = np.nanmean(flat, axis=0)
    else:
        col_fill = np.asarray(fill, dtype=np.float32)
    col_fill = np.where(np.isfinite(col_fill), col_fill, 0.0).astype(np.float32)
    for j in range(c):
        bad = ~np.isfinite(flat[:, j])
        flat[bad, j] = col_fill[j]
    out[:] = flat.reshape(n, l, c).transpose(0, 2, 1)
    return out, col_fill


def normalize_channels(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return ((X - mu.reshape(1, -1, 1)) / sigma.reshape(1, -1, 1)).astype(np.float32)


def channel_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _n, c, _l = X.shape
    flat = X.transpose(0, 2, 1).reshape(-1, c)
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


RAW_MODALITY_CHANNELS: dict[str, int] = {
    "ecg2": 2,
    "ppg": 1,
    "gsr": 1,
    "accel3": 3,
}

FEATURE_MODALITY_KEYS: frozenset[str] = frozenset({"ecg", "eda", "ppg"})

# CNN+GRU uses two pool_size=4 layers; short EDA vectors (5) need padding.
MIN_FEATURE_SEQ_LEN = 32


def resolve_in_channels(modality: str, sample_x: np.ndarray | None = None) -> int:
    if modality in RAW_MODALITY_CHANNELS:
        return RAW_MODALITY_CHANNELS[modality]
    if modality in FEATURE_MODALITY_KEYS:
        return 1
    if sample_x is not None and sample_x.ndim == 3:
        return int(sample_x.shape[1])
    raise ValueError(f"Cannot resolve in_channels for modality {modality!r}")


def pad_feature_seq(X: np.ndarray, min_len: int = MIN_FEATURE_SEQ_LEN) -> np.ndarray:
    """Pad (N, C, L) along L with zeros so CNN+GRU pools do not collapse."""
    if X.shape[2] >= min_len:
        return X
    n, c, l = X.shape
    pad = np.zeros((n, c, min_len - l), dtype=np.float32)
    return np.concatenate([X, pad], axis=2)


def resolve_seq_len(
    modality: str,
    target_len: int,
    sample_x: np.ndarray | None,
) -> int:
    if modality in FEATURE_MODALITY_KEYS and sample_x is not None and sample_x.ndim == 3:
        return int(sample_x.shape[2])
    return target_len


def load_all_participants(
    clas_root: Path,
    pids: list[int],
    modality: str,
    window_sec: float,
    stride_sec: float,
    target_len: int,
    scheme: str,
    min_quality: float | None,
    quality_modality: str,
    processed_root: Path | None = None,
    nk_sub_win_sec: float = 2.0,
    nk_sub_stride_sec: float = 2.0,
    require_cache: bool = False,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pid in pids:
        X, y = collect_windows_for_participant(
            clas_root,
            pid,
            modality=modality,
            window_sec=window_sec,
            stride_sec=stride_sec,
            target_len=target_len,
            scheme=scheme,
            min_quality=min_quality,
            quality_modality=quality_modality,
            processed_root=processed_root,
            nk_sub_win_sec=nk_sub_win_sec,
            nk_sub_stride_sec=nk_sub_stride_sec,
            require_cache=require_cache,
        )
        if X.shape[0] > 0:
            out[pid] = (X, y)
    return out


def encode_numpy(encoder: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    encoder.eval()
    zs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            z = encoder.encode(xb) if hasattr(encoder, "encode") else encoder(xb)
            zs.append(z.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(zs, axis=0) if zs else np.zeros((0, 1), np.float64)


def build_encoder(
    encoder_name: str,
    in_channels: int,
    target_len: int,
    embedding_dim: int,
    gru_hidden: int,
    dropout: float,
    device: torch.device,
) -> tuple[nn.Module, int]:
    if encoder_name == "linear":
        enc = CLASLinearEncoder(in_channels, target_len, embedding_dim).to(device)
        return enc, embedding_dim
    if encoder_name == "gru":
        enc = CLASGruEncoder(
            in_channels=in_channels,
            seq_len=target_len,
            hidden_size=gru_hidden,
            num_layers=1,
            dropout=dropout,
        ).to(device)
        return enc, enc.embedding_dim
    if encoder_name == "cnn_gru":
        enc = CLASCnnGruEncoder(
            in_channels=in_channels,
            seq_len=target_len,
            gru_hidden=gru_hidden,
            dropout=dropout,
        ).to(device)
        return enc, enc.embedding_dim
    raise ValueError(f"Unknown encoder {encoder_name!r}")


def run_loso_encoder_lr(
    *,
    clas_root: Path,
    modality: str,
    encoder: str,
    window_sec: float,
    stride_sec: float,
    target_len: int,
    embedding_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    gru_hidden: int,
    seed: int,
    device: torch.device,
    min_quality: float | None,
    quality_modality: str,
    scale_z: bool,
    max_subjects: int | None,
    scheme: str = "high_vs_low",
    data: dict[int, tuple[np.ndarray, np.ndarray]] | None = None,
    processed_root: Path | None = None,
    nk_sub_win_sec: float = 2.0,
    nk_sub_stride_sec: float = 2.0,
    require_cache: bool = False,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> list[dict[str, object]]:
    """
    Leave-one-subject-out: per fold train encoder, LR on train embeddings, test on held subject.

    If ``data`` is provided (pid -> (X, y)), it is used directly; otherwise data is loaded
    from ``clas_root`` for ``modality``.
    """
    if data is None:
        pids_all = discover_participant_ids(clas_root)
        if max_subjects is not None:
            pids_all = pids_all[:max_subjects]
        data = load_all_participants(
            clas_root,
            pids_all,
            modality,
            window_sec,
            stride_sec,
            target_len,
            scheme,
            min_quality,
            quality_modality,
            processed_root=processed_root,
            nk_sub_win_sec=nk_sub_win_sec,
            nk_sub_stride_sec=nk_sub_stride_sec,
            require_cache=require_cache,
        )

    subjects = sorted(data.keys())
    if len(subjects) < 2:
        return []

    sample_x = data[subjects[0]][0]
    if modality in FEATURE_MODALITY_KEYS:
        data = {pid: (pad_feature_seq(X), y) for pid, (X, y) in data.items()}
        sample_x = data[subjects[0]][0]
    in_channels = resolve_in_channels(modality, sample_x)
    seq_len = resolve_seq_len(modality, target_len, sample_x)
    fold_rows: list[dict[str, object]] = []

    fold_desc = progress_desc or f"LOSO {modality}/{encoder}"
    fold_iter: list[int] | tqdm = (
        tqdm(subjects, desc=fold_desc, unit="fold", leave=False)
        if show_progress
        else subjects
    )

    for held in fold_iter:
        train_pids = [p for p in subjects if p != held]
        X_tr_list = [data[p][0] for p in train_pids]
        y_tr_list = [data[p][1] for p in train_pids]
        X_tr = np.concatenate(X_tr_list, axis=0)
        y_tr = np.concatenate(y_tr_list, axis=0)
        X_te, y_te = data[held]

        if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
            continue

        X_tr_imp, fill = impute_feature_tensor(X_tr)
        X_te_imp, _ = impute_feature_tensor(X_te, fill)
        mu, sig = channel_stats(X_tr_imp)
        X_trn = normalize_channels(X_tr_imp, mu, sig)
        X_ten = normalize_channels(X_te_imp, mu, sig)

        X_enc, X_val, y_enc, y_val = train_test_split(
            X_trn, y_tr, test_size=0.15, random_state=seed + 1, stratify=y_tr
        )

        enc, emb_dim = build_encoder(
            encoder, in_channels, seq_len, embedding_dim, gru_hidden, dropout, device
        )
        model = EncoderWithHead(enc, emb_dim, num_classes=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        counts = np.bincount(y_enc.astype(np.int64), minlength=2)
        w = counts.sum() / (2 * np.maximum(counts, 1))
        ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=device))

        tr_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_enc),
                torch.from_numpy(y_enc.astype(np.int64)),
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
        )
        va_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val),
                torch.from_numpy(y_val.astype(np.int64)),
            ),
            batch_size=batch_size,
            shuffle=False,
        )

        best_state = None
        best_val = float("inf")
        epoch_iter = range(epochs)
        if show_progress and epochs > 1:
            epoch_iter = tqdm(
                epoch_iter,
                desc=f"  train Part{held}",
                unit="epoch",
                leave=False,
                position=2,
            )
        for _ in epoch_iter:
            model.train()
            for xb, yb in tr_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = ce(logits, yb)
                loss.backward()
                opt.step()
            model.eval()
            tot = 0.0
            n = 0
            with torch.no_grad():
                for xb, yb in va_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    logits = model(xb)
                    tot += ce(logits, yb).item() * xb.size(0)
                    n += xb.size(0)
            vloss = tot / max(n, 1)
            if vloss < best_val:
                best_val = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_state is not None:
            model.load_state_dict(best_state)

        full_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_trn),
                torch.from_numpy(y_tr.astype(np.int64)),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_ten),
                torch.from_numpy(y_te.astype(np.int64)),
            ),
            batch_size=batch_size,
            shuffle=False,
        )
        Z_tr = encode_numpy(model.encoder, full_loader, device)
        Z_te = encode_numpy(model.encoder, test_loader, device)
        Z_tr = np.nan_to_num(Z_tr, nan=0.0, posinf=0.0, neginf=0.0)
        Z_te = np.nan_to_num(Z_te, nan=0.0, posinf=0.0, neginf=0.0)
        if scale_z:
            scaler = StandardScaler()
            Z_tr_f = scaler.fit_transform(Z_tr)
            Z_te_f = scaler.transform(Z_te)
        else:
            Z_tr_f, Z_te_f = Z_tr, Z_te

        lr_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
        lr_model.fit(Z_tr_f, y_tr.astype(np.int64))
        pred = lr_model.predict(Z_te_f)

        f1_per = f1_score(y_te, pred, labels=[0, 1], average=None, zero_division=0)
        f1_c0, f1_c1 = float(f1_per[0]), float(f1_per[1])

        acc = float(accuracy_score(y_te, pred))
        macro_f1 = float(f1_score(y_te, pred, average="macro"))
        fold_rows.append(
            {
                "held_out": held,
                "n_train_windows": int(X_tr.shape[0]),
                "n_test_windows": int(X_te.shape[0]),
                "acc": acc,
                "macro_f1": macro_f1,
                "f1_class_0": f1_c0,
                "f1_class_1": f1_c1,
                "precision_macro": float(
                    precision_score(y_te, pred, average="macro", zero_division=0)
                ),
                "recall_macro": float(recall_score(y_te, pred, average="macro", zero_division=0)),
            }
        )
        if show_progress and isinstance(fold_iter, tqdm):
            fold_iter.set_postfix(held=f"Part{held}", acc=f"{acc:.3f}", f1=f"{macro_f1:.3f}")

    return fold_rows
