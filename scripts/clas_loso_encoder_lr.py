#!/usr/bin/env python3
"""LOSO: train encoder (linear or CNN+GRU), then logistic regression on embeddings.

Uses CLAS by_block segments under data/CLAS_Database/CLAS. Default binary labels:
high-vs-low load (see src.data.clas_dataset).

Example (quick smoke):
  python scripts/clas_loso_encoder_lr.py --encoder linear --max-subjects 8 \\
      --epochs 5 --device cpu
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from src.data.clas_dataset import (
    DEFAULT_CLAS_ROOT,
    collect_windows_for_participant,
    discover_participant_ids,
)
from src.models.clas_encoders import CLASCnnGruEncoder, CLASLinearEncoder, EncoderWithHead


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def _normalize_channels(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """X (N,C,L), mu/sigma (C,) or scalars broadcast."""
    return ((X - mu.reshape(1, -1, 1)) / sigma.reshape(1, -1, 1)).astype(np.float32)


def _channel_stats(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, c, l = X.shape
    flat = X.transpose(0, 2, 1).reshape(-1, c)
    mu = flat.mean(axis=0)
    sigma = flat.std(axis=0)
    sigma = np.where(sigma < 1e-6, 1.0, sigma)
    return mu.astype(np.float32), sigma.astype(np.float32)


def _load_all_participants(
    clas_root: Path,
    pids: list[int],
    modality: str,
    window_sec: float,
    stride_sec: float,
    target_len: int,
    scheme: str,
    min_quality: float | None,
    quality_modality: str,
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
        )
        if X.shape[0] > 0:
            out[pid] = (X, y)
    return out


def _encode_numpy(
    encoder: nn.Module, loader: DataLoader, device: torch.device
) -> np.ndarray:
    encoder.eval()
    zs: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            z = encoder.encode(xb) if hasattr(encoder, "encode") else encoder(xb)
            zs.append(z.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(zs, axis=0) if zs else np.zeros((0, 1), np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="CLAS LOSO encoder + logistic regression")
    parser.add_argument("--clas-root", type=Path, default=DEFAULT_CLAS_ROOT)
    parser.add_argument("--encoder", choices=["linear", "cnn_gru"], default="linear")
    parser.add_argument("--modality", choices=["ecg2", "ppg", "gsr", "accel3"], default="ppg")
    parser.add_argument("--window-sec", type=float, default=8.0)
    parser.add_argument("--stride-sec", type=float, default=8.0)
    parser.add_argument("--target-len", type=int, default=1024, help="Resampled length per window")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--min-quality", type=float, default=None)
    parser.add_argument("--quality-modality", choices=["ecg", "eda", "ppg"], default="ecg")
    parser.add_argument("--scale-z", action="store_true", help="StandardScaler on embeddings before LR")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--out-csv", type=Path, default=Path("runs/clas_loso_encoder_lr.csv"))
    args = parser.parse_args()

    _seed_all(args.seed)
    device = _device(args.device)

    pids_all = discover_participant_ids(args.clas_root)
    if args.max_subjects is not None:
        pids_all = pids_all[: args.max_subjects]

    data = _load_all_participants(
        args.clas_root,
        pids_all,
        args.modality,
        args.window_sec,
        args.stride_sec,
        args.target_len,
        "high_vs_low",
        args.min_quality,
        args.quality_modality,
    )
    subjects = sorted(data.keys())
    print(f"Loaded windows for {len(subjects)} participants (requested ids up to {len(pids_all)}).")
    if len(subjects) < 2:
        raise SystemExit("Need at least 2 participants with windows. Check CLAS paths and filters.")

    in_channels = {"ecg2": 2, "ppg": 1, "gsr": 1, "accel3": 3}[args.modality]

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "held_out",
        "n_train_windows",
        "n_test_windows",
        "acc",
        "macro_f1",
        "f1_class_0",
        "f1_class_1",
        "precision_macro",
        "recall_macro",
    ]
    fold_rows: list[dict[str, object]] = []

    for held in subjects:
        train_pids = [p for p in subjects if p != held]
        X_tr_list = [data[p][0] for p in train_pids]
        y_tr_list = [data[p][1] for p in train_pids]
        X_tr = np.concatenate(X_tr_list, axis=0)
        y_tr = np.concatenate(y_tr_list, axis=0)
        X_te, y_te = data[held]

        if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
            # Skip degenerate folds for metrics that assume both classes
            continue

        mu, sig = _channel_stats(X_tr)
        X_trn = _normalize_channels(X_tr, mu, sig)
        X_ten = _normalize_channels(X_te, mu, sig)

        # train encoder on 85% of windows
        X_enc, X_val, y_enc, y_val = train_test_split(
            X_trn, y_tr, test_size=0.15, random_state=args.seed + 1, stratify=y_tr
        )

        if args.encoder == "linear":
            enc = CLASLinearEncoder(in_channels, args.target_len, args.embedding_dim).to(device)
            emb_dim = args.embedding_dim
        else:
            enc = CLASCnnGruEncoder(
                in_channels=in_channels,
                seq_len=args.target_len,
                gru_hidden=args.gru_hidden,
                dropout=args.dropout,
            ).to(device)
            emb_dim = enc.embedding_dim

        model = EncoderWithHead(enc, emb_dim, num_classes=2).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        counts = np.bincount(y_enc.astype(np.int64), minlength=2)
        w = counts.sum() / (2 * np.maximum(counts, 1))
        ce = nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32, device=device),
        )

        tr_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_enc),
                torch.from_numpy(y_enc.astype(np.int64)),
            ),
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=False,
        )
        va_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_val),
                torch.from_numpy(y_val.astype(np.int64)),
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )

        best_state = None
        best_val = float("inf")
        for _ in range(args.epochs):
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

        # Logistic regression on embeddings (full train windows)
        full_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_trn),
                torch.from_numpy(y_tr.astype(np.int64)),
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(X_ten),
                torch.from_numpy(y_te.astype(np.int64)),
            ),
            batch_size=args.batch_size,
            shuffle=False,
        )
        Z_tr = _encode_numpy(model.encoder, full_loader, device)
        Z_te = _encode_numpy(model.encoder, test_loader, device)
        if args.scale_z:
            scaler = StandardScaler()
            Z_tr_f = scaler.fit_transform(Z_tr)
            Z_te_f = scaler.transform(Z_te)
        else:
            Z_tr_f, Z_te_f = Z_tr, Z_te

        lr_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=args.seed,
        )
        lr_model.fit(Z_tr_f, y_tr.astype(np.int64))
        pred = lr_model.predict(Z_te_f)

        f1_per = f1_score(y_te, pred, labels=[0, 1], average=None, zero_division=0)
        f1_c0, f1_c1 = float(f1_per[0]), float(f1_per[1])

        fold_rows.append(
            {
                "held_out": held,
                "n_train_windows": int(X_tr.shape[0]),
                "n_test_windows": int(X_te.shape[0]),
                "acc": float(accuracy_score(y_te, pred)),
                "macro_f1": float(f1_score(y_te, pred, average="macro")),
                "f1_class_0": f1_c0,
                "f1_class_1": f1_c1,
                "precision_macro": float(precision_score(y_te, pred, average="macro", zero_division=0)),
                "recall_macro": float(recall_score(y_te, pred, average="macro", zero_division=0)),
            }
        )

    if not fold_rows:
        raise SystemExit("No valid LOSO folds (need both classes in train and test).")

    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in fold_rows:
            w.writerow({k: row[k] for k in fieldnames})

    accs = [float(r["acc"]) for r in fold_rows]
    f1s = [float(r["macro_f1"]) for r in fold_rows]
    print(
        f"LOSO folds: {len(fold_rows)} | acc mean={np.mean(accs):.4f} std={np.std(accs):.4f} | "
        f"macro_f1 mean={np.mean(f1s):.4f} std={np.std(f1s):.4f} | wrote {args.out_csv}"
    )


if __name__ == "__main__":
    main()
