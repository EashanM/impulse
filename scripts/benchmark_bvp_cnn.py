#!/usr/bin/env python3
"""Standalone LOSO benchmark for BVP CNN classifier on raw BVP windows.

Input: aligned BVP .pt files from scripts/align_bvp_to_ecg_windows.py
       (data/processed_bvp_raw_aligned_to_ecg/S*.pt)

Each .pt contains:
    bvp_windows:    (N, 1280) float32 — raw filtered BVP samples
    labels:         (N,)      int32   — WESAD labels {1,2,3,4}
    timestamps_sec: (N,)      float64

The script trains a BVPCNNClassifier per LOSO fold, evaluates on the
held-out subject, and writes a CSV with per-subject and aggregate metrics.
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_subject_pt(path: Path, stress_label: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Load BVP windows and binary labels."""
    d = torch.load(path, weights_only=False)
    X = np.asarray(d["bvp_windows"], dtype=np.float32)          # (N, 1280)
    y_raw = np.asarray(d["labels"], dtype=np.int64)
    y = (y_raw == int(stress_label)).astype(np.int64)

    # Remove any windows with NaN/Inf
    valid = np.isfinite(X).all(axis=1)
    return X[valid], y[valid]


def _fit_scaler(train_subjects: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, float]:
    """Compute global mean/std across all training windows for z-score normalization."""
    all_x = np.concatenate([x for x, _ in train_subjects], axis=0)
    mu = float(np.mean(all_x))
    sigma = float(np.std(all_x))
    if sigma < 1e-8:
        sigma = 1.0
    return mu, sigma


def _apply_scaler(X: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    return ((X - mu) / sigma).astype(np.float32)


def _split_train_val_subjects(subjects: list[int], seed: int) -> tuple[list[int], list[int]]:
    if len(subjects) <= 1:
        return subjects, []
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy()
    rng.shuffle(shuffled)
    n_val = max(1, int(round(0.2 * len(shuffled))))
    return sorted(shuffled[n_val:]), sorted(shuffled[:n_val])


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    y_true_list, y_pred_list = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1)[:, 1]
            pred = (probs >= threshold).long().cpu().numpy()
            y_pred_list.extend(pred.tolist())
            y_true_list.extend(yb.numpy().tolist())
    return np.asarray(y_true_list, dtype=np.int64), np.asarray(y_pred_list, dtype=np.int64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="BVP CNN LOSO benchmark")
    parser.add_argument("--data-root", default="data/processed_bvp_raw_aligned_to_ecg")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument("--stress-label", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="balanced")
    parser.add_argument("--gru-hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--out-csv", default="runs/bvp_cnn_loso.csv")
    parser.add_argument("--save-model-path", default=None)
    parser.add_argument("--save-all-folds-dir", default=None,
                        help="Directory to save per-fold checkpoints (bvp_cnn_fold_S{held_out}.pt)")
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

    data_root = Path(args.data_root)
    files = sorted(data_root.glob("S*.pt"), key=lambda p: int(p.stem.lstrip("S")))
    if not files:
        raise FileNotFoundError(f"No S*.pt files in {data_root}")

    subjects = [int(p.stem.lstrip("S")) for p in files]
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
        files = [data_root / f"S{sid}.pt" for sid in subjects]

    print(f"BVP CNN LOSO | subjects={subjects} | device={device}")

    subj_data = {sid: _load_subject_pt(p, stress_label=args.stress_label) for sid, p in zip(subjects, files)}

    # Determine input length from first subject
    first_X = subj_data[subjects[0]][0]
    input_len = first_X.shape[1]
    print(f"  input_len={input_len} (samples per window)")

    rows = []
    last_model = None

    for held_out in subjects:
        _seed_all(args.seed)

        train_ids_all = [sid for sid in subjects if sid != held_out]
        train_ids, val_ids = _split_train_val_subjects(train_ids_all, seed=args.seed)
        if not train_ids:
            train_ids = train_ids_all.copy()

        # Fit scaler on training subjects
        mu, sigma = _fit_scaler([subj_data[sid] for sid in train_ids])

        # Build train/val/test data
        X_tr = np.concatenate([_apply_scaler(subj_data[sid][0], mu, sigma) for sid in train_ids], axis=0)
        y_tr = np.concatenate([subj_data[sid][1] for sid in train_ids], axis=0)

        if val_ids:
            X_va = np.concatenate([_apply_scaler(subj_data[sid][0], mu, sigma) for sid in val_ids], axis=0)
            y_va = np.concatenate([subj_data[sid][1] for sid in val_ids], axis=0)
        else:
            X_va, y_va = np.empty((0, input_len), np.float32), np.empty((0,), np.int64)

        X_te = _apply_scaler(subj_data[held_out][0], mu, sigma)
        y_te = subj_data[held_out][1]

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=args.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va)),
            batch_size=args.batch_size, shuffle=False,
        ) if len(y_va) > 0 else None
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
            batch_size=args.batch_size, shuffle=False,
        )

        # Build model
        model = BVPCNNClassifier(
            input_len=input_len,
            gru_hidden=args.gru_hidden,
            dropout=args.dropout,
        ).to(device)

        # Loss
        if args.class_weight == "balanced":
            n_neg = float(np.sum(y_tr == 0))
            n_pos = float(np.sum(y_tr == 1))
            w0 = 0.5 * (n_neg + n_pos) / max(n_neg, 1.0)
            w1 = 0.5 * (n_neg + n_pos) / max(n_pos, 1.0)
            cw = torch.tensor([w0, w1], dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=cw)
        else:
            criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        best_state = None
        best_val_f1 = float("-inf")
        bad = 0

        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if val_loader is not None:
                yv_true, yv_pred = _evaluate(model, val_loader, device, args.decision_threshold)
                val_f1 = float(f1_score(yv_true, yv_pred, zero_division=0))
                print(f"    e{epoch+1:02d} loss={avg_loss:.4f} val_f1={val_f1:.3f} {'*' if val_f1 > best_val_f1 + 1e-6 else ''}")
                if val_f1 > best_val_f1 + 1e-6:
                    best_val_f1 = val_f1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= args.patience:
                        print(f"    early stop at epoch {epoch+1}")
                        break
            else:
                print(f"    e{epoch+1:02d} loss={avg_loss:.4f}")

        if best_state is not None:
            model.load_state_dict(best_state)
        last_model = model

        # Save per-fold checkpoint for warm-start RL
        if args.save_all_folds_dir:
            folds_dir = Path(args.save_all_folds_dir)
            folds_dir.mkdir(parents=True, exist_ok=True)
            fold_path = folds_dir / f"bvp_cnn_fold_S{held_out}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": {
                        "held_out": held_out,
                        "input_len": input_len,
                        "gru_hidden": args.gru_hidden,
                        "dropout": args.dropout,
                    },
                },
                fold_path,
            )

        # Evaluate on test
        yt_true, yt_pred = _evaluate(model, test_loader, device, args.decision_threshold)

        row = {
            "subject": held_out,
            "accuracy": float(accuracy_score(yt_true, yt_pred)),
            "f1": float(f1_score(yt_true, yt_pred, zero_division=0)),
            "precision": float(precision_score(yt_true, yt_pred, zero_division=0)),
            "recall": float(recall_score(yt_true, yt_pred, zero_division=0)),
            "n_samples": int(len(yt_true)),
            "stress_pct": float(np.mean(yt_true == 1) * 100.0),
        }
        rows.append(row)
        print(
            f"  S{held_out}: acc={row['accuracy']:.3f} f1={row['f1']:.3f} "
            f"prec={row['precision']:.3f} rec={row['recall']:.3f} "
            f"(n={row['n_samples']}, stress={row['stress_pct']:.1f}%)"
        )

    if not rows:
        print("No results.")
        return

    # Summary
    acc = np.array([r["accuracy"] for r in rows], dtype=np.float64)
    f1_arr = np.array([r["f1"] for r in rows], dtype=np.float64)
    pr = np.array([r["precision"] for r in rows], dtype=np.float64)
    rc = np.array([r["recall"] for r in rows], dtype=np.float64)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        fieldnames = ["subject", "accuracy", "f1", "precision", "recall", "n_samples", "stress_pct"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow({
            "subject": "MEAN±STD",
            "accuracy": f"{acc.mean():.6f}±{acc.std():.6f}",
            "f1": f"{f1_arr.mean():.6f}±{f1_arr.std():.6f}",
            "precision": f"{pr.mean():.6f}±{pr.std():.6f}",
            "recall": f"{rc.mean():.6f}±{rc.std():.6f}",
            "n_samples": int(np.mean([r["n_samples"] for r in rows])),
            "stress_pct": f"{np.mean([r['stress_pct'] for r in rows]):.3f}",
        })

    print(f"\nmean_accuracy: {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"mean_f1: {f1_arr.mean():.3f} ± {f1_arr.std():.3f}")
    print(f"mean_precision: {pr.mean():.3f} ± {pr.std():.3f}")
    print(f"mean_recall: {rc.mean():.3f} ± {rc.std():.3f}")
    print(f"saved_csv: {out_csv}")

    if args.save_model_path and last_model is not None:
        save_path = Path(args.save_model_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": last_model.state_dict(),
            "meta": {
                "input_len": input_len,
                "gru_hidden": args.gru_hidden,
                "dropout": args.dropout,
            },
        }, save_path)
        print(f"saved_model: {save_path}")


if __name__ == "__main__":
    main()
