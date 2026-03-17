#!/usr/bin/env python3
"""Supervised 3-Modality Dynamic Gate Benchmark (LOSO) using Train-Set Normalized Proxies.

This script acts as the perfect, fair Supervised baseline to compare against the 3-Agent MARL system.
It loads the exact same Train-Set Normalized proxy features from `data/rl_episodes_3mod` and trains
a neural network to map those probabilities and noise metrics into a single fused stress prediction.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset


class ModalityAttention3(nn.Module):
    """Learned 3-way gate: outputs softmax weights for ECG, EDA, BVP probabilities."""
    def __init__(self, noise_dim: int = 12, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, x_prob: torch.Tensor, x_noise: torch.Tensor) -> torch.Tensor:
        # x_prob: (B, 3) probabilities
        # x_noise: (B, 12) proxy noise features
        weights = torch.softmax(self.mlp(x_noise), dim=1)  # (B, 3)
        # Fused prob = w1*p1 + w2*p2 + w3*p3
        p_fused = torch.sum(weights * x_prob, dim=1)
        return p_fused


class DirectFusionMLP(nn.Module):
    """Directly maps all 15 proxy columns (probs + noise) to a single stress probability."""
    def __init__(self, input_dim: int = 15, hidden_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x_prob: torch.Tensor, x_noise: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_prob, x_noise], dim=1)
        return self.mlp(x).squeeze(1)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes-root", default="data/rl_episodes_3mod")
    parser.add_argument("--out-csv", default="runs/benchmark_3mod_supervised_dynamic_gate_f1.csv")
    parser.add_argument("--architecture", default="attention", choices=["attention", "mlp"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _seed_all(args.seed)
    device = torch.device("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
    
    root = Path(args.episodes_root)
    if not root.exists():
        raise FileNotFoundError(f"Missing episodes root: {root}")

    # Gather fold directories (e.g. fold_S10, fold_S11)
    fold_dirs = sorted([d for d in root.iterdir() if d.is_dir() and d.name.startswith("fold_S")], 
                       key=lambda d: int(d.name.replace("fold_S", "")))

    all_results = []

    print(f"Starting 3-Modality Supervised Gate Baseline...")
    print(f"Architecture: {args.architecture} | Device: {device}")

    for fold_dir in fold_dirs:
        _seed_all(args.seed)
        test_subject = int(fold_dir.name.replace("fold_S", ""))
        
        # Load all .pt files inside this fold
        dataset = {}
        for pt_file in fold_dir.glob("S*.pt"):
            sid = int(pt_file.stem.replace("S", ""))
            d = torch.load(pt_file, weights_only=False)
            dataset[sid] = {
                "proxies": d["proxies"],  # (T, 15)
                "labels": d["labels"]
            }

        train_sids = [sid for sid in dataset.keys() if sid != test_subject]
        
        # Split train/val
        np.random.shuffle(train_sids)
        n_val = max(1, int(0.2 * len(train_sids)))
        val_sids = train_sids[:n_val]
        train_sids = train_sids[n_val:]

        def _build_tensors(sids):
            px, yx = [], []
            for s in sids:
                px.append(dataset[s]["proxies"])
                yx.append(dataset[s]["labels"])
            if not px:
                return torch.zeros(0, 15), torch.zeros(0)
            return torch.from_numpy(np.concatenate(px)), torch.from_numpy(np.concatenate(yx)).float()

        X_tr, y_tr = _build_tensors(train_sids)
        X_va, y_va = _build_tensors(val_sids)
        X_te, y_te = _build_tensors([test_subject])

        if len(y_te) == 0:
            continue

        train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_va, y_va), batch_size=args.batch_size, shuffle=False)
        
        # Initialize Model
        if args.architecture == "attention":
            model = ModalityAttention3(noise_dim=12, hidden_dim=32).to(device)
        else:
            model = DirectFusionMLP(input_dim=15, hidden_dim=32).to(device)

        # We optimize using standard Binary Cross Entropy
        # We also enforce class-balancing since stress is ~22%
        pos_weight = (len(y_tr) - y_tr.sum()) / max(y_tr.sum(), 1)
        criterion = nn.BCELoss(weight=torch.where(y_tr.to(device) == 1, pos_weight, 1.0))

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        best_f1, best_state = 0.0, None
        
        for epoch in range(args.epochs):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                p_prob, p_noise = bx[:, 0:3], bx[:, 3:15]
                
                optimizer.zero_grad()
                p_fused = model(p_prob, p_noise)
                
                # Dynamic loss weighting
                batch_weights = torch.where(by == 1, pos_weight, 1.0)
                loss = nn.functional.binary_cross_entropy(p_fused, by, weight=batch_weights)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                va_prob, va_noise = X_va[:, 0:3].to(device), X_va[:, 3:15].to(device)
                va_preds = (model(va_prob, va_noise) >= args.threshold).long().cpu().numpy()
                va_f1 = f1_score(y_va.numpy(), va_preds, zero_division=0)
                
                if va_f1 > best_f1:
                    best_f1 = va_f1
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Evaluate on Held-out test fold
        if best_state is not None:
            model.load_state_dict(best_state)
        
        model.eval()
        with torch.no_grad():
            te_prob, te_noise = X_te[:, 0:3].to(device), X_te[:, 3:15].to(device)
            p_preds = model(te_prob, te_noise)
            preds = (p_preds >= args.threshold).long().cpu().numpy()
            
            # Get individual modality baselines (using threshold 0.5)
            ecg_baseline = (X_te[:, 0].numpy() >= 0.5).astype(int)
            eda_baseline = (X_te[:, 1].numpy() >= 0.5).astype(int)
            bvp_baseline = (X_te[:, 2].numpy() >= 0.5).astype(int)

            ecg_m = _metrics(y_te.numpy(), ecg_baseline)
            eda_m = _metrics(y_te.numpy(), eda_baseline)
            bvp_m = _metrics(y_te.numpy(), bvp_baseline)
            fused_m = _metrics(y_te.numpy(), preds)

            print(f"  [Fold S{test_subject}] ECG: {ecg_m['f1']:.3f} | EDA: {eda_m['f1']:.3f} | BVP: {bvp_m['f1']:.3f} -> SUP. GATE F1: {fused_m['f1']:.3f}")

            all_results.append({
                "subject": test_subject,
                "ecg_f1": ecg_m["f1"],
                "eda_f1": eda_m["f1"],
                "bvp_f1": bvp_m["f1"],
                "fused_f1": fused_m["f1"],
                "fused_acc": fused_m["accuracy"],
                "fused_prec": fused_m["precision"],
                "fused_rec": fused_m["recall"],
            })

    # Save to CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if all_results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
            
            # Means
            means = {"subject": "MEAN"}
            for k in all_results[0].keys():
                if k != "subject":
                    means[k] = np.mean([r[k] for r in all_results])
            writer.writerow(means)

        print(f"\nSaved cross-validation results to {out_csv}")
        print(f"GLOBAL MEAN SUPERVISED F1: {means['fused_f1']:.3f}")
        print(f"GLOBAL MEAN ACCURACY:      {means['fused_acc']:.3f}")

if __name__ == "__main__":
    main()
