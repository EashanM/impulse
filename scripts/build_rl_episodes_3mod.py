#!/usr/bin/env python3
"""Build RL episodes from frozen encoder checkpoints.

For each LOSO fold, loads frozen ECG GRU, EDA GRU, and BVP CNN checkpoints,
forward-passes each subject's data through them to extract embeddings and
probabilities, computes quality proxies, and saves episode files.

Output: data/rl_episodes_3mod/fold_S{held_out}/S{sid}.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.models.gru_classifier import GRUClassifier
from src.models.bvp_cnn_classifier import BVPCNNClassifier


# ---------------------------------------------------------------------------
# Signal quality proxies (from benchmark_supervised_dynamic_gate.py)
# ---------------------------------------------------------------------------

def _sequence_snr_proxy(X: np.ndarray) -> float:
    """SNR proxy for a feature sequence (T, F) or raw signal (1, L).

    For multi-row inputs, diff along axis=0 (time).
    For single-row inputs (BVP raw), diff along axis=1 (samples).
    """
    signal_power = np.mean(np.square(X))
    if X.shape[0] > 1:
        diff = np.diff(X, axis=0)
    else:
        diff = np.diff(X, axis=1)
    noise_power = np.mean(np.square(diff)) + 1e-8
    return float(signal_power / noise_power)


def _sequence_variance_proxy(X: np.ndarray) -> float:
    """Variance proxy for a feature sequence (T, F) or raw signal (1, L)."""
    return float(np.var(X))


def _sequence_relvar_proxy(X: np.ndarray, eps: float = 1e-3) -> float:
    """Relative variation (coefficient of variation) proxy.

    For single-row BVP, treats the entire row as one signal.
    """
    if X.shape[0] == 1:
        # BVP raw: compute CV over the 1D signal
        mu = np.mean(X)
        sd = np.std(X)
        return float(np.clip(sd / (np.abs(mu) + eps), 0, 50))
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0)
    rel = sd / (np.abs(mu) + eps)
    return float(np.mean(np.clip(rel, 0, 50)))


def _binary_entropy(p: float) -> float:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)))


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def _load_gru_checkpoint(ckpt_path: Path, device: torch.device) -> tuple:
    """Load a GRU checkpoint and return (model, meta)."""
    d = torch.load(ckpt_path, weights_only=False, map_location=device)
    meta = d["meta"]
    model = GRUClassifier(
        input_dim=meta["input_dim"],
        hidden_dim=meta["hidden_dim"],
        num_layers=meta.get("num_layers", 1),
        dropout=meta.get("dropout", 0.0),
        num_classes=meta.get("num_classes", 2),
    ).to(device)
    model.load_state_dict(d["model_state_dict"])
    model.eval()
    return model, meta


def _load_bvp_checkpoint(ckpt_path: Path, device: torch.device) -> tuple:
    """Load a BVP CNN checkpoint and return (model, meta)."""
    d = torch.load(ckpt_path, weights_only=False, map_location=device)
    meta = d["meta"]
    model = BVPCNNClassifier(
        input_len=meta["input_len"],
        gru_hidden=meta["gru_hidden"],
        dropout=meta.get("dropout", 0.3),
    ).to(device)
    model.load_state_dict(d["model_state_dict"])
    model.eval()
    return model, meta


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_gru_embeddings(
    model: GRUClassifier,
    X_seq: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract GRU hidden states and stress probabilities.

    Returns (hidden_states (N, H), probs (N,)).
    """
    loader = DataLoader(TensorDataset(torch.from_numpy(X_seq)), batch_size=batch_size, shuffle=False)
    hiddens, probs = [], []

    for (xb,) in loader:
        xb = xb.to(device)
        _, p_seq, h_seq = model.forward_with_state(xb)
        hiddens.append(h_seq[:, -1, :].cpu().numpy())  # last timestep
        probs.append(p_seq[:, -1].cpu().numpy())

    return np.concatenate(hiddens), np.concatenate(probs)


@torch.no_grad()
def extract_bvp_embeddings(
    model: BVPCNNClassifier,
    X_raw: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract BVP CNN-GRU hidden states and stress probabilities.

    Returns (hidden_states (N, H), probs (N,)).
    """
    loader = DataLoader(TensorDataset(torch.from_numpy(X_raw)), batch_size=batch_size, shuffle=False)
    hiddens, probs = [], []

    for (xb,) in loader:
        xb = xb.to(device)
        _, p_seq, h_seq = model.forward_with_state(xb)
        hiddens.append(h_seq[:, -1, :].cpu().numpy())
        probs.append(p_seq[:, -1].cpu().numpy())

    return np.concatenate(hiddens), np.concatenate(probs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Build 3-modality RL episodes")
    parser.add_argument("--ecg-root", default="data/processed_cardiomind_strict_ratio")
    parser.add_argument("--eda-root", default="data/processed_eda_strict_ratio_aligned_to_ecg")
    parser.add_argument("--bvp-root", default="data/processed_bvp_raw_aligned_to_ecg")
    parser.add_argument("--checkpoint-dir", default="runs/checkpoints")
    parser.add_argument("--out-root", default="data/rl_episodes_3mod")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--stress-label", type=int, default=2)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

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
    bvp_root = Path(args.bvp_root)
    ckpt_dir = Path(args.checkpoint_dir)
    out_root = Path(args.out_root)

    # Discover subjects from ECG data
    ecg_files = sorted(ecg_root.glob("S*.pt"), key=lambda p: int(p.stem.lstrip("S")))
    subjects = [int(p.stem.lstrip("S")) for p in ecg_files]
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]

    print(f"Building RL episodes | subjects={subjects} | device={device}")

    # Load raw data per subject
    def load_ecg(sid):
        d = torch.load(ecg_root / f"S{sid}.pt", weights_only=False)
        X = np.asarray(d["cardiac_features"], dtype=np.float32)
        y = np.asarray(d["labels"], dtype=np.int64)
        ts = np.asarray(d["timestamps_sec"], dtype=np.float64)
        valid = np.isfinite(X).all(axis=1)
        return X[valid], y[valid], ts[valid]

    def load_eda(sid):
        d = torch.load(eda_root / f"S{sid}.pt", weights_only=False)
        X = np.asarray(d["cardiac_features"], dtype=np.float32)
        y_raw = np.asarray(d["labels"], dtype=np.int64)
        y = (y_raw == args.stress_label).astype(np.int64)
        valid = np.isfinite(X).all(axis=1)
        return X[valid], y[valid]

    def load_bvp(sid):
        d = torch.load(bvp_root / f"S{sid}.pt", weights_only=False)
        X = np.asarray(d["bvp_windows"], dtype=np.float32)
        y_raw = np.asarray(d["labels"], dtype=np.int64)
        y = (y_raw == args.stress_label).astype(np.int64)
        valid = np.isfinite(X).all(axis=1)
        return X[valid], y[valid]

    def build_sequences(X, y, seq_len):
        if len(X) < seq_len:
            return np.zeros((0, seq_len, X.shape[1]), np.float32), np.zeros((0,), np.int64)
        xs, ys = [], []
        for t in range(seq_len - 1, len(X)):
            xs.append(X[t - seq_len + 1:t + 1])
            ys.append(y[t])
        return np.array(xs, np.float32), np.array(ys, np.int64)

    subj_data = {}
    for sid in subjects:
        ecg_x, ecg_y, ecg_ts = load_ecg(sid)
        eda_x, eda_y = load_eda(sid)
        bvp_x, bvp_y = load_bvp(sid)

        # Align: use min length
        T = min(len(ecg_x), len(eda_x), len(bvp_x))
        subj_data[sid] = {
            "ecg": ecg_x[:T], "eda": eda_x[:T], "bvp": bvp_x[:T],
            "labels": ecg_y[:T],  # labels should match
            "timestamps_sec": ecg_ts[:T],
        }

    # Process each LOSO fold
    for held_out in subjects:
        print(f"\n  Fold S{held_out}:")

        # Load fold-specific checkpoints
        ecg_ckpt = ckpt_dir / "ecg" / f"ecg_gru_fold_S{held_out}.pt"
        eda_ckpt = ckpt_dir / "eda" / f"eda_gru_fold_S{held_out}.pt"
        bvp_ckpt = ckpt_dir / "bvp" / f"bvp_cnn_fold_S{held_out}.pt"

        if not ecg_ckpt.exists() or not eda_ckpt.exists() or not bvp_ckpt.exists():
            print(f"    Skipping — missing checkpoints for fold S{held_out}")
            continue

        ecg_model, ecg_meta = _load_gru_checkpoint(ecg_ckpt, device)
        eda_model, eda_meta = _load_gru_checkpoint(eda_ckpt, device)
        bvp_model, bvp_meta = _load_bvp_checkpoint(bvp_ckpt, device)

        seq_len = ecg_meta.get("seq_len", args.seq_len)

        # Compute training statistics for z-scoring proxies
        train_ids = [s for s in subjects if s != held_out]
        snr_c_list, snr_e_list, snr_v_list = [], [], []
        var_c_list, var_e_list, var_v_list = [], [], []

        # Compute scaler from training data
        ecg_all = np.concatenate([subj_data[s]["ecg"] for s in train_ids])
        eda_all = np.concatenate([subj_data[s]["eda"] for s in train_ids])
        ecg_mu, ecg_sig = ecg_all.mean(axis=0), ecg_all.std(axis=0)
        ecg_sig[ecg_sig < 1e-8] = 1.0
        eda_mu, eda_sig = eda_all.mean(axis=0), eda_all.std(axis=0)
        eda_sig[eda_sig < 1e-8] = 1.0

        bvp_all = np.concatenate([subj_data[s]["bvp"] for s in train_ids])
        bvp_mu, bvp_sig = float(bvp_all.mean()), float(bvp_all.std())
        if bvp_sig < 1e-8:
            bvp_sig = 1.0

        # Save episodes for all subjects in this fold
        fold_dir = out_root / f"fold_S{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_data = {}
        for sid in subjects:
            data = subj_data[sid]

            # Z-score normalize features
            ecg_norm = (data["ecg"] - ecg_mu) / ecg_sig
            eda_norm = (data["eda"] - eda_mu) / eda_sig
            bvp_norm = (data["bvp"] - bvp_mu) / bvp_sig

            # Build sequences for GRU models
            ecg_seq, _ = build_sequences(ecg_norm, data["labels"], seq_len)
            eda_seq, _ = build_sequences(eda_norm, data["labels"], seq_len)

            # BVP: raw windows (no sequence building needed)
            T_seq = len(ecg_seq)
            bvp_aligned = bvp_norm[seq_len - 1:seq_len - 1 + T_seq]
            labels_aligned = data["labels"][seq_len - 1:seq_len - 1 + T_seq]
            ts_aligned = data["timestamps_sec"][seq_len - 1:seq_len - 1 + T_seq]

            if T_seq == 0:
                print(f"    S{sid}: skipped (no sequences)")
                continue

            # Extract embeddings
            ecg_embed, ecg_probs = extract_gru_embeddings(ecg_model, ecg_seq, device)
            eda_embed, eda_probs = extract_gru_embeddings(eda_model, eda_seq, device)
            bvp_embed, bvp_probs = extract_bvp_embeddings(bvp_model, bvp_aligned, device)

            # Compute per-timestep quality proxies
            proxies = np.zeros((T_seq, 15), dtype=np.float32)
            for i in range(T_seq):
                # Probabilities
                proxies[i, 0] = ecg_probs[i]
                proxies[i, 1] = eda_probs[i]
                proxies[i, 2] = bvp_probs[i]

                # Entropies
                proxies[i, 3] = _binary_entropy(ecg_probs[i])
                proxies[i, 4] = _binary_entropy(eda_probs[i])
                proxies[i, 5] = _binary_entropy(bvp_probs[i])

                # SNR
                proxies[i, 6] = _sequence_snr_proxy(ecg_seq[i])
                proxies[i, 7] = _sequence_snr_proxy(eda_seq[i])
                proxies[i, 8] = _sequence_snr_proxy(bvp_aligned[i:i + 1])

                # Variance
                proxies[i, 9] = _sequence_variance_proxy(ecg_seq[i])
                proxies[i, 10] = _sequence_variance_proxy(eda_seq[i])
                proxies[i, 11] = _sequence_variance_proxy(bvp_aligned[i:i + 1])

            # Pairwise relative variation deltas
            for i in range(T_seq):
                rv_c = _sequence_relvar_proxy(ecg_seq[i])
                rv_e = _sequence_relvar_proxy(eda_seq[i])
                rv_v = _sequence_relvar_proxy(bvp_aligned[i:i + 1])
                proxies[i, 12] = rv_c - rv_e
                proxies[i, 13] = rv_c - rv_v
                proxies[i, 14] = rv_e - rv_v
            
            fold_data[sid] = {
                "ecg_embed": ecg_embed,
                "eda_embed": eda_embed,
                "bvp_embed": bvp_embed,
                "proxies": proxies,
                "labels": labels_aligned,
                "T_seq": T_seq
            }
        
        # Train-Set constraints for proxy normalization
        train_proxies = np.concatenate([fold_data[s]["proxies"] for s in train_ids if s in fold_data], axis=0)
        train_mu = train_proxies.mean(axis=0)
        train_std = train_proxies.std(axis=0)
        train_std[train_std < 1e-8] = 1.0

        for sid in subjects:
            if sid not in fold_data:
                continue
            d = fold_data[sid]
            proxies = d["proxies"]
            
            # Z-score cols 6 to 14 using rigorous train-set stats
            # We explicitly do NOT normalize probabilities/entropies (cols 0-5)
            proxies[:, 6:15] = (proxies[:, 6:15] - train_mu[6:15]) / train_std[6:15]

            # Save episode
            episode_path = fold_dir / f"S{sid}.pt"
            torch.save({
                "subject_id": sid,
                "cardiac_embed": d["ecg_embed"].astype(np.float32),
                "somatic_embed": d["eda_embed"].astype(np.float32),
                "vascular_embed": d["bvp_embed"].astype(np.float32),
                "proxies": proxies.astype(np.float32),
                "labels": d["labels"].astype(np.int64),
            }, episode_path)

            stress_pct = float(np.mean(d["labels"] == 1) * 100)
            print(f"    S{sid}: T={d['T_seq']} embeds=({d['ecg_embed'].shape[1]},{d['eda_embed'].shape[1]},{d['bvp_embed'].shape[1]}) stress={stress_pct:.1f}%")

    print(f"\nDone. Episodes saved to {out_root}/")


if __name__ == "__main__":
    main()
