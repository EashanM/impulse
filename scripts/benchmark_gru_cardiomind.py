#!/usr/bin/env python3
"""Compact GRU LOSO benchmark on CardioMind-style processed features."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from src.models.gru_classifier import GRUClassifier


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_subject_pt(path: Path) -> tuple[np.ndarray, np.ndarray]:
    d = torch.load(path, weights_only=False)
    X = np.asarray(d["cardiac_features"], dtype=np.float32)
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
    return (
        float(np.mean(losses)) if losses else 0.0,
        float(accuracy_score(y_true_np, y_pred_np)),
        float(f1_score(y_true_np, y_pred_np, zero_division=0)),
        float(precision_score(y_true_np, y_pred_np, zero_division=0)),
        float(recall_score(y_true_np, y_pred_np, zero_division=0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LOSO GRU benchmark on CardioMind features")
    parser.add_argument("--data-root", default="data/processed_cardiomind_strict_ratio")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--out-csv", default="runs/gru_cardiomind_loso.csv")
    parser.add_argument("--decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--save-model-path",
        default=None,
        help="Optional path to save a trained supervised GRU checkpoint for warm-start",
    )
    parser.add_argument(
        "--save-held-out",
        type=int,
        default=None,
        help="If set, save checkpoint from the fold where this subject is held out (default: first fold)",
    )
    parser.add_argument(
        "--save-all-folds-dir",
        default=None,
        help="Directory to save per-fold checkpoints (ecg_gru_fold_S{held_out}.pt)",
    )
    parser.add_argument(
        "--loss-csv",
        default=None,
        help="Optional path to save per-epoch train/val losses for each LOSO fold",
    )
    parser.add_argument(
        "--print-epoch-loss",
        action="store_true",
        help="Print per-epoch train/val losses during training",
    )
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
        raise FileNotFoundError(f"No files found in {data_root}")

    subjects = [int(p.stem.lstrip("S")) for p in files]
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
        files = [data_root / f"S{sid}.pt" for sid in subjects]

    subj_data = {sid: _load_subject_pt(p) for sid, p in zip(subjects, files)}
    input_dim = next(iter(subj_data.values()))[0].shape[1]

    print(
        f"GRU LOSO | subjects={subjects} | data_root={data_root} | seq_len={args.seq_len} | "
        f"hidden={args.hidden_dim} | layers={args.num_layers} | class_weight={args.class_weight} | "
        f"threshold={args.decision_threshold:.3f} | device={device}"
    )

    rows = []
    checkpoint_saved = False
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
        X_tr_seq = np.concatenate(tr_seq_list, axis=0)
        y_tr_seq = np.concatenate(tr_y_list, axis=0)

        if len(val_ids) > 0:
            va_seq_list, va_y_list = [], []
            for sid in val_ids:
                X, y = subj_data[sid]
                X = _apply_scaler(X, mu, sigma)
                sx, sy = _build_sequences(X, y, seq_len=args.seq_len)
                if len(sy) > 0:
                    va_seq_list.append(sx)
                    va_y_list.append(sy)
            X_va_seq = np.concatenate(va_seq_list, axis=0) if va_seq_list else np.zeros((0, args.seq_len, input_dim), dtype=np.float32)
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

        model = GRUClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            num_classes=2,
        ).to(device)

        if args.class_weight == "balanced":
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
            val_loss = np.nan

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
                    "val_loss": float(val_loss) if not np.isnan(val_loss) else "",
                    "best_val_loss": float(best_val_loss) if best_val_loss < float("inf") else "",
                    "bad_epochs": int(bad_epochs),
                }
            )
            if args.print_epoch_loss:
                val_str = f"{val_loss:.5f}" if not np.isnan(val_loss) else "NA"
                best_str = f"{best_val_loss:.5f}" if best_val_loss < float("inf") else "NA"
                print(
                    f"    [S{held_out}] epoch={epoch_idx} train_loss={train_loss:.5f} "
                    f"val_loss={val_str} best_val={best_str} bad_epochs={bad_epochs}"
                )

        if best_state is not None:
            model.load_state_dict(best_state)

        _, acc, f1, prec, rec = _evaluate(model, test_loader, device, args.decision_threshold)

        if args.save_model_path and (not checkpoint_saved):
            should_save = args.save_held_out is None or held_out == args.save_held_out
            if should_save:
                save_path = Path(args.save_model_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "meta": {
                            "held_out": held_out,
                            "input_dim": input_dim,
                            "seq_len": args.seq_len,
                            "hidden_dim": args.hidden_dim,
                            "num_layers": args.num_layers,
                            "dropout": args.dropout,
                        },
                    },
                    save_path,
                )
                checkpoint_saved = True
                print(f"saved_model: {save_path} (from held_out={held_out})")

        # Save per-fold checkpoint for warm-start RL
        if args.save_all_folds_dir:
            folds_dir = Path(args.save_all_folds_dir)
            folds_dir.mkdir(parents=True, exist_ok=True)
            fold_path = folds_dir / f"ecg_gru_fold_S{held_out}.pt"
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "meta": {
                        "held_out": held_out,
                        "input_dim": input_dim,
                        "seq_len": args.seq_len,
                        "hidden_dim": args.hidden_dim,
                        "num_layers": args.num_layers,
                        "dropout": args.dropout,
                    },
                },
                fold_path,
            )

        row = {
            "subject": held_out,
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "n_samples": int(len(y_te_seq)),
            "stress_pct": float(np.mean(y_te_seq == 1) * 100.0),
        }
        rows.append(row)
        print(
            f"  S{held_out}: acc={row['accuracy']:.3f} f1={row['f1']:.3f} "
            f"prec={row['precision']:.3f} rec={row['recall']:.3f} "
            f"(n={row['n_samples']}, stress={row['stress_pct']:.1f}%)"
        )

    acc = np.asarray([r["accuracy"] for r in rows], dtype=np.float64)
    f1 = np.asarray([r["f1"] for r in rows], dtype=np.float64)
    pr = np.asarray([r["precision"] for r in rows], dtype=np.float64)
    rc = np.asarray([r["recall"] for r in rows], dtype=np.float64)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        fieldnames = ["subject", "accuracy", "f1", "precision", "recall", "n_samples", "stress_pct"]
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
                "n_samples": int(np.mean([r['n_samples'] for r in rows])),
                "stress_pct": f"{np.mean([r['stress_pct'] for r in rows]):.3f}",
            }
        )

    print(f"mean_accuracy: {acc.mean():.3f} ± {acc.std():.3f}")
    print(f"mean_f1: {f1.mean():.3f} ± {f1.std():.3f}")
    print(f"mean_precision: {pr.mean():.3f} ± {pr.std():.3f}")
    print(f"mean_recall: {rc.mean():.3f} ± {rc.std():.3f}")
    print(f"saved_csv: {out_csv}")

    if args.loss_csv:
        loss_csv = Path(args.loss_csv)
        loss_csv.parent.mkdir(parents=True, exist_ok=True)
        with loss_csv.open("w", newline="") as f:
            fieldnames = ["subject", "epoch", "train_loss", "val_loss", "best_val_loss", "bad_epochs"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(loss_rows)
        print(f"saved_loss_csv: {loss_csv}")


if __name__ == "__main__":
    main()
