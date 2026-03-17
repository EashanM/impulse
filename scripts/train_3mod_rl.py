#!/usr/bin/env python3
"""Train 3-modality RL agents for stress detection.

Uses pre-built episodes (from build_rl_episodes_3mod.py) and trains
three independent PPO agents with shared reward under various configurations.

Supports LOSO evaluation: trains on all folds except held-out, evaluates on held-out.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.config import EnvConfig, RewardConfig
from src.data.dataset_3mod import SubjectEpisode3Mod, PROXY_DIM
from src.envs.stress_env_3mod import StressMonitor3ModEnv
from src.models.actor_critic_3mod import ThreeModActorCritic
from src.training.trainer_3mod import collect_rollout_batched, train_episode_batch


def _load_episode(path: Path) -> SubjectEpisode3Mod:
    """Load a pre-built episode from .pt file."""
    d = torch.load(path, weights_only=False)
    # Read seq_len from config or infer from cardiac_embed
    return SubjectEpisode3Mod(
        subject_id=d["subject_id"],
        cardiac_embed=d["cardiac_embed"],
        somatic_embed=d["somatic_embed"],
        vascular_embed=d["vascular_embed"],
        proxies=d["proxies"],
        labels=d["labels"],
        seq_len=30,  # default, overridden by config
    )


def _evaluate_agent(
    env: StressMonitor3ModEnv,
    agents: ThreeModActorCritic,
    episode: SubjectEpisode3Mod,
    device: torch.device,
    consensus: str = "majority",
) -> dict:
    """Evaluate trained agents on one subject episode (all timesteps)."""
    y_true, y_pred = [], []
    alerts = {"cardiac": 0, "somatic": 0, "vascular": 0}
    detection_latencies = []
    stressor_block_info = episode.precompute_stressor_block_info()
    current_stress_block_start = -1

    T = len(episode)

    # Batch forward pass for speed
    all_obs = {}
    for a in ["cardiac", "somatic", "vascular"]:
        obs_list = [episode.get_observation(t, a) for t in range(T)]
        all_obs[a] = torch.from_numpy(np.array(obs_list, dtype=np.float32)).to(device)

    all_actions = {}
    with torch.no_grad():
        for a in ["cardiac", "somatic", "vascular"]:
            logits, _ = agents.get_agent(a).forward(all_obs[a])
            all_actions[a] = logits.argmax(dim=1).cpu().numpy()

    for t in range(T):
        for a in ["cardiac", "somatic", "vascular"]:
            if all_actions[a][t] == 1:
                alerts[a] += 1

        votes = sum(int(all_actions[a][t]) for a in ["cardiac", "somatic", "vascular"])
        if consensus == "majority":
            joint_alert = votes >= 2
        elif consensus == "and":
            joint_alert = votes == 3
        else:
            joint_alert = votes >= 1

        label = int(episode.labels[t])
        y_true.append(label)
        y_pred.append(int(joint_alert))

        # Track detection latency
        if label == 1:
            steps_into = stressor_block_info[t, 0]
            if steps_into == 0:
                current_stress_block_start = t
            if joint_alert and current_stress_block_start >= 0:
                latency = t - current_stress_block_start
                detection_latencies.append(latency)
                current_stress_block_start = -1

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_samples": len(y_true),
        "stress_pct": float(np.mean(y_true == 1) * 100),
        "alerts_cardiac": alerts["cardiac"],
        "alerts_somatic": alerts["somatic"],
        "alerts_vascular": alerts["vascular"],
        "mean_detection_latency": float(np.mean(detection_latencies)) if detection_latencies else -1.0,
        "n_stressor_blocks_detected": len(detection_latencies),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train 3-modality RL agents")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--episodes-dir", default="data/rl_episodes_3mod")
    parser.add_argument("--seq-len", type=int, default=30)
    parser.add_argument("--total-updates", type=int, default=500)
    parser.add_argument("--episodes-per-update", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--mini-batch-size", type=int, default=256)
    parser.add_argument("--consensus", default="majority", choices=["majority", "and", "or"])
    parser.add_argument("--no-proxies", action="store_true", help="Ablation: zero out proxy features")
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--out-csv", default="runs/rl_3mod_results.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load reward config from YAML
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    reward_cfg = cfg.get("reward", {})
    reward_config = RewardConfig(
        tp_reward=reward_cfg.get("tp_reward", 1.0),
        tn_reward=reward_cfg.get("tn_reward", 0.0),
        fp_penalty=reward_cfg.get("fp_penalty", -1.0),
        fn_penalty=reward_cfg.get("fn_penalty", -1.0),
        lead_time_scaling=reward_cfg.get("lead_time_scaling", False),
        lead_time_floor=reward_cfg.get("lead_time_floor", 0.1),
        terminate_on_fp=reward_cfg.get("terminate_on_fp", False),
        fp_cap=reward_cfg.get("fp_cap", -5.0),
        fp_repeat_scale=reward_cfg.get("fp_repeat_scale", 1.0),
        baseline_step_penalty=reward_cfg.get("baseline_step_penalty", 0.0),
    )

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    episodes_dir = Path(args.episodes_dir)
    fold_dirs = sorted(episodes_dir.glob("fold_S*"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories in {episodes_dir}")

    subjects = [int(d.name.split("_S")[1]) for d in fold_dirs]
    if args.max_subjects:
        subjects = subjects[:args.max_subjects]

    print(f"3-Modality RL | config={args.config} | subjects={subjects} | consensus={args.consensus} | device={device}")

    rows = []
    for fold_idx, held_out in enumerate(subjects):
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        fold_dir = episodes_dir / f"fold_S{held_out}"
        print(f"\n  [{fold_idx+1}/{len(subjects)}] Fold S{held_out}: loading episodes...", flush=True)

        # Load train episodes (all subjects except held_out)
        train_episodes = []
        for sid in subjects:
            if sid == held_out:
                continue
            ep_path = fold_dir / f"S{sid}.pt"
            if ep_path.exists():
                ep = _load_episode(ep_path)
                ep.seq_len = args.seq_len

                if args.no_proxies:
                    ep.proxies = np.zeros_like(ep.proxies)

                train_episodes.append(ep)

        test_path = fold_dir / f"S{held_out}.pt"
        if not test_path.exists():
            print(f"  S{held_out}: skipped (no test episode)")
            continue
        test_episode = _load_episode(test_path)
        test_episode.seq_len = args.seq_len
        if args.no_proxies:
            test_episode.proxies = np.zeros_like(test_episode.proxies)

        if not train_episodes:
            print(f"  S{held_out}: skipped (no train episodes)")
            continue

        # Determine observation dims from data
        sample = train_episodes[0]
        D_c = sample.cardiac_embed.shape[1] + PROXY_DIM
        D_s = sample.somatic_embed.shape[1] + PROXY_DIM
        D_v = sample.vascular_embed.shape[1] + PROXY_DIM

        # Create model
        agents = ThreeModActorCritic(
            cardiac_obs_dim=D_c,
            somatic_obs_dim=D_s,
            vascular_obs_dim=D_v,
            seq_len=args.seq_len,
            hidden=args.hidden_dim,
        ).to(device)

        optimizers = {
            "cardiac": torch.optim.Adam(agents.cardiac.parameters(), lr=args.lr),
            "somatic": torch.optim.Adam(agents.somatic.parameters(), lr=args.lr),
            "vascular": torch.optim.Adam(agents.vascular.parameters(), lr=args.lr),
        }

        env_config = EnvConfig(seq_len=args.seq_len, consensus=args.consensus, random_start=False)
        env = StressMonitor3ModEnv(train_episodes[0], reward_config, env_config)

        # Training loop
        best_reward = float("-inf")
        best_state = None
        t_start = time.time()
        print(f"    Training {args.total_updates} updates...", flush=True)

        for update in range(args.total_updates):
            stats = train_episode_batch(
                env, agents, optimizers, train_episodes, device,
                gamma=args.gamma, gae_lambda=args.gae_lambda,
                clip_epsilon=args.clip_epsilon, entropy_coef=args.entropy_coef,
                ppo_epochs=args.ppo_epochs, mini_batch_size=args.mini_batch_size,
                episodes_per_update=args.episodes_per_update,
            )

            if stats["mean_reward"] > best_reward:
                best_reward = stats["mean_reward"]
                best_state = {k: v.detach().cpu().clone() for k, v in agents.state_dict().items()}

            if (update + 1) % 10 == 0 or update == 0:
                elapsed = time.time() - t_start
                rate = (update + 1) / elapsed
                eta = (args.total_updates - update - 1) / rate
                print(f"    update {update + 1:3d}/{args.total_updates} | "
                      f"reward={stats['mean_reward']:6.1f} best={best_reward:6.1f} | "
                      f"{elapsed:.0f}s elapsed, ~{eta:.0f}s left", flush=True)

        # Load best model
        if best_state:
            agents.load_state_dict(best_state)

        # Evaluate on held-out subject
        test_env = StressMonitor3ModEnv(test_episode, reward_config, env_config)
        result = _evaluate_agent(test_env, agents, test_episode, device, args.consensus)
        result["subject"] = held_out

        rows.append(result)
        print(f"  S{held_out}: F1={result['f1']:.3f} acc={result['accuracy']:.3f} "
              f"latency={result['mean_detection_latency']:.1f} "
              f"alerts=({result['alerts_cardiac']},{result['alerts_somatic']},{result['alerts_vascular']})")

    if not rows:
        print("No results.")
        return

    # Aggregate
    f1s = np.array([r["f1"] for r in rows])
    accs = np.array([r["accuracy"] for r in rows])
    latencies = np.array([r["mean_detection_latency"] for r in rows if r["mean_detection_latency"] >= 0])

    print(f"\nMean F1: {f1s.mean():.3f} ± {f1s.std():.3f}")
    print(f"Mean Accuracy: {accs.mean():.3f} ± {accs.std():.3f}")
    if len(latencies) > 0:
        print(f"Mean Det. Latency: {latencies.mean():.1f} ± {latencies.std():.1f} steps")

    # Save CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
