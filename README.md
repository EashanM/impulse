# IMPULSE: Implementing MARL for Prediction of Longitudinal Stress Episodes

This repository contains the codebase for IMPULSE, a Multi-Agent Reinforcement Learning (MARL) architecture designed to detect acute physiological stress using three modalities: Electrocardiogram (ECG), Electrodermal Activity (EDA), and Blood Volume Pulse (BVP). 

By leveraging independent actor-critic networks that vote on a final consensus, this system quantifies sensor noise, motion artifacts, and varying signal quality in wearable data, outperforming traditional supervised late-fusion algorithms. Validation is performed on the **WESAD** dataset using Leave-One-Subject-Out (LOSO) cross-validation.

---

## 📖 Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [Methodology: The "Train-Set Normalization" Mandate](#2-methodology-the-train-set-normalization-mandate)
3. [Repository Structure (Necessary Files)](#3-repository-structure-necessary-files)
4. [Installation & Setup](#4-installation--setup)
5. [End-to-End Execution Pipeline](#5-end-to-end-execution-pipeline)
6. [Execution of Baselines](#6-baseline--execution)

---

## 1. Architecture Overview

The system is separated into a **Supervised Pre-training Phase** and a **Multi-Agent RL Fusion Phase**.

### 1.1 Supervised Feature Extractors
Instead of training the RL agents from scratch on raw time-series data, we first pre-train three independent neural networks using standard Supervised Cross-Entropy Loss:
*   **Cardiac (ECG):** A GRU processing HRV/HR features over sliding 20-second windows.
*   **Somatic (EDA):** A GRU processing Skin Conductance Response (SCR) features over sliding 20-second windows.
*   **Vascular (BVP):** A 1D-CNN processing raw 1280-length snippets (exactly 20 seconds at 64Hz).

### 1.2 The Observation Space (Embeddings + Proxies)
To build the RL episodes, data is passed through the frozen supervised models. Every second, each RL agent receives an observation vector containing:
1.  **The High-Dimensional Embedding:** The frozen hidden states (64-dim for ECG/BVP, 32-dim for EDA) that encode the structural trends of the biological signal.
2.  **The 15-Dim Quality Proxy Vector:** Explicit metadata to define sensor trust.
    *   `[0:3]`: Raw Probabilities (P(Stress) from ECG, EDA, BVP)
    *   `[3:6]`: Binary Entropy of those probabilities
    *   `[6:9]`: Signal-to-Noise Ratio (SNR)
    *   `[9:12]`: Raw Variance 
    *   `[12:15]`: Relative Variation Deltas between pairs of signals

### 1.3 The Multi-Agent Environment
Three independent Proximal Policy Optimization (PPO) agents make binary decisions (`0` for Wait, `1` for Alert). The environment aggregates these votes using a **Majority Consensus Rule**. 
*   If $N_{alert} \ge 2$, the system output is Stress.
*   Rewards are distributed asymmetrically to optimize for operational targets (e.g., F1 Score: True Positive = `+1.0`, False Positive = `-1.0`).

---

## 2. Methodology: The "Train-Set Normalization" Mandate

A classic pitfall in multimodal time-series fusion is **Data Leakage** via future temporal look-ahead (e.g., standardizing an entire 60-minute test session using its own future mean and variance). 

To ensure our 15-dimensional Quality Proxies (SNR, Variance) don't leak future stress states to the RL agent, we use **Train-Set Normalization** during the LOSO evaluation:
1. For Fold $K$ (e.g., Test Subject 10), we isolate S10's data completely.
2. We compute the Population Mean ($\mu_{train}$) and Standard Deviation ($\sigma_{train}$) of the proxy features across the remaining 14 *Training* subjects.
3. We Z-score S10's data (and the training data) using only these population constants.

Note: we don't normalize the raw probabilities or their associated entropies.

---

## 3. Repository Structure
 
The following 20 core Python files represent the complete pipeline for WESAD.

### Data & Preprocessing (`src/data/`)
*   `wesad_loader.py`: Core loader for the raw WESAD chest/wrist `.pkl` files.
*   `features.py`: Mathematical feature extraction (HRV, SCR) via NeuroKit2 & SciPy.
*   `protocol.py`: Defines WESAD study phases (Baseline, Stress, Amusement, etc.).
*   `preprocessor.py`: Orchestrates ECG feature extraction & CardioMind ratio scaling.
*   `preprocess_eda_strict.py`: Modifies the preprocessor strictly for robust EDA.
*   `preprocess_bvp_raw.py`: Modifies the preprocessor to extract raw BVP snips.
*   `dataset_3mod.py`: PyTorch Module that loads the compiled RL `.pt` episodes.

### Neural Architectures (`src/models/`)
*   `gru_classifier.py`: Supervised architecture for ECG & EDA features.
*   `bvp_cnn_classifier.py`: Supervised 1D-CNN architecture for raw BVP signals.
*   `actor_critic_3mod.py`: The 3 independent Multi-Agent MLPs (Cardiac, Somatic, Vascular).

### Reinforcement Learning (`src/envs/` & `src/training/`)
*   `stress_env_3mod.py`: PettingZoo 3-Agent environment with majority voting logic.
*   `reward.py`: PPO logic for True Positives, False Positives, and early-detection scaling.
*   `trainer_3mod.py`: Main PPO Loop, Rollout Buffer collection, and GAE calculation.

### Executable Scripts (`scripts/`)
*   `align_eda_to_ecg_windows.py` & `align_bvp_to_ecg_windows.py`: Timestamp sync utilities.
*   `benchmark_gru_cardiomind.py`: Trains ECG baseline, saves `.pt` LOSO checkpoints.
*   `benchmark_gru_eda_strict.py`: Trains EDA baseline, saves `.pt` LOSO checkpoints.
*   `benchmark_bvp_cnn.py`: Trains BVP baseline, saves `.pt` LOSO checkpoints.
*   `benchmark_supervised_dynamic_gate.py`: Our ultimate heuristic late-fusion baseline.
*   `build_rl_episodes_3mod.py`: The crucial episode builder that generates RL observations + Proxies using Train-Set Normalization.
*   `train_3mod_rl.py`: The executable that trains the MARL agents and outputs final metrics.

---

## 4. Installation & Setup

1. **Clone the repository.**
2. **Use `uv` to create and manage the environment.** This repository is intended to be run with the included `pyproject.toml` and `uv.lock`.
```bash
uv sync
```
3. **Run commands through `uv`**, e.g.:
```bash
uv run python scripts/benchmark_gru_cardiomind.py --epochs 20
```
4. **Data Placement:** Ensure the raw WESAD dataset is unpacked and accessible. The default path expected by the loader is often `data/WESAD/` or similar per the config.

---

## 5. End-to-End Execution Pipeline

To reproduce the state-of-the-art results from scratch, follow these four steps.

### Step 1: Extract Physiological Features from Raw Data
Convert the raw 700Hz/64Hz waveforms into structured feature arrays per subject.
```bash
# 1. Extract ECG Heart Rate Variability features (CardioMind)
uv run python scripts/preprocess.py --config configs/cardiomind_strict_ratio.yaml

# 2. Extract EDA Skin Conductance Responses (Strict formulation)
uv run python scripts/preprocess_eda_strict.py --config configs/cardiomind_strict_ratio.yaml

# 3. Extract purely Raw BVP 20-second signal snippets
uv run python scripts/preprocess_bvp_raw.py --config configs/cardiomind_strict_ratio.yaml
```

### Step 2: Temporal Alignment
Synchronize the high-frequency wearable modalities to the primary ECG window sequences.
```bash
uv run python scripts/align_eda_to_ecg_windows.py
uv run python scripts/align_bvp_to_ecg_windows.py
```

### Step 2: Evaluate Standalone Supervised Baselines and Setup Checkpoints
Train the standalone neural networks via LOSO. You **must** include the `--save-all-folds-dir` flag, as the RL agents require the frozen weights from every fold ($K=1..15$) to generate episodes.
```bash
# ECG Backbone 
uv run python scripts/benchmark_gru_cardiomind.py --save-all-folds-dir runs/checkpoints/ecg --epochs 20

# EDA Backbone
uv run python scripts/benchmark_gru_eda_strict.py --save-all-folds-dir runs/checkpoints/eda --epochs 30

# BVP Backbone
uv run python scripts/benchmark_bvp_cnn.py --save-all-folds-dir runs/checkpoints/bvp --epochs 30 --batch-size 512
```

### Step 4: Run the Supervised Dynamic Gating Baseline 
For a fair comparison against the RL agents, we employ a supervised dynamic gate model using the exact same pre-trained Modality Embeddings + Train-Set Normalized proxies.
```bash
uv run python scripts/benchmark_3mod_supervised_dynamic_gate.py --architecture attention
```

### Step 5: Build the Normalized RL Episodes
This script performs a 2-pass compilation for each fold: (1) calculating Population Mean/STD for the 14 training subjects, and (2) Z-scoring the structural noise proxies (SNR, Variance, RelVar) without leaking future data. Output episodes are saved as `.pt` tensors.
```bash
# Build embeddings and normalized quality proxies (~2 mins)
uv run python scripts/build_rl_episodes_3mod.py --out-root data/rl_episodes_3mod
```

### Step 6: Train the Multi-Agent RL Ecosystem
Initialize the 3 agents, train them against the completed dataset utilizing the asymmetric (`-1.0, +1.0`) reward structure found in `configs/rl_3mod_f1.yaml`, and evaluate independently on the exact same 15-target test layout.
```bash
uv run python scripts/train_3mod_rl.py \
    --config configs/rl_3mod_f1.yaml \
    --out-csv runs/rl_3mod_f1_loso_final.csv \
    --total-updates 200 \
    --mini-batch-size 512
```
---

## 6. Execution of Legacy and Unimodal Baselines
To demonstrate the advantage of multimodal RL fusion, use these scripts to evaluate the performance of each modality in isolation.

### 6.1 Unimodal Deep Learning Baselines (Without Fusion)
You can directly evaluate the performance of the ECG GRU, EDA GRU, and BVP CNN acting alone:
```bash
# Standalone ECG Evaluation
uv run python scripts/benchmark_gru_cardiomind.py --epochs 20

# Standalone EDA Evaluation
uv run python scripts/benchmark_gru_eda_strict.py --epochs 30

# Standalone BVP Evaluation
uv run python scripts/benchmark_bvp_cnn.py --epochs 30 --batch-size 512
```

### 6.2 Linear Logistic Regression Baselines
For the simplest possible ablation, we evaluate standard Logistic Regression over the extracted features:
```bash
# Unimodal Logistic Regression - ECG Features Only
uv run python scripts/benchmark_linear_cardiomind.py

# Unimodal Logistic Regression - EDA Features Only
uv run python scripts/benchmark_linear_eda_strict.py

# Early-Fusion Multimodal Logistic Regression (ECG + EDA)
uv run python scripts/benchmark_linear.py
```
