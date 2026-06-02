# SWELL-KW benchmark pipelines

Leave-one-subject-out (LOSO) baselines for **SWELL-KW** stress detection. All pipelines use the same binary label rule unless noted:

- **Stress (1):** time pressure (`T`) or interruption (`I`)
- **Non-stress (0):** neutral (`N`); rest (`R`) is dropped when `--exclude-rest` / overview-based extractors omit `R`

Run all commands from the **repository root**:

```bash
uv run python scripts/<script>.py [args...]
```

(`python` in your conda env works the same if you omit `uv run`.)

---

## Quick reference

| Pipeline | Input | Models | Main scripts |
|---|---|---|---|
| **Albaladejo ECG** | 50 HRV features / window | GRU, LSTM | `extract_swell_hrv_features_albaladejo.py` → `benchmark_albaladejo_swell_gru.py` |
| **Albaladejo EDA** | 23 EDA features / window | GRU, LSTM | `extract_swell_eda_features_albaladejo.py` → `benchmark_albaladejo_swell_gru.py` |
| **Minute-level** | HR, RMSSD, SCL per minute | Logistic regression | `preprocess_swell.py` → `benchmark_swell_minute_level.py` |
| **Raw waveform** | 1-channel ECG or EDA windows | Linear, GRU, CNN-GRU | `preprocess_swell_waveform.py` → `benchmark_swell_waveform_loso.py` |

**Default excluded subjects (Albaladejo RNN):** `PP7`, `PP8`, `PP11`, `PP23` (`--exclude PP7 PP8 PP11 PP23`).

**Results / figures:** `runs/` CSVs; confusion matrices and AUROC in `notebooks/albaladejo_w210_s60_confusion_auroc.ipynb`.

---

## Shared library files

| File | Role |
|---|---|
| `src/data/poly5_portilab.py` | Read SWELL Poly5 `.S00` files |
| `src/data/swell_poly5_channels.py` | Pick ECG / EDA channels from Poly5 |
| `src/data/swell_labels.py` | Minute CSV label helpers (`preprocess_swell.py`) |
| `src/data/swell_hrv_albaladejo.py` | ECG HRV feature extraction (50 features) |
| `src/data/swell_eda_albaladejo.py` | EDA feature extraction (23 features) |
| `src/training/albaladejo_swell_sequences.py` | Sequence building, LOSO training loop |
| `src/models/gru_classifier.py` | `GRUClassifier`, `LSTMClassifier` |
| `src/models/swell_baselines.py` | Raw-waveform Linear / GRU / CNN-GRU |

---

## 1. Albaladejo ECG (HRV features) — GRU / LSTM

Sliding windows on **ECG** from Poly5 → handcrafted HRV features → sequences of windows → unidirectional RNN.

### Relevant files

| Step | File |
|---|---|
| Extract | `scripts/extract_swell_hrv_features_albaladejo.py` |
| Train / LOSO | `scripts/benchmark_albaladejo_swell_gru.py` |
| Features | `src/data/swell_hrv_albaladejo.py` |

**Raw inputs:** `data/raw/SWELL/0 - Raw data/.../Mobi signals (raw and filtered)/*.S00`, `data/raw/SWELL/SWELL-KW - overview available data.xlsx`

**Outputs:** `data/processed_swell_hrv_albaladejo_w{W}_s{H}/PP*.npz`, `runs/albaladejo_swell_*_loso_*.csv`, `runs/*_predictions.csv`

### Step 1 — Extract features

```bash
uv run python scripts/extract_swell_hrv_features_albaladejo.py \
  --out-dir data/processed_swell_hrv_albaladejo_w210_s60 \
  --window-sec 210 \
  --stride-sec 60
```

Another common setting (20 s / 5 s):

```bash
uv run python scripts/extract_swell_hrv_features_albaladejo.py \
  --out-dir data/processed_swell_hrv_albaladejo_w20_s5 \
  --window-sec 20 \
  --stride-sec 5
```

Each `PP{N}.npz` contains `X` `(n_windows, 50)`, `y`, `block_num`, `timestamp_sec`, etc.

### Step 2 — GRU

```bash
uv run python scripts/benchmark_albaladejo_swell_gru.py \
  --data-root data/processed_swell_hrv_albaladejo_w210_s60 \
  --rnn-type gru \
  --seq-len 14 \
  --hidden-dim 32 \
  --num-layers 1 \
  --dropout 0.1 \
  --lr 0.001 \
  --weight-decay 0 \
  --class-weight balanced \
  --decision-threshold-tune f1_val \
  --epochs 30 \
  --patience 7 \
  --batch-size 256 \
  --seed 42 \
  --exclude PP7 PP8 PP11 PP23 \
  --out-csv runs/albaladejo_swell_gru_loso_w210_s60.csv
```

Predictions are written automatically to `runs/albaladejo_swell_gru_loso_w210_s60_predictions.csv` unless you pass `--out-predictions-csv`.

### Step 2 — LSTM

Same command as GRU; set `--rnn-type lstm` and change the output path:

```bash
uv run python scripts/benchmark_albaladejo_swell_gru.py \
  --data-root data/processed_swell_hrv_albaladejo_w210_s60 \
  --rnn-type lstm \
  --seq-len 14 \
  --hidden-dim 32 \
  --num-layers 1 \
  --dropout 0.1 \
  --lr 0.001 \
  --weight-decay 0 \
  --class-weight balanced \
  --decision-threshold-tune f1_val \
  --epochs 30 \
  --patience 7 \
  --batch-size 256 \
  --seed 42 \
  --exclude PP7 PP8 PP11 PP23 \
  --out-csv runs/albaladejo_swell_ecg_lstm_loso_w210_s60.csv
```

**Default RNN training:** Adam `lr=1e-3`, `weight_decay=0`, batch 256, up to 30 epochs, patience 7 (early stop on val F1), class-balanced CE, 20% of train subjects held for validation, optional `f1_val` threshold tuning.

---

## 2. Albaladejo EDA (extracted features) — GRU / LSTM

Same training script as ECG; different extractor and NPZ directory.

### Relevant files

| Step | File |
|---|---|
| Extract | `scripts/extract_swell_eda_features_albaladejo.py` |
| Train / LOSO | `scripts/benchmark_albaladejo_swell_gru.py` |
| Features | `src/data/swell_eda_albaladejo.py` |

### Step 1 — Extract features

```bash
uv run python scripts/extract_swell_eda_features_albaladejo.py \
  --out-dir data/processed_swell_eda_albaladejo_w20_s5_v2 \
  --window-sec 20 \
  --stride-sec 5
```

Each `PP{N}.npz` contains `X` `(n_windows, 23)`, `y`, etc. (23 EDA features: mean, SCR counts, phasic/tonic stats, …).

### Step 2 — GRU

```bash
uv run python scripts/benchmark_albaladejo_swell_gru.py \
  --data-root data/processed_swell_eda_albaladejo_w20_s5_v2 \
  --rnn-type gru \
  --seq-len 14 \
  --hidden-dim 32 \
  --num-layers 1 \
  --dropout 0.1 \
  --lr 0.001 \
  --class-weight balanced \
  --decision-threshold-tune f1_val \
  --epochs 30 \
  --patience 7 \
  --batch-size 256 \
  --seed 42 \
  --exclude PP7 PP8 PP11 PP23 \
  --out-csv runs/albaladejo_swell_eda_gru_loso_w20_s5_v2_seq14.csv
```

### Step 2 — LSTM

```bash
uv run python scripts/benchmark_albaladejo_swell_gru.py \
  --data-root data/processed_swell_eda_albaladejo_w20_s5_v2 \
  --rnn-type lstm \
  --seq-len 14 \
  --hidden-dim 32 \
  --num-layers 1 \
  --dropout 0.1 \
  --lr 0.001 \
  --class-weight balanced \
  --decision-threshold-tune f1_val \
  --epochs 30 \
  --patience 7 \
  --batch-size 256 \
  --seed 42 \
  --exclude PP7 PP8 PP11 PP23 \
  --out-csv runs/albaladejo_swell_eda_lstm_loso_w20_s5_v2_seq14.csv
```

---

## 3. Minute-level logistic regression

Dataset-provided **HR, RMSSD, SCL** at 1-minute resolution (no sliding windows).

### Relevant files

| Step | File |
|---|---|
| Preprocess | `scripts/preprocess_swell.py` |
| Benchmark | `scripts/benchmark_swell_minute_level.py` |

**Raw input:** `data/raw/SWELL/3 - Feature dataset/per sensor/D - Physiology features (HR_HRV_SCL - final).csv`

**Outputs:** `data/processed_swell_no_rest/S*.pt`, `runs/swell_minute_logistic_*_loso.csv`, `runs/*_predictions.csv`

### Step 1 — Preprocess minute features

```bash
uv run python scripts/preprocess_swell.py \
  --csv "data/raw/SWELL/3 - Feature dataset/per sensor/D - Physiology features (HR_HRV_SCL - final).csv" \
  --exclude-rest \
  --out-dir data/processed_swell_no_rest
```

### Step 2 — Logistic regression (all features: HR + RMSSD + SCL)

```bash
uv run python scripts/benchmark_swell_minute_level.py \
  --data-root data/processed_swell_no_rest \
  --feature-subset all \
  --penalty l2 \
  --sklearn-C 1.0 \
  --scaler standard \
  --fit-on loso_train \
  --class-weight balanced \
  --decision-threshold-tune f1_val \
  --out-csv runs/swell_minute_simple_loso.csv
```

### Step 2 — Unimodal or pair subsets

Repeat Step 2 with `--feature-subset` set to one of:

`hr`, `rmssd`, `scl`, `hr_rmssd`, `hr_scl`, `rmssd_scl`, `all`

Example (RMSSD only):

```bash
uv run python scripts/benchmark_swell_minute_level.py \
  --data-root data/processed_swell_no_rest \
  --feature-subset rmssd \
  --penalty l2 \
  --sklearn-C 1.0 \
  --scaler standard \
  --fit-on loso_train \
  --class-weight balanced \
  --decision-threshold-tune f1_val
```

If `--out-csv` is omitted, the benchmark picks a default name under `runs/` from the subset and flags (e.g. `swell_minute_logistic_standard_loso_train_rmssd_balanced_loso.csv`).

**Defaults:** L2 logistic regression (`C=1`), `StandardScaler` on LOSO-train minutes, class-balanced loss, F1-tuned decision threshold on validation subjects.

---

## 4. Raw waveform — Linear, GRU, CNN-GRU

One **ECG or EDA** channel per run (never mixed). One window = one classification sample.

### Relevant files

| Step | File |
|---|---|
| Preprocess | `scripts/preprocess_swell_waveform.py` |
| Benchmark | `scripts/benchmark_swell_waveform_loso.py` |
| Models | `src/models/swell_baselines.py` |

**Outputs:** `data/processed_swell_waveform_*/S*.pt`, `runs/<subdir>/swell_waveform_loso_{eda|ecg}_{linear|gru|cnn_gru}.csv`, optional embeddings under `runs/<subdir>/swell_waveform_embeddings/`.

### Step 1 — Build waveform tensors

Example: 20 s window, 15 s hop, drop rest minutes:

```bash
uv run python scripts/preprocess_swell_waveform.py \
  --csv "data/raw/SWELL/3 - Feature dataset/per sensor/D - Physiology features (HR_HRV_SCL - final).csv" \
  --s00-dir "data/raw/SWELL/0 - Raw data/D - Physiology - raw data/Mobi signals (raw and filtered)" \
  --exclude-rest \
  --out-dir data/processed_swell_waveform_w20_h15_exclude_rest \
  --win-sec 20 \
  --hop-sec 15
```

Each `S{n}.pt` stores ECG and EDA waveform windows; the benchmark loads **one modality at a time**.

### Step 2 — Single architecture / modality

```bash
# EDA + GRU
uv run python scripts/benchmark_swell_waveform_loso.py \
  --modality eda \
  --architecture gru \
  --data-root data/processed_swell_waveform_w20_h15_exclude_rest \
  --results-subdir w20_h15_exclude_rest \
  --decision-threshold-tune f1_val \
  --device auto
```

```bash
# ECG + CNN-GRU (legacy conv trunk)
uv run python scripts/benchmark_swell_waveform_loso.py \
  --modality ecg \
  --architecture cnn_gru \
  --cnn-gru-style legacy \
  --conv-channels 16 \
  --hidden-dim 16 \
  --dropout 0.2 \
  --data-root data/processed_swell_waveform_w20_h15_exclude_rest \
  --results-subdir w20_h15_exclude_rest \
  --decision-threshold-tune bacc_val \
  --device auto
```

```bash
# EDA + linear baseline
uv run python scripts/benchmark_swell_waveform_loso.py \
  --modality eda \
  --architecture linear \
  --data-root data/processed_swell_waveform_w20_h15_exclude_rest \
  --results-subdir w20_h15_exclude_rest \
  --device auto
```

### Step 2 — All six runs (EDA/ECG × linear/GRU/CNN-GRU)

```bash
uv run python scripts/benchmark_swell_waveform_loso.py \
  --run-all \
  --skip-existing \
  --data-root data/processed_swell_waveform_w20_h15_exclude_rest \
  --results-subdir w20_h15_exclude_rest \
  --decision-threshold-tune f1_val \
  --device auto
```

EDA-only or ECG-only sweep:

```bash
uv run python scripts/benchmark_swell_waveform_loso.py \
  --run-all \
  --run-all-modality eda \
  --data-root data/processed_swell_waveform_w20_h15_exclude_rest \
  --results-subdir w20_h15_exclude_rest \
  --device auto
```

Quick wiring check (1 epoch, 4 subjects, no embedding files):

```bash
uv run python scripts/benchmark_swell_waveform_loso.py \
  --smoke-test \
  --no-save-embeddings \
  --modality eda \
  --architecture linear \
  --data-root data/processed_swell_waveform_w20_h15_exclude_rest
```

### Raw-waveform default hyperparameters

| Setting | Default |
|---|---|
| Optimizer | Adam, `lr=1e-3`, `weight_decay=1e-4` |
| Batch size | 64 |
| Epochs / patience | 30 / 5 (early stop on val loss) |
| Class weights | none (unweighted CE) |
| GRU | `L=1`, `H=64`, `dropout=0.2` (head), last-pool |
| CNN-GRU (legacy) | 1× Conv1d (16 ch, `k=3`) + ReLU → GRU `H=16` |
| CNN-GRU (physio, current code default) | Conv-BN-ReLU-MaxPool ×2 (`ch=64,64`, `k=7,5`, pool=4) → GRU `H=64` |
| Input norm | per-fold amplitude z-score on train subjects |

Use `--downsample-hz 128` on ECG if windows are too long at native Poly5 rate (~2048 Hz).

---

## Output file naming cheat sheet

| Pattern | Meaning |
|---|---|
| `runs/albaladejo_swell_gru_loso_*.csv` | Per-subject LOSO summary (Albaladejo ECG GRU) |
| `runs/albaladejo_swell_ecg_lstm_loso_*.csv` | ECG LSTM (`--rnn-type lstm`) |
| `runs/albaladejo_swell_eda_gru_loso_*.csv` | EDA GRU |
| `runs/*_predictions.csv` | Row-level `y_true`, `y_pred`, `p_stress` for plotting |
| `runs/swell_minute_logistic_*_loso.csv` | Minute-level LR |
| `runs/<subdir>/swell_waveform_loso_{modality}_{arch}.csv` | Raw waveform LOSO |
| `runs/figures/` | Saved PNG/PDF figures from the notebook |

---

## Related (not covered above)

- `scripts/benchmark_swell_baselines.py` — older unified minute/window baseline entry point; prefer the dedicated scripts above for new runs.
- `scripts/benchmark_swell_waveform_embeddings_loso.py` — sklearn head on saved waveform embeddings.
