from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class DataConfig:
    wesad_root: str
    processed_root: str
    subjects: List[int]
    ecg_hz: int
    dataset: str = "wesad"  # "wesad" | "wearable"
    wearable_root: str = ""
    wearable_subjects: List[str] | None = None  # None = discover from STRESS folder


@dataclass
class PreprocessingConfig:
    window_sec: int
    stride_sec: int
    pre_stress_window_sec: int
    labels_of_interest: List[int]
    feature_profile: str = "default"  # "default" | "cardiomind"


@dataclass
class FeaturesConfig:
    cardiac: List[str]
    somatic: List[str]


@dataclass
class NormalizationConfig:
    method: str


@dataclass
class TCNConfig:
    num_channels: List[int]
    kernel_size: int
    dropout: float
    use_depthwise_separable: bool = False


@dataclass
class MLPConfig:
    hidden_dims: List[int]


@dataclass
class ModelConfig:
    encoder: str  # "tcn" | "se_tcn" | "gru" | "mlp"
    tcn: TCNConfig
    mlp: MLPConfig
    embedding_dim: int
    actor_hidden: int
    critic_hidden: int


@dataclass
class EnvConfig:
    seq_len: int
    consensus: str  # "and" | "or"
    random_start: bool


@dataclass
class RewardConfig:
    tp_reward: float
    tn_reward: float
    fp_penalty: float
    fn_penalty: float
    lead_time_scaling: bool
    lead_time_floor: float
    terminate_on_fp: bool
    fp_cap: float | None  # cap total FP penalty per episode (e.g. -20); None = no cap
    fp_repeat_scale: float = 1.0
    baseline_step_penalty: float = 0.0


@dataclass
class TrainingConfig:
    lr: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    entropy_coef: float
    value_coef: float
    max_grad_norm: float
    ppo_epochs: int
    mini_batch_size: int
    episodes_per_update: int
    total_episodes: int
    seed: int
    checkpoint_dir: str
    log_interval: int
    warm_start_path: str | None = None
    warm_start_mode: str = "encoder"  # "encoder" | "full"


@dataclass
class EvaluationConfig:
    loso: bool
    non_overlapping_test: bool


@dataclass
class Config:
    data: DataConfig
    preprocessing: PreprocessingConfig
    features: FeaturesConfig
    normalization: NormalizationConfig
    model: ModelConfig
    env: EnvConfig
    reward: RewardConfig
    training: TrainingConfig
    evaluation: EvaluationConfig


def _build_nested(cls, raw: dict):
    """Recursively instantiate dataclasses from a raw dict."""
    hints = cls.__dataclass_fields__
    kwargs = {}
    for name, f in hints.items():
        val = raw[name]
        if hasattr(f.type, "__dataclass_fields__") if isinstance(f.type, type) else False:
            val = _build_nested(f.type, val)
        kwargs[name] = val
    return cls(**kwargs)


def load_config(path: str = "configs/default.yaml") -> Config:
    """Load YAML config file and return a typed Config object."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    data_raw = dict(raw.get("data", {}))
    data_raw.setdefault("dataset", "wesad")
    data_raw.setdefault("wearable_root", "")
    data_raw.setdefault("wearable_subjects", None)

    preprocessing_raw = dict(raw["preprocessing"])
    preprocessing_raw.setdefault("feature_profile", "default")

    tcn_raw = dict(raw["model"]["tcn"])
    tcn_raw.setdefault("use_depthwise_separable", False)

    reward_raw = dict(raw["reward"])
    reward_raw.setdefault("fp_repeat_scale", 1.0)
    reward_raw.setdefault("baseline_step_penalty", 0.0)

    training_raw = dict(raw["training"])
    training_raw.setdefault("warm_start_path", None)
    training_raw.setdefault("warm_start_mode", "encoder")

    return Config(
        data=DataConfig(**data_raw),
        preprocessing=PreprocessingConfig(**preprocessing_raw),
        features=FeaturesConfig(**raw["features"]),
        normalization=NormalizationConfig(**raw["normalization"]),
        model=ModelConfig(
            encoder=raw["model"]["encoder"],
            tcn=TCNConfig(**tcn_raw),
            mlp=MLPConfig(**raw["model"]["mlp"]),
            embedding_dim=raw["model"]["embedding_dim"],
            actor_hidden=raw["model"]["actor_hidden"],
            critic_hidden=raw["model"]["critic_hidden"],
        ),
        env=EnvConfig(**raw["env"]),
        reward=RewardConfig(**reward_raw),
        training=TrainingConfig(**training_raw),
        evaluation=EvaluationConfig(**raw["evaluation"]),
    )
