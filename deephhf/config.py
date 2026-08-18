from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any
import yaml


@dataclass
class DataConfig:
    target_rate_hz: int = 128
    hours: float = 24
    window_seconds: int = 30
    sequence_step_seconds: int = 120
    standardize_mode: str = "pad_or_crop_center"
    require_ecg_classes: bool = True

    @property
    def samples_per_window(self) -> int:
        return self.target_rate_hz * self.window_seconds

    @property
    def samples_per_recording(self) -> int:
        return int(round(self.target_rate_hz * 60 * 60 * self.hours))


@dataclass
class ModelConfig:
    embedding_dim: int = 128
    encoder_channels: list[int] = field(default_factory=lambda: [32, 64, 96, 128])
    transformer_heads: int = 4
    transformer_layers: int = 3
    transformer_ff_dim: int = 256
    dropout: float = 0.15


@dataclass
class TrainingConfig:
    batch_size: int = 32
    sequence_batch_size: int = 2
    encoder_lr: float = 1e-3
    sequence_lr: float = 5e-5
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 7
    num_workers: int = 0
    amp: bool = True


@dataclass
class PathsConfig:
    output_dir: str = "outputs"


@dataclass
class Config:
    seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path | None = None) -> Config:
    if path is None:
        return Config()
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    return config_from_dict(raw)


def config_from_dict(raw: dict[str, Any]) -> Config:
    return Config(
        seed=raw.get("seed", 42),
        data=DataConfig(**raw.get("data", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        paths=PathsConfig(**raw.get("paths", {})),
    )
