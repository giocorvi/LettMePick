""" Configuration models and loading utilities for training runs."""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class DataConfig:
    """ Describe inputs, splits, hashing, and the prepared-data cache.

    Args:
        ratings_path: Trusted ``torch.save`` file containing user ratings.
        movies_path: Trusted ``torch.save`` file containing movie metadata.
        cache_path: Optional prepared-data cache path.
        test_ratio: Fraction of users reserved for testing.
        val_ratio: Fraction of users reserved for validation.
        num_id_buckets: Number of movie-ID hash buckets.
        num_actor_buckets: Number of actor hash buckets.
        num_genre_buckets: Number of genre hash buckets.
        num_director_buckets: Number of director hash buckets.
    """

    ratings_path: Path
    movies_path: Path
    cache_path: Path | None = None
    test_ratio: float = 0.1
    val_ratio: float = 0.1
    num_id_buckets: int = 10_000
    num_actor_buckets: int = 10_000
    num_genre_buckets: int = 1_000
    num_director_buckets: int = 5_000


@dataclass(frozen=True)
class ModelConfig:
    """ Describe the LettMePick architecture.

    Args:
        feature_size: Width of metadata feature embeddings.
        model_embed_dim: Width of movie and attention representations.
        num_attention_heads: Number of attention heads per block.
        num_movie_attention_blocks: Number of movie-feature attention blocks.
        num_self_attention_blocks: Number of context attention blocks.
        num_cross_attention_blocks: Number of query-to-context attention blocks.
        use_feature_type_embeddings: Whether to add metadata-type embeddings.
    """

    feature_size: int = 32
    model_embed_dim: int = 128
    num_attention_heads: int = 8
    num_movie_attention_blocks: int = 1
    num_self_attention_blocks: int = 2
    num_cross_attention_blocks: int = 2
    use_feature_type_embeddings: bool = True


@dataclass(frozen=True)
class TrainingConfig:
    """ Describe optimization, batch sampling, and evaluation.

    Args:
        epochs: Number of training epochs.
        steps_per_epoch: Sampled batches per epoch.
        validation_steps: Sampled batches per validation pass.
        test_steps: Sampled batches in the final test pass.
        batch_size: Users sampled per batch.
        min_context_size: Smallest requested context length.
        max_context_size: Largest requested context length.
        min_target_size: Smallest requested query length.
        max_target_size: Largest requested query length.
        learning_rate: AdamW learning rate.
        weight_decay: AdamW weight-decay coefficient.
        ranking_weight: Pairwise loss weight.
        mse_weight: Mean squared error weight.
        ranking_margin: Pairwise ranking margin.
        ranking_loss: Pairwise objective: legacy margin ranking or twice Huber
            regression on rating differences (logged as ``ranking``).
        ranking_huber_delta: Positive finite difference-error threshold for Huber.
        max_grad_norm: Optional gradient clipping threshold.
        max_batch_attempts: Maximum invalid-batch retries.
        seed: Python and PyTorch random seed.
        progress: Whether to display progress bars.
    """

    epochs: int = 20
    steps_per_epoch: int = 400
    validation_steps: int = 50
    test_steps: int = 50
    batch_size: int = 32
    min_context_size: int = 8
    max_context_size: int = 64
    min_target_size: int = 2
    max_target_size: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    ranking_weight: float = 1.0
    mse_weight: float = 0.25
    ranking_margin: float = 0.1
    max_grad_norm: float | None = None
    max_batch_attempts: int = 20
    seed: int = 42
    progress: bool = True
    ranking_loss: Literal["margin", "huber"] = "margin"
    ranking_huber_delta: float = 0.1


@dataclass(frozen=True)
class RuntimeConfig:
    """ Describe output and execution settings.

    Args:
        device: PyTorch device name or ``auto``.
        log_path: Optional file that also receives terminal log messages.
        checkpoint_dir: Optional directory for checkpoints; ``None`` disables saving.
        resume_path: Optional checkpoint to resume.
        save_every: Epoch interval for numbered snapshots, or zero to disable.
    """

    device: str = "auto"
    log_path: Path | None = None
    checkpoint_dir: Path | None = None
    resume_path: Path | None = None
    save_every: int = 0


@dataclass(frozen=True)
class ExperimentConfig:
    """ Group all experiment configuration sections.

    Args:
        data: Data preparation settings.
        model: Model architecture settings.
        training: Optimization settings.
        runtime: Execution settings.
    """

    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    runtime: RuntimeConfig


def _path(value: str | None, base: Path) -> Path | None:
    """ Resolve an optional path relative to a configuration directory.

    Args:
        value: Configured path, or ``None``.
        base: Configuration directory.

    Returns:
        Resolved path, or ``None``.
    """
    if value is None:
        return None
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: Path) -> ExperimentConfig:
    """ Load and validate an experiment TOML file.

    Args:
        path: Configuration path.

    Returns:
        Parsed experiment configuration.
    """
    path = path.resolve()
    with path.open("rb") as file:
        raw = tomllib.load(file)
    data = dict(raw.get("data", {}))
    missing = {"ratings_path", "movies_path"} - data.keys()
    if missing:
        raise ValueError(f"Missing required [data] values: {sorted(missing)}")
    data["ratings_path"] = _path(data["ratings_path"], path.parent)
    data["movies_path"] = _path(data["movies_path"], path.parent)
    data["cache_path"] = _path(data.get("cache_path"), path.parent)
    runtime = dict(raw.get("runtime", {}))
    runtime["log_path"] = _path(runtime.get("log_path"), path.parent)
    runtime["checkpoint_dir"] = _path(runtime.get("checkpoint_dir"), path.parent)
    runtime["resume_path"] = _path(runtime.get("resume_path"), path.parent)
    config = ExperimentConfig(
        DataConfig(**data),
        ModelConfig(**raw.get("model", {})),
        TrainingConfig(**raw.get("training", {})),
        RuntimeConfig(**runtime),
    )
    validate_config(config)
    return config


def validate_config(config: ExperimentConfig) -> None:
    """ Reject settings that cannot produce valid training batches.

    Args:
        config: Configuration to validate.
    """
    training = config.training
    required_positive = {
        "epochs": training.epochs,
        "steps_per_epoch": training.steps_per_epoch,
        "validation_steps": training.validation_steps,
        "test_steps": training.test_steps,
        "batch_size": training.batch_size,
        "min_context_size": training.min_context_size,
        "min_target_size": training.min_target_size,
        "max_batch_attempts": training.max_batch_attempts,
    }
    for name, value in required_positive.items():
        if value <= 0:
            raise ValueError(f"training.{name} must be larger than 0")
    if training.max_context_size < training.min_context_size:
        raise ValueError("max_context_size must be at least min_context_size")
    if training.max_target_size < training.min_target_size:
        raise ValueError("max_target_size must be at least min_target_size")
    if training.ranking_loss not in ("margin", "huber"):
        raise ValueError("training.ranking_loss must be 'margin' or 'huber'")
    if not math.isfinite(training.ranking_huber_delta) or training.ranking_huber_delta <= 0:
        raise ValueError("training.ranking_huber_delta must be finite and greater than zero")
    if (
        min(training.ranking_weight, training.mse_weight) < 0
        or training.ranking_weight + training.mse_weight == 0
    ):
        raise ValueError("Loss weights must be non-negative and not both zero")
    if training.ranking_weight and training.min_target_size < 2:
        raise ValueError("Pairwise ranking loss requires min_target_size >= 2")
    if not 0 < config.data.test_ratio < 1 or not 0 < config.data.val_ratio < 1:
        raise ValueError("test_ratio and val_ratio must be between 0 and 1")
    if config.data.test_ratio + config.data.val_ratio >= 1:
        raise ValueError("test_ratio + val_ratio must be smaller than 1")
    if config.runtime.save_every < 0:
        raise ValueError("save_every cannot be negative")
    if config.runtime.save_every and config.runtime.checkpoint_dir is None:
        raise ValueError("save_every requires runtime.checkpoint_dir")
