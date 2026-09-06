""" Reusable training helpers for LettMePick experiments."""

from __future__ import annotations

import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor

from src.data.data import get_train_batch, prepare_dataset
from src.losses import (
    mean_squared_error_loss,
    pairwise_huber_loss,
    pairwise_margin_ranking_loss,
)
from src.models.lett_me_pick import LettMePick
from src.utils.config import DataConfig, ExperimentConfig, TrainingConfig


def resolve_device(name: str) -> torch.device:
    """ Resolve an automatic or explicit PyTorch device.

    Args:
        name: Device name or ``auto``.

    Returns:
        Available PyTorch device.
    """
    if name == "auto":
        name = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def save_torch(value: Any, path: Path) -> None:
    """ Atomically save a PyTorch object.

    Args:
        value: Serializable value.
        path: Destination path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _signature(path: Path) -> dict[str, str | int]:
    """ Describe a source file for cache invalidation.

    Args:
        path: Source path.

    Returns:
        Path, size, and modification time.
    """
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def prepare_training_data(config: DataConfig) -> tuple[dict, dict[str, Tensor]]:
    """ Load or prepare normalized ratings and a hashed movie bank.

    Args:
        config: Data settings.

    Returns:
        Prepared ratings and movie bank.
    """
    sources = {
        "ratings": _signature(config.ratings_path),
        "movies": _signature(config.movies_path),
    }
    buckets = {
        key: getattr(config, f"num_{key}_buckets")
        for key in ("id", "actor", "genre", "director")
    }
    if config.cache_path and config.cache_path.exists():
        cached = torch.load(config.cache_path, map_location="cpu", weights_only=False)
        if (
            isinstance(cached, dict)
            and cached.get("version") == 1
            and cached.get("sources") == sources
            and cached.get("buckets") == buckets
        ):
            logger.info("Loaded prepared data from {}", config.cache_path)
            return cached["ratings"], cached["movie_bank"]

    ratings = torch.load(config.ratings_path, map_location="cpu", weights_only=False)
    movies = torch.load(config.movies_path, map_location="cpu", weights_only=False)
    if not isinstance(ratings, dict) or not isinstance(movies, dict):
        raise TypeError("ratings_path and movies_path must contain dictionaries")

    # Preparation normalizes scores, filters unavailable movies, and remaps IDs in place.
    logger.info("Preparing ratings and movie metadata")
    bank = prepare_dataset(
        ratings,
        movies,
        **{f"num_{key}_buckets": value for key, value in buckets.items()},
        verbose=False,
    )
    if config.cache_path:
        save_torch(
            {
                "version": 1,
                "sources": sources,
                "buckets": buckets,
                "ratings": ratings,
                "movie_bank": bank,
            },
            config.cache_path,
        )
        logger.info("Stored prepared data in {}", config.cache_path)
    return ratings, bank


def eligible_users(ratings: dict, minimum: int, name: str) -> dict:
    """ Keep users able to produce the configured minimum batch shape.

    Args:
        ratings: Ratings from one split.
        minimum: Required rating count.
        name: Split name for errors.

    Returns:
        Eligible user ratings.
    """
    result = {
        user: values
        for user, values in ratings.items()
        if len(values.get("movie_ids", [])) >= minimum
    }
    if not result:
        raise ValueError(f"The {name} split has no users with at least {minimum} ratings")
    return result


def build_model(config: ExperimentConfig) -> LettMePick:
    """ Construct LettMePick from experiment settings.

    Args:
        config: Experiment settings.

    Returns:
        Initialized model.
    """
    values = asdict(config.model)
    values.update(
        {
            f"num_{key}_buckets": getattr(config.data, f"num_{key}_buckets")
            for key in ("id", "actor", "genre", "director")
        }
    )
    return LettMePick(**values)


def sample_batch(
    ratings: dict,
    bank: dict[str, Tensor],
    config: TrainingConfig,
    device: torch.device,
) -> tuple:
    """ Sample a variable-shape batch with bounded retries.

    Args:
        ratings: Eligible user ratings.
        bank: Prepared movie bank.
        config: Sampling settings.
        device: Batch destination device.

    Returns:
        Rated movies, rated scores, query movies, and targets.
    """
    error = None
    for _ in range(config.max_batch_attempts):
        try:
            return get_train_batch(
                ratings,
                bank,
                batch_size=config.batch_size,
                context_size=random.randint(
                    config.min_context_size, config.max_context_size
                ),
                target_size=random.randint(
                    config.min_target_size, config.max_target_size
                ),
                device=device,
            )
        except ValueError as caught:
            error = caught
    raise RuntimeError("Unable to sample a valid batch") from error


def loss_components(
    predictions: Tensor, targets: Tensor, config: TrainingConfig
) -> tuple[Tensor, Tensor, Tensor]:
    """ Compute the weighted objective and unweighted components.

    Args:
        predictions: Predicted ratings.
        targets: Ground-truth ratings.
        config: Loss settings.

    Returns:
        Total, MSE, and the selected pairwise loss (logged as ``ranking``).
    """
    mse = mean_squared_error_loss(predictions, targets)
    if config.ranking_loss == "huber":
        ranking = pairwise_huber_loss(
            predictions, targets, delta=config.ranking_huber_delta
        )
    elif config.ranking_loss == "margin":
        ranking = pairwise_margin_ranking_loss(
            predictions, targets, margin=config.ranking_margin
        )
    else:
        raise ValueError("ranking_loss must be 'margin' or 'huber'")
    return config.mse_weight * mse + config.ranking_weight * ranking, mse, ranking


def normalized_dcg(predictions: Tensor, targets: Tensor) -> Tensor:
    """ Compute mean normalized discounted cumulative gain.

    Args:
        predictions: Predicted relevance shaped ``(batch, query)``.
        targets: Ground-truth relevance with the same shape.

    Returns:
        Scalar mean NDCG.
    """
    discounts = torch.log2(
        torch.arange(
            2,
            targets.shape[1] + 2,
            device=targets.device,
            dtype=targets.dtype,
        )
    )
    gains = torch.exp2(targets) - 1
    dcg = gains.gather(1, predictions.argsort(1, descending=True)).div(discounts).sum(1)
    ideal = gains.gather(1, targets.argsort(1, descending=True)).div(discounts).sum(1)
    return torch.where(ideal > 0, dcg / ideal, torch.zeros_like(dcg)).mean()


def evaluate(
    model: LettMePick,
    ratings: dict,
    bank: dict[str, Tensor],
    config: TrainingConfig,
    device: torch.device,
    steps: int,
    seed: int,
) -> dict[str, float]:
    """ Evaluate repeatably over sampled batches.

    Args:
        model: Model to evaluate.
        ratings: Validation or test ratings.
        bank: Prepared movie bank.
        config: Sampling and loss settings.
        device: Model device.
        steps: Number of batches.
        seed: Temporary sampling seed.

    Returns:
        Mean objective, MSE, ranking loss, and NDCG.
    """
    totals = {key: 0.0 for key in ("loss", "mse", "ranking", "ndcg")}
    state = random.getstate()
    random.seed(seed)
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(steps):
                rated, scores, query, targets = sample_batch(
                    ratings, bank, config, device
                )
                predictions = model(rated, query, scores)
                loss, mse, ranking = loss_components(predictions, targets, config)
                for key, value in (
                    ("loss", loss),
                    ("mse", mse),
                    ("ranking", ranking),
                    ("ndcg", normalized_dcg(predictions, targets)),
                ):
                    totals[key] += value.item()
    finally:
        random.setstate(state)
    return {key: value / steps for key, value in totals.items()}


def copy_model_state(model: LettMePick) -> dict[str, Tensor]:
    """ Copy model parameters and buffers to CPU for best-model selection.

    Args:
        model: Model whose state should be copied.

    Returns:
        Detached CPU copy of the model state.
    """
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def checkpoint_payload(
    model: LettMePick,
    optimizer: torch.optim.Optimizer,
    config: ExperimentConfig,
    epoch: int,
    best: float,
    best_model_state: dict[str, Tensor],
) -> dict[str, Any]:
    """ Assemble resumable training state.

    Args:
        model: Trained model.
        optimizer: Active optimizer.
        config: Experiment settings.
        epoch: Completed epoch index.
        best: Best validation loss.
        best_model_state: Validation-selected parameters retained across resumes.

    Returns:
        Serializable checkpoint.
    """
    result: dict[str, Any] = {
        "version": 2,
        "epoch": epoch,
        "best_validation_loss": best,
        "best_model_state_dict": best_model_state,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        result["cuda_random_state"] = torch.cuda.get_rng_state_all()
    return result


def resume_training(
    path: Path,
    model: LettMePick,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, float, dict[str, Tensor] | None]:
    """ Restore training state from a trusted checkpoint.

    Args:
        path: Checkpoint path.
        model: Model receiving parameters.
        optimizer: Optimizer receiving state.
        device: Tensor destination device.

    Returns:
        Next epoch, best validation loss, and optional best model state.
    """
    saved = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(saved["model_state_dict"])
    optimizer.load_state_dict(saved["optimizer_state_dict"])
    random.setstate(saved["python_random_state"])
    torch.set_rng_state(saved["torch_random_state"].cpu())
    if device.type == "cuda" and "cuda_random_state" in saved:
        torch.cuda.set_rng_state_all(saved["cuda_random_state"])
    best_model_state = saved.get("best_model_state_dict")
    if best_model_state is None:
        logger.warning(
            "Checkpoint has no best-model state; validation selection restarts"
        )
        return saved["epoch"] + 1, float("inf"), None
    return saved["epoch"] + 1, saved["best_validation_loss"], best_model_state


def configure_logger(log_path: Path | None) -> None:
    """ Configure mandatory terminal logging and an optional file sink.

    Args:
        log_path: Optional path receiving a copy of all log messages.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=sys.stderr.isatty())
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, level="INFO", encoding="utf-8")
