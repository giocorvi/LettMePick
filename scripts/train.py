""" Train LettMePick from a reproducible TOML experiment configuration."""

# TODO: Add optional automatic mixed-precision training after its behavior is tested.

from __future__ import annotations

import argparse
import random
import sys
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import Tensor
from tqdm import tqdm

# Direct script execution does not otherwise expose the repository's ``src`` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data import get_train_batch, prepare_dataset, split_user_ratings_dict
from src.losses import mean_squared_error_loss, pairwise_margin_ranking_loss
from src.models.lett_me_pick import LettMePick


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
    _validate_config(config)
    return config


def _validate_config(config: ExperimentConfig) -> None:
    """ Reject settings that cannot produce valid training batches.

    Args:
        config: Configuration to validate.
    """
    train = config.training
    required_positive = {
        "epochs": train.epochs, "steps_per_epoch": train.steps_per_epoch,
        "validation_steps": train.validation_steps, "test_steps": train.test_steps,
        "batch_size": train.batch_size, "min_context_size": train.min_context_size,
        "min_target_size": train.min_target_size, "max_batch_attempts": train.max_batch_attempts,
    }
    for name, value in required_positive.items():
        if value <= 0:
            raise ValueError(f"training.{name} must be larger than 0")
    if train.max_context_size < train.min_context_size:
        raise ValueError("max_context_size must be at least min_context_size")
    if train.max_target_size < train.min_target_size:
        raise ValueError("max_target_size must be at least min_target_size")
    if min(train.ranking_weight, train.mse_weight) < 0 or train.ranking_weight + train.mse_weight == 0:
        raise ValueError("Loss weights must be non-negative and not both zero")
    if train.ranking_weight and train.min_target_size < 2:
        raise ValueError("Pairwise ranking loss requires min_target_size >= 2")
    if not 0 < config.data.test_ratio < 1 or not 0 < config.data.val_ratio < 1:
        raise ValueError("test_ratio and val_ratio must be between 0 and 1")
    if config.data.test_ratio + config.data.val_ratio >= 1:
        raise ValueError("test_ratio + val_ratio must be smaller than 1")
    if config.runtime.save_every < 0:
        raise ValueError("save_every cannot be negative")
    if config.runtime.save_every and config.runtime.checkpoint_dir is None:
        raise ValueError("save_every requires runtime.checkpoint_dir")


def _device(name: str) -> torch.device:
    """ Resolve an automatic or explicit PyTorch device.

    Args:
        name: Device name or ``auto``.

    Returns:
        Available PyTorch device.
    """
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return device


def _save(value: Any, path: Path) -> None:
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


def _prepare(config: DataConfig) -> tuple[dict, dict[str, Tensor]]:
    """ Load or prepare normalized ratings and a hashed movie bank.

    Args:
        config: Data settings.

    Returns:
        Prepared ratings and movie bank.
    """
    sources = {"ratings": _signature(config.ratings_path), "movies": _signature(config.movies_path)}
    buckets = {key: getattr(config, f"num_{key}_buckets") for key in ("id", "actor", "genre", "director")}
    if config.cache_path and config.cache_path.exists():
        cached = torch.load(config.cache_path, map_location="cpu", weights_only=False)
        if isinstance(cached, dict) and cached.get("version") == 1 and cached.get("sources") == sources and cached.get("buckets") == buckets:
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
        _save({"version": 1, "sources": sources, "buckets": buckets, "ratings": ratings, "movie_bank": bank}, config.cache_path)
        logger.info("Stored prepared data in {}", config.cache_path)
    return ratings, bank


def _eligible(ratings: dict, minimum: int, name: str) -> dict:
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


def _model(config: ExperimentConfig) -> LettMePick:
    """ Construct LettMePick from experiment settings.

    Args:
        config: Experiment settings.

    Returns:
        Initialized model.
    """
    values = asdict(config.model)
    values.update({f"num_{key}_buckets": getattr(config.data, f"num_{key}_buckets") for key in ("id", "actor", "genre", "director")})
    return LettMePick(**values)


def _batch(ratings: dict, bank: dict[str, Tensor], config: TrainingConfig, device: torch.device) -> tuple:
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
                ratings, bank, batch_size=config.batch_size,
                context_size=random.randint(config.min_context_size, config.max_context_size),
                target_size=random.randint(config.min_target_size, config.max_target_size), device=device,
            )
        except ValueError as caught:
            error = caught
    raise RuntimeError("Unable to sample a valid batch") from error


def _loss(predictions: Tensor, targets: Tensor, config: TrainingConfig) -> tuple[Tensor, Tensor, Tensor]:
    """ Compute the weighted objective and unweighted components.

    Args:
        predictions: Predicted ratings.
        targets: Ground-truth ratings.
        config: Loss settings.

    Returns:
        Total, MSE, and pairwise ranking losses.
    """
    mse = mean_squared_error_loss(predictions, targets)
    ranking = pairwise_margin_ranking_loss(predictions, targets, margin=config.ranking_margin)
    return config.mse_weight * mse + config.ranking_weight * ranking, mse, ranking


def normalized_dcg(predictions: Tensor, targets: Tensor) -> Tensor:
    """ Compute mean normalized discounted cumulative gain.

    Args:
        predictions: Predicted relevance shaped ``(batch, query)``.
        targets: Ground-truth relevance with the same shape.

    Returns:
        Scalar mean NDCG.
    """
    discounts = torch.log2(torch.arange(2, targets.shape[1] + 2, device=targets.device, dtype=targets.dtype))
    gains = torch.exp2(targets) - 1
    dcg = gains.gather(1, predictions.argsort(1, descending=True)).div(discounts).sum(1)
    ideal = gains.gather(1, targets.argsort(1, descending=True)).div(discounts).sum(1)
    return torch.where(ideal > 0, dcg / ideal, torch.zeros_like(dcg)).mean()


def _evaluate(model: LettMePick, ratings: dict, bank: dict[str, Tensor], config: TrainingConfig, device: torch.device, steps: int, seed: int) -> dict[str, float]:
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
                rated, scores, query, targets = _batch(ratings, bank, config, device)
                predictions = model(rated, query, scores)
                loss, mse, ranking = _loss(predictions, targets, config)
                for key, value in (("loss", loss), ("mse", mse), ("ranking", ranking), ("ndcg", normalized_dcg(predictions, targets))):
                    totals[key] += value.item()
    finally:
        random.setstate(state)
    return {key: value / steps for key, value in totals.items()}


def _copy_model_state(model: LettMePick) -> dict[str, Tensor]:
    """ Copy model parameters and buffers to CPU for best-model selection.

    Args:
        model: Model whose state should be copied.

    Returns:
        Detached CPU copy of the model state.
    """
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def _checkpoint(
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
    result = {
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


def _resume(
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


def _configure_logger(log_path: Path | None) -> None:
    """ Configure mandatory terminal logging and an optional file sink.

    Args:
        log_path: Optional path receiving a copy of all log messages.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO", colorize=sys.stderr.isatty())
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_path, level="INFO", encoding="utf-8")


def train(config: ExperimentConfig) -> dict[str, float]:
    """ Train, select a checkpoint, and perform final testing.

    Args:
        config: Validated experiment settings.

    Returns:
        Metrics from the validation-selected model state on the test split.
    """
    _validate_config(config)
    _configure_logger(config.runtime.log_path)
    device = _device(config.runtime.device)
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    ratings, bank = _prepare(config.data)
    train_users, val_users, test_users = split_user_ratings_dict(
        ratings,
        config.data.test_ratio,
        config.data.val_ratio,
        config.training.seed,
    )
    minimum = config.training.min_context_size + config.training.max_target_size
    train_users = _eligible(train_users, minimum, "training")
    val_users = _eligible(val_users, minimum, "validation")
    test_users = _eligible(test_users, minimum, "test")

    model = _model(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
    start, best = 0, float("inf")
    best_model_state: dict[str, Tensor] | None = None
    if config.runtime.resume_path:
        start, best, best_model_state = _resume(
            config.runtime.resume_path, model, optimizer, device
        )
    logger.info(
        "Training on {}: {} train / {} validation / {} test users",
        device,
        len(train_users),
        len(val_users),
        len(test_users),
    )

    for epoch in range(start, config.training.epochs):
        model.train()
        totals = {key: 0.0 for key in ("loss", "mse", "ranking", "ndcg")}
        progress = tqdm(range(config.training.steps_per_epoch), desc=f"epoch {epoch + 1}/{config.training.epochs}", disable=not config.training.progress)
        for _ in progress:
            rated, scores, query, targets = _batch(train_users, bank, config.training, device)
            optimizer.zero_grad(set_to_none=True)
            predictions = model(rated, query, scores)
            loss, mse, ranking = _loss(predictions, targets, config.training)
            loss.backward()
            if config.training.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            values = {"loss": loss, "mse": mse, "ranking": ranking, "ndcg": normalized_dcg(predictions.detach(), targets)}
            for key, value in values.items():
                totals[key] += value.detach().item()
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")

        train_metrics = {key: value / config.training.steps_per_epoch for key, value in totals.items()}
        val_metrics = _evaluate(model, val_users, bank, config.training, device, config.training.validation_steps, config.training.seed + 10_000)
        improved = val_metrics["loss"] < best
        if improved:
            best = val_metrics["loss"]
            best_model_state = _copy_model_state(model)

        # Persist state only when checkpoint storage was explicitly configured.
        if config.runtime.checkpoint_dir is not None:
            checkpoint_dir = config.runtime.checkpoint_dir
            assert best_model_state is not None
            saved = _checkpoint(
                model, optimizer, config, epoch, best, best_model_state
            )
            _save(saved, checkpoint_dir / "last.pt")
            if improved:
                _save(saved, checkpoint_dir / "best.pt")
            if config.runtime.save_every and (epoch + 1) % config.runtime.save_every == 0:
                _save(saved, checkpoint_dir / f"epoch-{epoch + 1:04d}.pt")

        logger.info(
            "Epoch {}/{} | train: loss={:.4f}, mse={:.4f}, ranking={:.4f}, ndcg={:.4f} "
            "| validation: loss={:.4f}, mse={:.4f}, ranking={:.4f}, ndcg={:.4f}",
            epoch + 1,
            config.training.epochs,
            train_metrics["loss"],
            train_metrics["mse"],
            train_metrics["ranking"],
            train_metrics["ndcg"],
            val_metrics["loss"],
            val_metrics["mse"],
            val_metrics["ranking"],
            val_metrics["ndcg"],
        )

    # Evaluate only once on test users, using the in-memory validation selection.
    if best_model_state is None:
        best_model_state = _copy_model_state(model)
        logger.warning("No epochs ran; evaluating the currently loaded model")
    model.load_state_dict(best_model_state)
    metrics = _evaluate(model, test_users, bank, config.training, device, config.training.test_steps, config.training.seed + 20_000)
    logger.info(
        "Test | loss={:.4f}, mse={:.4f}, ranking={:.4f}, ndcg={:.4f}",
        metrics["loss"],
        metrics["mse"],
        metrics["ranking"],
        metrics["ndcg"],
    )
    return metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """ Parse command-line arguments.

    Args:
        argv: Optional argument list.

    Returns:
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", help="override runtime.device")
    parser.add_argument("--resume", type=Path, help="override runtime.resume_path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """ Load CLI overrides and run training.

    Args:
        argv: Optional command-line arguments.

    Returns:
        Process exit status.
    """
    args = _parse_args(argv)
    config = load_config(args.config)
    runtime = asdict(config.runtime)
    if args.device:
        runtime["device"] = args.device
    if args.resume:
        runtime["resume_path"] = args.resume.resolve()
    train(ExperimentConfig(config.data, config.model, config.training, RuntimeConfig(**runtime)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
