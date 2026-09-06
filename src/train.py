""" Train LettMePick from a reproducible TOML experiment configuration."""

# TODO: Add optional automatic mixed-precision training after its behavior is tested.

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from loguru import logger
from tqdm import tqdm

# Direct script execution does not otherwise expose the repository's ``src`` package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.data import split_user_ratings_dict
from src.utils.config import ExperimentConfig, RuntimeConfig, load_config, validate_config
from src.utils.train import (
    build_model,
    checkpoint_payload,
    configure_logger,
    copy_model_state,
    eligible_users,
    evaluate,
    loss_components,
    normalized_dcg,
    prepare_training_data,
    resolve_device,
    resume_training,
    sample_batch,
    save_torch,
)


def train(config: ExperimentConfig) -> dict[str, float]:
    """ Train, select a checkpoint, and perform final testing.

    Args:
        config: Validated experiment settings.

    Returns:
        Metrics from the validation-selected model state on the test split.
    """
    validate_config(config)
    configure_logger(config.runtime.log_path)
    device = resolve_device(config.runtime.device)
    random.seed(config.training.seed)
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)

    ratings, bank = prepare_training_data(config.data)
    train_users, val_users, test_users = split_user_ratings_dict(
        ratings,
        config.data.test_ratio,
        config.data.val_ratio,
        config.training.seed,
    )
    minimum = config.training.min_context_size + config.training.max_target_size
    train_users = eligible_users(train_users, minimum, "training")
    val_users = eligible_users(val_users, minimum, "validation")
    test_users = eligible_users(test_users, minimum, "test")

    model = build_model(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start_epoch, best_validation_loss = 0, float("inf")
    best_model_state = None
    if config.runtime.resume_path:
        start_epoch, best_validation_loss, best_model_state = resume_training(
            config.runtime.resume_path, model, optimizer, device
        )
    logger.info(
        "Training on {}: {} train / {} validation / {} test users",
        device,
        len(train_users),
        len(val_users),
        len(test_users),
    )

    for epoch in range(start_epoch, config.training.epochs):
        model.train()
        totals = {key: 0.0 for key in ("loss", "mse", "ranking", "ndcg")}
        progress = tqdm(
            range(config.training.steps_per_epoch),
            desc=f"epoch {epoch + 1}/{config.training.epochs}",
            disable=not config.training.progress,
        )
        for _ in progress:
            rated, scores, query, targets = sample_batch(
                train_users, bank, config.training, device
            )
            optimizer.zero_grad(set_to_none=True)
            predictions = model(rated, query, scores)
            loss, mse, ranking = loss_components(
                predictions, targets, config.training
            )
            loss.backward()
            if config.training.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.training.max_grad_norm
                )
            optimizer.step()

            values = {
                "loss": loss,
                "mse": mse,
                "ranking": ranking,
                "ndcg": normalized_dcg(predictions.detach(), targets),
            }
            for key, value in values.items():
                totals[key] += value.detach().item()
            progress.set_postfix(loss=f"{loss.detach().item():.4f}")

        train_metrics = {
            key: value / config.training.steps_per_epoch
            for key, value in totals.items()
        }
        validation_metrics = evaluate(
            model,
            val_users,
            bank,
            config.training,
            device,
            config.training.validation_steps,
            config.training.seed + 10_000,
        )
        improved = validation_metrics["loss"] < best_validation_loss
        if improved:
            best_validation_loss = validation_metrics["loss"]
            best_model_state = copy_model_state(model)

        # Persist state only when checkpoint storage was explicitly configured.
        if config.runtime.checkpoint_dir is not None:
            checkpoint_dir = config.runtime.checkpoint_dir
            assert best_model_state is not None
            saved = checkpoint_payload(
                model,
                optimizer,
                config,
                epoch,
                best_validation_loss,
                best_model_state,
            )
            save_torch(saved, checkpoint_dir / "last.pt")
            if improved:
                save_torch(saved, checkpoint_dir / "best.pt")
            if (
                config.runtime.save_every
                and (epoch + 1) % config.runtime.save_every == 0
            ):
                save_torch(saved, checkpoint_dir / f"epoch-{epoch + 1:04d}.pt")

        logger.info(
            "Epoch {}/{} | train: loss={:.4f}, mse={:.4f}, ranking={:.4f}, ndcg={:.4f} "
            "| validation: loss={:.4f}, mse={:.4f}, ranking={:.4f}, ndcg={:.4f}",
            epoch + 1,
            config.training.epochs,
            train_metrics["loss"],
            train_metrics["mse"],
            train_metrics["ranking"],
            train_metrics["ndcg"],
            validation_metrics["loss"],
            validation_metrics["mse"],
            validation_metrics["ranking"],
            validation_metrics["ndcg"],
        )

    # Evaluate only once on test users, using the in-memory validation selection.
    if best_model_state is None:
        best_model_state = copy_model_state(model)
        logger.warning("No epochs ran; evaluating the currently loaded model")
    model.load_state_dict(best_model_state)
    metrics = evaluate(
        model,
        test_users,
        bank,
        config.training,
        device,
        config.training.test_steps,
        config.training.seed + 20_000,
    )
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
    train(
        ExperimentConfig(
            config.data,
            config.model,
            config.training,
            RuntimeConfig(**runtime),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
