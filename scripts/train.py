"""Basic training loop for the movie recommendation model.

Usage:
    python scripts/train.py configs/example_train_config.toml
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import torch

from src.data.data import get_train_batch, get_user_ratings_dict
from src.models.lett_me_pick import LettMePick

# Prefer stdlib tomllib (Python 3.11+) and fallback to tomli if available.
try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError as inner_exc:  # pragma: no cover - runtime guard
        raise RuntimeError(
            "tomllib (Python 3.11+) or tomli is required to load TOML configs."
        ) from inner_exc


@dataclasses.dataclass
class TrainConfig:
    dataset_path: Path
    ratings_path: Path
    device: str = "cpu"
    normalization_divisor: float = 10.0
    hidden_size: int = 64
    output_size: int = 32
    num_layers: int = 2
    num_attention_heads: int = 4
    learning_rate: float = 1e-3
    context_size: int = 64
    target_size: int = 8
    epochs: int = 3
    steps_per_epoch: int = 100

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "TrainConfig":
        """Create a TrainConfig from a TOML dictionary."""
        cfg = raw.copy()
        cfg["dataset_path"] = Path(cfg["dataset_path"])
        cfg["ratings_path"] = Path(cfg["ratings_path"])
        return cls(**cfg)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    # Support top-level table or nested under "train"
    if "train" in data:
        data = data["train"]
    return data


def build_normalization_fn(divisor: float) -> Callable[[float], torch.Tensor]:
    """Normalize raw rating values with a divisor (e.g., 10.0)."""
    return lambda x: torch.tensor(x / divisor, dtype=torch.float32)


def log_parameter_counts(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(
        f"Model parameters (total/trainable/frozen): "
        f"{total_params:,} / {trainable_params:,} / {frozen_params:,}"
    )


def train(config_path: Path) -> None:
    raw_config = load_config(config_path)
    config = TrainConfig.from_dict(raw_config)

    device = torch.device(config.device)
    print(f"Using device: {device}")

    dataset_path = config.dataset_path
    ratings_path = config.ratings_path

    print(f"Loading dataset from {dataset_path}...")
    dataset = torch.load(dataset_path, map_location="cpu")
    movie_embeddings = dataset["data"]  # keep on CPU; batches move to device
    movie_ids = dataset["ids"]
    num_features = movie_embeddings.shape[1]
    embedding_dim = movie_embeddings.shape[2]

    print(f"Loaded embeddings with shape: {movie_embeddings.shape}")

    print(f"Reading ratings from {ratings_path}...")
    user_ratings_dict = torch.load(ratings_path, map_location="cpu")

    model = LettMePick(
        data_embed_dim=embedding_dim,
        num_features=num_features,
        model_embed_dim=int(config.output_size),
        num_attention_heads=int(config.num_attention_heads),
        encoder_hidden_size=int(config.hidden_size),
        encoder_num_layers=int(config.num_layers),
    ).to(device)

    log_parameter_counts(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))

    context_size = int(config.context_size)
    target_size = int(config.target_size)
    epochs = int(config.epochs)
    steps_per_epoch = int(config.steps_per_epoch)

    print("Starting training loop...")
    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        steps_completed = 0

        for _ in range(steps_per_epoch):
            # get_train_batch can raise if a sampled user has too few ratings; retry a few times
            batch = None
            for _ in range(5):
                try:
                    batch = get_train_batch(
                        user_ratings_dict,
                        movie_embeddings,
                        context_size=context_size,
                        target_size=target_size,
                        device=device,
                    )
                    break
                except ValueError:
                    continue
            if batch is None:
                continue

            train_movie_embeddings, train_movies_scores, pred_movie_embeddings, pred_movies_scores = batch

            predictions = model(
                context_embed=train_movie_embeddings,
                query_embed=pred_movie_embeddings,
                context_scores=train_movies_scores,
            )
            loss = model.compute_loss(predictions, pred_movies_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps_completed += 1

        if steps_completed:
            avg_loss = epoch_loss / steps_completed
            print(f"Epoch {epoch}/{epochs} - steps: {steps_completed}, avg_loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch}/{epochs} - no valid batches; check data/config.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple recommendation model.")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to TOML config file with training hyperparameters.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    train(args.config)
