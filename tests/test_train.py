from pathlib import Path
from typing import Literal

import pytest
import torch

from src.train import (
    normalized_dcg,
    train,
)
from src.utils.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
    load_config,
    validate_config,
)
from src.utils.train import loss_components


def test_normalized_dcg_rewards_correct_ordering() -> None:
    """ Verify NDCG is maximal when predictions preserve target order."""
    targets = torch.tensor([[1.0, 0.5, 0.0]])
    correct = normalized_dcg(targets, targets)
    reversed_order = normalized_dcg(targets.flip(1), targets)

    torch.testing.assert_close(correct, torch.tensor(1.0))
    assert reversed_order < correct


@pytest.mark.parametrize("checkpointing", [False, True])
@pytest.mark.parametrize("ranking_loss", ["margin", "huber"])
def test_training_logs_and_optionally_writes_checkpoints(
    tmp_path: Path, checkpointing: bool, ranking_loss: Literal["margin", "huber"]
) -> None:
    """ Verify a CPU run always logs and only saves checkpoints when configured.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        checkpointing: Whether the run receives a checkpoint directory.
        ranking_loss: Pairwise objective to exercise throughout training and evaluation.
    """
    movies = {
        f"m{index}": {
            "year_released": 1990 + index,
            "letterboxd_genres": ["Drama"],
            "actors": [f"Actor {index}"],
            "director": ["Director"],
        }
        for index in range(4)
    }
    ratings = {
        f"user-{index}": {
            "movie_ids": list(movies),
            "rating_vals": [2 + index % 3, 5, 8, 10 - index % 3],
        }
        for index in range(10)
    }
    ratings_path = tmp_path / "ratings.pt"
    movies_path = tmp_path / "movies.pt"
    torch.save(ratings, ratings_path)
    torch.save(movies, movies_path)

    config = ExperimentConfig(
        data=DataConfig(
            ratings_path=ratings_path,
            movies_path=movies_path,
            cache_path=tmp_path / "prepared.pt",
            test_ratio=0.2,
            val_ratio=0.2,
            num_id_buckets=16,
            num_actor_buckets=16,
            num_genre_buckets=8,
            num_director_buckets=8,
        ),
        model=ModelConfig(
            feature_size=4,
            model_embed_dim=8,
            num_attention_heads=2,
            num_movie_attention_blocks=1,
            num_self_attention_blocks=1,
            num_cross_attention_blocks=1,
        ),
        training=TrainingConfig(
            epochs=1,
            steps_per_epoch=1,
            validation_steps=1,
            test_steps=1,
            batch_size=2,
            min_context_size=1,
            max_context_size=2,
            min_target_size=2,
            max_target_size=2,
            progress=False,
            seed=7,
            ranking_loss=ranking_loss,
            ranking_huber_delta=0.2,
        ),
        runtime=RuntimeConfig(
            device="cpu",
            log_path=tmp_path / "run" / "train.log",
            checkpoint_dir=(tmp_path / "run" / "checkpoints") if checkpointing else None,
        ),
    )

    metrics = train(config)

    assert set(metrics) == {"loss", "mse", "ranking", "ndcg"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert (tmp_path / "prepared.pt").is_file()
    assert (tmp_path / "run" / "checkpoints" / "best.pt").is_file() is checkpointing
    assert (tmp_path / "run" / "checkpoints" / "last.pt").is_file() is checkpointing
    log_text = (tmp_path / "run" / "train.log").read_text(encoding="utf-8")
    assert "Epoch 1/1" in log_text
    assert "Test | loss=" in log_text
    assert not (tmp_path / "run" / "metrics.jsonl").exists()
    if checkpointing:
        saved = torch.load(
            tmp_path / "run" / "checkpoints" / "best.pt", weights_only=False
        )
        assert saved["config"]["training"]["ranking_loss"] == ranking_loss
        assert saved["config"]["training"]["ranking_huber_delta"] == 0.2


def test_checkpoints_are_disabled_by_default() -> None:
    """ Verify checkpoint persistence requires an explicit directory."""
    assert RuntimeConfig().checkpoint_dir is None


@pytest.mark.parametrize(("objective", "expected_pairwise"), [("margin", 0.0), ("huber", 0.12)])
def test_loss_components_selects_and_weights_pairwise_objective(
    objective: str, expected_pairwise: float
) -> None:
    """ Verify objective selection, configured Huber threshold, and loss weighting.

    Args:
        objective: Selected pairwise loss.
        expected_pairwise: Analytic penalty for the tied target pair.
    """
    predictions = torch.tensor([[0.9, 0.5]], requires_grad=True)
    targets = torch.tensor([[0.5, 0.5]])
    config = TrainingConfig(
        ranking_loss=objective, ranking_huber_delta=0.2, mse_weight=2.0, ranking_weight=0.3
    )
    total, mse, pairwise = loss_components(predictions, targets, config)
    torch.testing.assert_close(mse, torch.tensor(0.08))
    torch.testing.assert_close(pairwise, torch.tensor(expected_pairwise))
    torch.testing.assert_close(total, torch.tensor(0.16 + 0.3 * expected_pairwise))
    total.backward()
    assert torch.isfinite(predictions.grad).all()


@pytest.mark.parametrize(
    "settings",
    [
        {"ranking_loss": "unknown"},
        {"ranking_huber_delta": 0.0},
        {"ranking_huber_delta": -0.1},
        {"ranking_huber_delta": float("nan")},
        {"ranking_huber_delta": float("inf")},
    ],
)
def test_config_rejects_invalid_pairwise_settings(settings: dict[str, str | float]) -> None:
    """ Reject unsupported objectives and invalid Huber thresholds before training.

    Args:
        settings: Invalid pairwise configuration override.
    """
    config = ExperimentConfig(
        DataConfig(Path("ratings.pt"), Path("movies.pt")),
        ModelConfig(),
        TrainingConfig(**settings),
        RuntimeConfig(),
    )
    with pytest.raises(ValueError, match="training.ranking_"):
        validate_config(config)


def test_collab_config_enables_huber_and_legacy_default_is_preserved() -> None:
    """ Verify the collaborative experiment selects Huber without changing old defaults."""
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "train-collab.toml")
    assert config.training.ranking_loss == "huber"
    assert config.training.ranking_huber_delta == 0.1
    assert config.training.mse_weight == 1.0
    assert config.training.ranking_weight == 0.1
    assert TrainingConfig().ranking_loss == "margin"
