from pathlib import Path

import pytest
import torch

from src.train import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
    normalized_dcg,
    train,
)


def test_normalized_dcg_rewards_correct_ordering() -> None:
    """ Verify NDCG is maximal when predictions preserve target order."""
    targets = torch.tensor([[1.0, 0.5, 0.0]])
    correct = normalized_dcg(targets, targets)
    reversed_order = normalized_dcg(targets.flip(1), targets)

    torch.testing.assert_close(correct, torch.tensor(1.0))
    assert reversed_order < correct


@pytest.mark.parametrize("checkpointing", [False, True])
def test_training_logs_and_optionally_writes_checkpoints(
    tmp_path: Path, checkpointing: bool
) -> None:
    """ Verify a CPU run always logs and only saves checkpoints when configured.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        checkpointing: Whether the run receives a checkpoint directory.
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


def test_checkpoints_are_disabled_by_default() -> None:
    """ Verify checkpoint persistence requires an explicit directory."""
    assert RuntimeConfig().checkpoint_dir is None
