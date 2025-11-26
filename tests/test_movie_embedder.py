import pytest
import torch

from src.data.movie_embedder import FeatureConfig, MovieEmbedder


def _build_embedder() -> MovieEmbedder:
    configs = [
        FeatureConfig(name="genre", cardinality=3),
        FeatureConfig(name="year_bucket", cardinality=4),
    ]
    embedder = MovieEmbedder(embedding_dim=2, feature_configs=configs)

    embedder.embeddings["genre"].weight.data.copy_(
        torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ]
        )
    )
    embedder.embeddings["year_bucket"].weight.data.copy_(
        torch.tensor(
            [
                [0.5, 0.0],
                [1.5, 0.0],
                [2.5, 0.0],
                [3.5, 0.0],
            ]
        )
    )
    return embedder


def test_movie_embedder_preserves_feature_order() -> None:
    embedder = _build_embedder()
    features = [
        {"genre": torch.tensor(0), "year_bucket": torch.tensor(2)},
        {"genre": torch.tensor(1), "year_bucket": torch.tensor(3)},
    ]

    output = embedder(features)

    assert output.shape == (2, 2, 2)
    torch.testing.assert_close(
        output[0],
        torch.tensor([[1.0, 10.0], [2.5, 0.0]]),
    )
    torch.testing.assert_close(
        output[1],
        torch.tensor([[2.0, 20.0], [3.5, 0.0]]),
    )


def test_missing_feature_fills_with_zeros() -> None:
    embedder = _build_embedder()
    features = [
        {"genre": torch.tensor(2)},
        {},
    ]

    output = embedder(features)

    torch.testing.assert_close(
        output[0],
        torch.tensor([[3.0, 30.0], [0.0, 0.0]]),
    )
    assert torch.all(output[1] == 0)


def test_multi_value_feature_pools_correctly() -> None:
    embedder = _build_embedder()
    features = [
        {"genre": torch.tensor([0, 2]), "year_bucket": torch.tensor(1)},
    ]

    output = embedder(features)

    torch.testing.assert_close(
        output[0, 0],
        torch.tensor([(1.0 + 3.0) / 2, (10.0 + 30.0) / 2]),
    )
    torch.testing.assert_close(
        output[0, 1],
        torch.tensor([1.5, 0.0]),
    )


def test_empty_feature_list_raises_value_error() -> None:
    embedder = _build_embedder()
    with pytest.raises(ValueError):
        embedder([])


def test_numeric_feature_embedding_normalizes_value() -> None:
    configs = [
        FeatureConfig(name="genre", cardinality=3),
        FeatureConfig(name="score", value_range=(0.0, 10.0)),
    ]
    embedder = MovieEmbedder(embedding_dim=2, feature_configs=configs)

    embedder.embeddings["genre"].weight.data.copy_(
        torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
    )

    features = [
        {"genre": torch.tensor(2), "score": torch.tensor(7.5)},
    ]

    output = embedder(features)

    torch.testing.assert_close(output[0, 0], torch.tensor([5.0, 6.0]))
    torch.testing.assert_close(
        output[0, 1],
        torch.full((2,), 0.75),
    )
