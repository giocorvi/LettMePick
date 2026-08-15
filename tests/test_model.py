import pytest
import torch

from src.data.data import Movie
from src.data.prehash import (
    build_prehashed_bank,
    collate_prehashed_bank,
    prehash_movies,
)
from src.models.lett_me_pick import LettMePick
from src.models.movie_encoder import MovieEncoder


def _bank() -> dict[str, torch.Tensor]:
    """ Build a representative hashed movie bank for model tests.

    Args:
        None.

    Returns:
        Hashed feature tensors for four movies.
    """
    movies = [
        Movie(
            id=f"m{index}",
            year=1990 + index,
            genres=["drama"] if index % 2 else ["comedy", "action"],
            actors=[f"actor-{index}"],
            directors=[f"director-{index % 2}"],
        )
        for index in range(4)
    ]
    return build_prehashed_bank(
        prehash_movies(
            movies,
            num_id_buckets=32,
            num_actor_buckets=16,
            num_genre_buckets=8,
            num_director_buckets=8,
        )
    )


def _model(score_embedding_mode: str = "fusion") -> LettMePick:
    """ Construct a compact recommendation model for tests.

    Args:
        score_embedding_mode: Rating integration mode to configure.

    Returns:
        An initialized recommendation model.
    """
    return LettMePick(
        feature_size=4,
        prefix_size=4,
        num_id_buckets=32,
        num_actor_buckets=16,
        num_genre_buckets=8,
        num_director_buckets=8,
        model_embed_dim=8,
        num_attention_heads=2,
        score_embedding_mode=score_embedding_mode,
    )


def test_movie_encoder_returns_tokens_and_mask() -> None:
    """ Verify the encoder returns aligned tokens and validity masks.

    Args:
        None.

    Returns:
        None.
    """
    batch = collate_prehashed_bank([[0, 1]], _bank())
    encoder = MovieEncoder(
        feature_size=4,
        prefix_size=4,
        num_id_buckets=32,
        num_actor_buckets=16,
        num_genre_buckets=8,
        num_director_buckets=8,
    )

    embeddings, mask = encoder(batch)

    assert embeddings.shape[:2] == mask.shape
    assert embeddings.shape[2] == 8
    assert mask.dtype == torch.bool
    assert torch.all(mask[:, :2])


@pytest.mark.parametrize("score_embedding_mode", ["fusion", "concat"])
def test_model_scores_hashed_movies(score_embedding_mode: str) -> None:
    """ Verify each score integration mode predicts bounded ratings.

    Args:
        score_embedding_mode: Parameterized rating integration mode.

    Returns:
        None.
    """
    bank = _bank()
    context = collate_prehashed_bank([[0, 1]], bank)
    query = collate_prehashed_bank([[2, 3]], bank)
    scores = torch.tensor([[0.8, 0.4]])
    model = _model(score_embedding_mode).eval()

    with torch.inference_mode():
        predictions = model(context, query, scores)

    assert predictions.shape == (1, 2)
    assert torch.all((0.0 <= predictions) & (predictions <= 1.0))
    assert torch.isfinite(model.compute_mse_loss(predictions, torch.tensor([[0.9, 0.1]])))
    assert torch.isfinite(model.compute_mmr_loss(predictions, torch.tensor([[0.9, 0.1]])))


def test_cached_inference_matches_forward() -> None:
    """ Verify cached inference matches the standard forward pass.

    Args:
        None.

    Returns:
        None.
    """
    bank = _bank()
    context = collate_prehashed_bank([[0, 1]], bank)
    query = collate_prehashed_bank([[2, 3]], bank)
    scores = torch.tensor([[0.8, 0.4]])
    model = _model().eval()

    with torch.inference_mode():
        expected = model(context, query, scores)
    actual = next(model.context_cached_inference(context, scores, [query]))

    torch.testing.assert_close(actual, expected)


def test_freeze_and_unfreeze_encoder() -> None:
    """ Verify encoder gradient controls update every encoder parameter.

    Args:
        None.

    Returns:
        None.
    """
    model = _model()

    model.freeze_encoder()
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())

    model.unfreeze_encoder()
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
