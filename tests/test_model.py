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


def _model(**overrides: int | float | bool) -> LettMePick:
    """ Construct a compact recommendation model for tests.

    Args:
        overrides: Constructor values that replace compact test defaults.

    Returns:
        An initialized recommendation model.
    """
    config: dict[str, int | float | bool] = {
        "feature_size": 4,
        "num_id_buckets": 32,
        "num_actor_buckets": 16,
        "num_genre_buckets": 8,
        "num_director_buckets": 8,
        "model_embed_dim": 8,
        "num_attention_heads": 2,
    }
    config.update(overrides)
    return LettMePick(**config)


def test_movie_encoder_returns_movie_embeddings() -> None:
    """ Verify the encoder returns restored fixed-width movie embeddings."""
    batch = collate_prehashed_bank([[0, 1]], _bank())
    encoder = MovieEncoder(
        feature_size=4,
        model_embed_dim=8,
        num_attention_heads=2,
        num_attention_blocks=1,
        num_id_buckets=32,
        num_actor_buckets=16,
        num_genre_buckets=8,
        num_director_buckets=8,
    )

    embeddings = encoder(batch, batch_size=1, sequence_size=2)

    assert embeddings.shape == (1, 2, 8)
    assert torch.all(torch.isfinite(embeddings))


def test_movie_encoder_validates_explicit_batch_shape() -> None:
    """ Verify explicit batch dimensions must match flattened movie count."""
    batch = collate_prehashed_bank([[0, 1]], _bank())
    encoder = _model().encoder

    with pytest.raises(ValueError, match=r"batch_size \* sequence_size"):
        encoder(batch, batch_size=1, sequence_size=3)


def test_year_projection_preserves_values_beyond_reference_range() -> None:
    """ Verify learned year tokens are projected without boundary clamping."""
    batch = collate_prehashed_bank([[0, 1]], _bank())
    batch["year"] = torch.tensor([2030.0, 2120.0])
    encoder = _model().encoder
    with torch.no_grad():
        encoder.year_projection.weight.fill_(1.0)
        encoder.year_projection.bias.zero_()
        assert encoder.feature_type_embedding is not None
        encoder.feature_type_embedding.weight.zero_()

    tokens, _ = encoder._encode_tokens(batch)

    assert tokens.shape[-1] == encoder.feature_size
    torch.testing.assert_close(tokens[0, -1], torch.ones(4))
    torch.testing.assert_close(tokens[1, -1], torch.full((4,), 1.5))


def test_feature_type_embeddings_are_optional() -> None:
    """ Verify movie encoding works without additive feature-type embeddings."""
    batch = collate_prehashed_bank([[0, 1]], _bank())
    encoder = _model(use_feature_type_embeddings=False).encoder

    embeddings = encoder(batch, batch_size=1, sequence_size=2)

    assert encoder.feature_type_embedding is None
    assert embeddings.shape == (1, 2, 8)


@pytest.mark.parametrize(
    "block_count_name",
    [
        "num_movie_attention_blocks",
        "num_self_attention_blocks",
        "num_cross_attention_blocks",
    ],
)
def test_model_rejects_empty_attention_stacks(block_count_name: str) -> None:
    """ Verify every configured attention stack contains at least one block.

    Args:
        block_count_name: Constructor argument configured with zero blocks.
    """
    with pytest.raises(ValueError, match=f"{block_count_name} must be positive"):
        _model(**{block_count_name: 0})


def test_model_scores_hashed_movies() -> None:
    """ Verify the model predicts bounded ratings for hashed movies."""
    bank = _bank()
    context = collate_prehashed_bank([[0, 1]], bank)
    query = collate_prehashed_bank([[2, 3]], bank)
    scores = torch.tensor([[0.8, 0.4]])
    model = _model().eval()

    with torch.inference_mode():
        predictions = model(context, query, scores)

    assert predictions.shape == (1, 2)
    assert torch.all((0.0 <= predictions) & (predictions <= 1.0))


def test_score_shapes_produce_equivalent_predictions() -> None:
    """ Verify two- and three-dimensional score inputs are equivalent."""
    bank = _bank()
    context = collate_prehashed_bank([[0, 1]], bank)
    query = collate_prehashed_bank([[2, 3]], bank)
    scores = torch.tensor([[0.8, 0.4]])
    model = _model().eval()

    with torch.inference_mode():
        two_dimensional = model(context, query, scores)
        three_dimensional = model(context, query, scores.unsqueeze(-1))

    torch.testing.assert_close(three_dimensional, two_dimensional)


def test_score_conditioner_distinguishes_ratings() -> None:
    """ Verify different ratings produce different context conditioning."""
    model = _model()
    movie_embeddings = torch.zeros((1, 2, 8))
    scores = torch.tensor([[0.0, 1.0]])
    with torch.no_grad():
        model.score_encoder[0].weight.fill_(1.0)
        model.score_encoder[0].bias.zero_()
        model.score_encoder[2].weight.fill_(1.0 / 8.0)
        model.score_encoder[2].bias.zero_()

    conditioned = model._condition_context(movie_embeddings, scores)

    assert not torch.equal(conditioned[:, 0], conditioned[:, 1])


def test_score_encoder_receives_gradients() -> None:
    """ Verify training propagates gradients into the score encoder."""
    bank = _bank()
    context = collate_prehashed_bank([[0, 1]], bank)
    query = collate_prehashed_bank([[2, 3]], bank)
    model = _model()

    model(context, query, torch.tensor([[0.8, 0.4]])).sum().backward()

    assert all(parameter.grad is not None for parameter in model.score_encoder.parameters())
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())


def test_cached_inference_matches_forward() -> None:
    """ Verify cached inference matches the standard forward pass."""
    bank = _bank()
    context = collate_prehashed_bank([[0, 1]], bank)
    query = collate_prehashed_bank([[2, 3]], bank)
    scores = torch.tensor([[0.8, 0.4]])
    model = _model().eval()

    with torch.inference_mode():
        expected = model(context, query, scores)
    actual = next(model.context_cached_inference(context, scores, [query]))

    torch.testing.assert_close(actual, expected)


def test_freeze_and_unfreeze_embeddings() -> None:
    """ Verify embedding controls leave movie attention trainable."""
    model = _model()

    model.freeze_embeddings()
    assert not model.encoder.cls_token.requires_grad
    assert all(
        not parameter.requires_grad
        for module in (
            model.encoder.id_embedding,
            model.encoder.actor_embedding,
            model.encoder.genre_embedding,
            model.encoder.director_embedding,
            model.encoder.year_projection,
            model.encoder.feature_type_embedding,
        )
        if module is not None
        for parameter in module.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.encoder.attention_blocks.parameters()
    )
    assert all(
        parameter.requires_grad
        for parameter in model.encoder.output_projection.parameters()
    )

    model.unfreeze_embeddings()
    assert model.encoder.cls_token.requires_grad
    assert all(
        parameter.requires_grad
        for module in (
            model.encoder.id_embedding,
            model.encoder.actor_embedding,
            model.encoder.genre_embedding,
            model.encoder.director_embedding,
            model.encoder.year_projection,
            model.encoder.feature_type_embedding,
        )
        if module is not None
        for parameter in module.parameters()
    )


def test_freeze_and_unfreeze_encoder() -> None:
    """ Verify whole-encoder controls update every encoder parameter."""
    model = _model()

    model.freeze_encoder()
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())

    model.unfreeze_encoder()
    assert all(parameter.requires_grad for parameter in model.encoder.parameters())
