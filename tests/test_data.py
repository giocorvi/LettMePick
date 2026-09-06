import random

import torch

from src.data.data import (
    build_movie_catalog,
    filter_movie_catalog,
    get_train_batch,
    prepare_dataset,
    split_user_ratings_dict,
)


def _raw_movies() -> dict[str, dict[str, object]]:
    """ Create representative raw metadata for dataset tests.

    Returns:
        Raw metadata keyed by movie identifier.
    """
    return {
        "m1": {
            "year_released": 1990,
            "letterboxd_genres": ["Drama"],
            "letterboxd_rating_count": 1_000,
            "actors": ["Actor A"],
            "director": ["Director A"],
        },
        "m2": {
            "year_released": 2000,
            "letterboxd_genres": ["Comedy"],
            "letterboxd_rating_count": "100000",
            "actors": ["Actor B"],
            "director": ["Director B"],
        },
        "m3": {
            "year_released": 2010,
            "letterboxd_genres": ["Action"],
            "letterboxd_rating_count": -25,
            "actors": [],
            "director": ["Director C"],
        },
        "missing-year": {
            "year_released": None,
            "letterboxd_genres": ["Drama"],
        },
    }


def _prepare(
    ratings: dict[str, dict[str, list[int | str]]],
) -> dict[str, torch.Tensor]:
    """ Prepare a test tensor bank and remap the supplied ratings.

    Args:
        ratings: Raw user ratings to prepare in place.

    Returns:
        The recreated hashed movie tensor bank.
    """
    return prepare_dataset(
        ratings,
        _raw_movies(),
        num_id_buckets=32,
        num_actor_buckets=16,
        num_genre_buckets=8,
        num_director_buckets=8,
    )


def test_prepare_dataset_recreates_model_input_and_remaps_ratings() -> None:
    """ Verify raw metadata becomes model-compatible tensors and ratings."""
    ratings: dict[str, dict[str, list[int | str]]] = {
        "user-1": {
            "movie_ids": ["m1", "unknown", "m3", "missing-year"],
            "rating_vals": [8, 9, 6, 7],
        }
    }

    bank = _prepare(ratings)

    assert bank["id_idx"].shape == (3,)
    torch.testing.assert_close(bank["year"], torch.tensor([1990.0, 2000.0, 2010.0]))
    assert ratings["user-1"]["movie_ids"] == [0, 2]
    assert ratings["user-1"]["rating_vals"] == [0.8, 0.6]


def test_movie_catalog_aligns_with_bank_and_resolves_titles() -> None:
    """ Verify searchable metadata shares filtering and row order with the bank."""
    movies = _raw_movies()
    movies["m1"]["title"] = "First Feature"
    movies["m2"]["movie_title"] = "Second Feature"

    catalog = build_movie_catalog(movies)
    bank = prepare_dataset(
        {},
        movies,
        num_id_buckets=32,
        num_actor_buckets=16,
        num_genre_buckets=8,
        num_director_buckets=8,
    )

    assert [entry.bank_index for entry in catalog] == list(range(len(catalog)))
    assert [entry.title for entry in catalog] == ["First Feature", "Second Feature", "m3"]
    assert [entry.year for entry in catalog] == bank["year"].tolist()
    assert [entry.letterboxd_rating_count for entry in catalog] == [1_000, 100_000, 0]


def test_movie_catalog_normalizes_missing_and_malformed_popularity() -> None:
    """ Verify invalid raw rating counts become zero without dropping movies."""
    movies = _raw_movies()
    movies["m1"]["letterboxd_rating_count"] = None
    movies["m2"]["letterboxd_rating_count"] = "not-a-number"
    movies["m3"].pop("letterboxd_rating_count")

    catalog = build_movie_catalog(movies)

    assert [entry.letterboxd_rating_count for entry in catalog] == [0, 0, 0]


def test_movie_catalog_filters_year_genre_and_popularity_together() -> None:
    """ Verify inclusive popularity composes with years and genre union semantics."""
    movies = _raw_movies()
    movies["m1"]["letterboxd_genres"] = ["Drama", "Thriller"]
    catalog = build_movie_catalog(movies)

    filtered = filter_movie_catalog(
        catalog,
        min_year=1990,
        max_year=2005,
        genres=["comedy", "thriller"],
        min_rating_count=1_000,
    )

    assert [entry.id for entry in filtered] == ["m1", "m2"]


def test_get_train_batch_returns_collated_context_and_query() -> None:
    """ Verify training batches contain aligned context and query tensors."""
    ratings: dict[str, dict[str, list[int | str]]] = {
        "user-1": {
            "movie_ids": ["m1", "m2", "m3"],
            "rating_vals": [8, 7, 6],
        },
        "user-2": {
            "movie_ids": ["m1", "m2", "m3"],
            "rating_vals": [4, 5, 9],
        },
    }
    bank = _prepare(ratings)
    random.seed(7)

    context, context_scores, query, query_scores = get_train_batch(
        ratings,
        bank,
        batch_size=2,
        context_size=2,
        target_size=1,
    )

    assert context["id_idx"].shape == (4,)
    assert query["id_idx"].shape == (2,)
    assert context_scores.shape == (2, 2)
    assert query_scores.shape == (2, 1)
    assert context_scores.dtype == torch.float32


def test_split_user_ratings_is_seeded_and_disjoint() -> None:
    """ Verify seeded user splits are reproducible and disjoint."""
    ratings = {
        f"user-{index}": {"movie_ids": [0], "rating_vals": [0.5]}
        for index in range(10)
    }

    first = split_user_ratings_dict(ratings, test_ratio=0.2, val_ratio=0.3, seed=42)
    second = split_user_ratings_dict(ratings, test_ratio=0.2, val_ratio=0.3, seed=42)

    assert first == second
    train, validation, test = first
    assert (len(train), len(validation), len(test)) == (5, 3, 2)
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
