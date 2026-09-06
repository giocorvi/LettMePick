from dataclasses import dataclass
import random
from typing import Any

import torch

from src.data.prehash import build_movie_bank, collate_prehashed_bank


@dataclass(frozen=True)
class Movie:
    """ Store normalized movie metadata used by feature hashing."""

    id: str
    year: int
    genres: list[str]
    actors: list[str]
    directors: list[str]


@dataclass(frozen=True)
class MovieCatalogEntry:
    """ Describe one searchable movie while preserving its tensor-bank row."""

    bank_index: int
    id: str
    title: str
    year: int
    genres: list[str]
    letterboxd_rating_count: int = 0


POPULARITY_LEVELS: tuple[dict[str, str | int], ...] = (
    {"id": "any", "label": "Any", "min_rating_count": 0},
    {"id": "established", "label": "Established", "min_rating_count": 1_000},
    {"id": "popular", "label": "Popular", "min_rating_count": 100_000},
    {"id": "blockbuster", "label": "Blockbuster", "min_rating_count": 1_000_000},
)


def _normalize_rating_count(value: Any) -> int:
    """ Normalize raw Letterboxd rating counts to non-negative integers.

    Args:
        value: Raw metadata value.

    Returns:
        Parsed count, or zero when the value is absent or invalid.
    """
    if value is None or isinstance(value, bool):
        return 0
    try:
        count = int(value)
    except (TypeError, ValueError, OverflowError):
        return 0
    return max(count, 0)


def build_movie_catalog(
    movie_dataset: dict[str, dict[str, Any]],
) -> list[MovieCatalogEntry]:
    """ Build display metadata in the same order used by the hashed movie bank.

    Args:
        movie_dataset: Raw movie metadata keyed by movie identifier.

    Returns:
        Searchable catalog entries aligned with movie-bank rows.
    """
    catalog: list[MovieCatalogEntry] = []
    for movie_id, movie_info in movie_dataset.items():
        year = movie_info.get("year_released")
        if year is None:
            continue
        title = movie_info.get("title") or movie_info.get("movie_title") or movie_id
        genres = [str(genre).lower() for genre in movie_info.get("letterboxd_genres", [])]
        catalog.append(
            MovieCatalogEntry(
                bank_index=len(catalog),
                id=str(movie_id),
                title=str(title),
                year=int(year),
                genres=genres,
                letterboxd_rating_count=_normalize_rating_count(
                    movie_info.get("letterboxd_rating_count")
                ),
            )
        )
    return catalog


def filter_movie_catalog(
    catalog: list[MovieCatalogEntry],
    min_year: int | None = None,
    max_year: int | None = None,
    genres: list[str] | None = None,
    min_rating_count: int = 0,
) -> list[MovieCatalogEntry]:
    """ Filter catalog entries by year, genres, and minimum popularity.

    Args:
        catalog: Searchable movies to filter.
        min_year: Optional inclusive earliest release year.
        max_year: Optional inclusive latest release year.
        genres: Optional genres; a movie passes when any genre overlaps.
        min_rating_count: Inclusive minimum Letterboxd rating count.

    Returns:
        Matching catalog entries in their original bank order.
    """
    if min_year is not None and max_year is not None and min_year > max_year:
        raise ValueError("min_year cannot be greater than max_year.")
    if min_rating_count < 0:
        raise ValueError("min_rating_count cannot be negative.")
    selected_genres = {genre.casefold() for genre in genres or []}
    return [
        entry
        for entry in catalog
        if (min_year is None or entry.year >= min_year)
        and (max_year is None or entry.year <= max_year)
        and entry.letterboxd_rating_count >= min_rating_count
        and (
            not selected_genres
            or bool(selected_genres.intersection(genre.casefold() for genre in entry.genres))
        )
    ]


def prepare_dataset(
    user_ratings_dict: dict[str, dict[str, list[int | str]]],
    movie_dataset: dict[str, dict[str, Any]],
    num_id_buckets: int,
    num_actor_buckets: int,
    num_genre_buckets: int,
    num_director_buckets: int,
    verbose: bool = False,
) -> dict[str, torch.Tensor]:
    """ Build a hashed movie bank and align user ratings with its rows.

    Args:
        user_ratings_dict: User movie identifiers and ten-point rating values; updated
            in place with bank indices and normalized ratings.
        movie_dataset: Movie metadata keyed by movie identifier.
        num_id_buckets: Number of hash buckets for movie identifiers.
        num_actor_buckets: Number of hash buckets for actor names.
        num_genre_buckets: Number of hash buckets for genres.
        num_director_buckets: Number of hash buckets for director names.
        verbose: Whether to report ratings whose movies are omitted.

    Returns:
        A tensor bank containing hashed movie features and masks.
    """
    catalog = build_movie_catalog(movie_dataset)
    movie_id_to_index = (
        {entry.id: entry.bank_index for entry in catalog} if user_ratings_dict else {}
    )
    final_movie_dataset: list[Movie] = []
    for entry in catalog:
        movie_info = movie_dataset[entry.id]
        final_movie_dataset.append(
            Movie(
                id=entry.id,
                year=entry.year,
                genres=entry.genres,
                actors=movie_info.get("actors", []),
                directors=movie_info.get("director", []),
            )
        )

    for user_id in user_ratings_dict:
        movie_ids = user_ratings_dict[user_id]["movie_ids"]
        rating_vals = user_ratings_dict[user_id]["rating_vals"]
        new_movie_ids: list[int] = []
        new_rating_vals: list[int | str] = []

        for movie_id, rating_val in zip(movie_ids, rating_vals):
            movie_index = movie_id_to_index.get(movie_id)
            if movie_index is None:
                if verbose:
                    print(
                        f"Warning: movie_id {movie_id} not found in dataset for user {user_id}; skipping."
                    )
                continue
            new_movie_ids.append(movie_index)
            new_rating_vals.append(float(rating_val) / 10.0)

        user_ratings_dict[user_id]["movie_ids"] = new_movie_ids
        user_ratings_dict[user_id]["rating_vals"] = new_rating_vals

    return build_movie_bank(
        final_movie_dataset,
        num_id_buckets=num_id_buckets,
        num_actor_buckets=num_actor_buckets,
        num_genre_buckets=num_genre_buckets,
        num_director_buckets=num_director_buckets,
    )


def get_train_batch(
    user_ratings_dict: dict[str, dict[str, list[int | str]]],
    prehashed_bank: dict[str, torch.Tensor],
    batch_size: int = 1,
    context_size: int = 64,
    target_size: int = 8,
    device: torch.device | str = "cpu",
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    """ Sample context and query movies for a training batch.

    Args:
        user_ratings_dict: Prepared user ratings indexed into ``prehashed_bank``.
        prehashed_bank: Hashed movie feature tensors and masks.
        batch_size: Number of users to sample.
        context_size: Maximum number of rated context movies per user.
        target_size: Number of held-out query movies per user.
        device: Device receiving the collated tensors.

    Returns:
        Context features, context scores, query features, and query scores.
    """
    if not user_ratings_dict:
        raise ValueError("user_ratings_dict is empty; cannot create batch.")

    users = list(user_ratings_dict.keys())
    if batch_size <= len(users):
        batch_users = random.sample(users, batch_size)
    else:
        batch_users = random.choices(users, k=batch_size)

    user_lengths = [len(user_ratings_dict[u].get("movie_ids", [])) for u in batch_users]
    if any(length <= target_size for length in user_lengths):
        raise ValueError("At least one user has too few ratings for the requested target_size.")

    user_context_length = min(context_size + target_size, min(user_lengths))
    if user_context_length <= target_size:
        raise ValueError("Context length must be greater than target_size for all users.")

    train_scores_batch: list[list[float]] = []
    pred_scores_batch: list[list[float]] = []
    train_indices_batch: list[list[int]] = []
    pred_indices_batch: list[list[int]] = []

    for user_id in batch_users:
        movie_ids = user_ratings_dict[user_id].get("movie_ids", [])
        rating_vals = user_ratings_dict[user_id].get(
            "rating_val", user_ratings_dict[user_id].get("rating_vals", [])
        )
        if len(movie_ids) != len(rating_vals):
            raise ValueError(f"movie_ids and rating values length mismatch for user {user_id}.")

        chosen_positions = random.sample(range(len(movie_ids)), user_context_length)
        pred_positions = set(random.sample(chosen_positions, target_size))
        train_positions = [pos for pos in chosen_positions if pos not in pred_positions]

        train_indices = [movie_ids[pos] for pos in train_positions]
        pred_indices = [movie_ids[pos] for pos in pred_positions]
        train_indices_batch.append(train_indices)
        pred_indices_batch.append(pred_indices)

        train_scores = [float(rating_vals[pos]) for pos in train_positions]
        pred_scores = [float(rating_vals[pos]) for pos in pred_positions]
        train_scores_batch.append(train_scores)
        pred_scores_batch.append(pred_scores)

    device = torch.device(device)
    context_batch = collate_prehashed_bank(train_indices_batch, prehashed_bank, device=device)
    query_batch = collate_prehashed_bank(pred_indices_batch, prehashed_bank, device=device)
    context_scores = torch.tensor(train_scores_batch, dtype=torch.float32, device=device)
    query_scores = torch.tensor(pred_scores_batch, dtype=torch.float32, device=device)

    return (context_batch, context_scores, query_batch, query_scores)


def split_user_ratings_dict(
    user_ratings_dict: dict[str, dict[str, list[int | str]]],
    test_ratio: float,
    val_ratio: float = 0.0,
    seed: int | None = None,
) -> tuple[
    dict[str, dict[str, list[int | str]]],
    dict[str, dict[str, list[int | str]]],
    dict[str, dict[str, list[int | str]]],
]:
    """ Split prepared users into train, validation, and test mappings.

    Args:
        user_ratings_dict: Prepared ratings keyed by user identifier.
        test_ratio: Fraction of users assigned to the test mapping.
        val_ratio: Fraction of users assigned to the validation mapping.
        seed: Optional seed used to shuffle users reproducibly.

    Returns:
        Train, validation, and test user-rating mappings.
    """
    if not 0 <= test_ratio <= 1:
        raise ValueError("test_ratio must be between 0 and 1.")
    if not 0 <= val_ratio <= 1:
        raise ValueError("val_ratio must be between 0 and 1.")
    if test_ratio + val_ratio > 1:
        raise ValueError("test_ratio + val_ratio must be <= 1.")

    rng = random.Random(seed)
    train_dict: dict[str, dict[str, list[int | str]]] = {}
    val_dict: dict[str, dict[str, list[int | str]]] = {}
    test_dict: dict[str, dict[str, list[int | str]]] = {}

    user_ids = list(user_ratings_dict.keys())
    rng.shuffle(user_ids)

    test_count = int(len(user_ids) * test_ratio)
    val_count = int(len(user_ids) * val_ratio)
    test_users = set(user_ids[:test_count])
    val_users = set(user_ids[test_count : test_count + val_count])

    for user_id, user_data in user_ratings_dict.items():
        if user_id in test_users:
            test_dict[user_id] = user_data
        elif user_id in val_users:
            val_dict[user_id] = user_data
        else:
            train_dict[user_id] = user_data

    return train_dict, val_dict, test_dict
