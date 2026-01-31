import random
from typing import Any, Optional

from dataclasses import dataclass
from tqdm import tqdm

import torch


@dataclass(frozen=True)
class Movie:
    id: str
    year: int
    genres: list[str]
    actors: list[str]
    directors: list[str]
    # ratings_count: Optional[int]


def prepare_dataset(
    user_ratings_dict: dict[str, dict[str, list[int | str]]],
    movie_dataset: dict[str, dict[str, Any]],
) -> list[Movie]:
    """
    Creates the movie dataset and prepares the user ratings dict for faster acces during training.

    NOTE: This function modifies the user ratings dict in-place.
    
    :param user_ratings_dict: It's a dict of user ratings. Each user has 'movie_ids' and 'rating_val'.
    :param movie_dataset: It's a dict movie each with represented by a dict itself.
    :type user_ratings_dict: dict[str, dict[str, list]]
    """
    movie_id_to_index: dict[str, int] = {}
    final_movie_dataset: list[Movie] = []

    for movie_id, movie_info in movie_dataset.items():
        if 'year_released' not in movie_info:
            continue

        genres = movie_info.get('letterboxd_genres', [])
        genres = [g.lower() for g in genres]

        final_movie_dataset.append(
            Movie(
                id=movie_id,
                year=movie_info['year_released'],
                genres=genres,
                actors=movie_info.get('actors'),
                directors=movie_info.get('director'),
            )
        )
        movie_id_to_index[movie_id] = len(final_movie_dataset) - 1

    
    for user_id in user_ratings_dict:
        movie_ids = user_ratings_dict[user_id]['movie_ids']
        rating_vals = user_ratings_dict[user_id]['rating_vals']
        new_movie_ids: list[int] = []
        new_rating_vals: list[int | str] = []

        for movie_id, rating_val in zip(movie_ids, rating_vals):
            movie_index = movie_id_to_index.get(movie_id)
            if movie_index is None:
                print(
                    f"Warning: movie_id {movie_id} not found in dataset for user {user_id}; skipping."
                )
                continue
            new_movie_ids.append(movie_index)
            new_rating_vals.append(rating_val)

        user_ratings_dict[user_id]['movie_ids'] = new_movie_ids
        user_ratings_dict[user_id]['rating_vals'] = new_rating_vals

    return final_movie_dataset


def get_train_batch(
    user_ratings_dict: dict[str, dict[str, list[int | str]]],
    movie_dataset: list[Movie],
    batch_size: int = 1,
    context_size: int = 64,
    target_size: int = 8,
) -> tuple[list[list[Movie]], list[list[float]], list[list[Movie]], list[list[float]]]:
    """Creates a training batch."""
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

    train_movies_batch: list[list[Movie]] = []
    train_scores_batch: list[list[float]] = []
    pred_movies_batch: list[list[Movie]] = []
    pred_scores_batch: list[list[float]] = []

    for user_id in batch_users:
        movie_ids = user_ratings_dict[user_id].get("movie_ids", [])
        rating_vals = user_ratings_dict[user_id].get(
            "rating_val", user_ratings_dict[user_id].get("rating_vals", [])
        )
        if len(movie_ids) != len(rating_vals):
            raise ValueError(f"movie_ids and rating_val length mismatch for user {user_id}.")

        chosen_positions = random.sample(range(len(movie_ids)), user_context_length)
        pred_positions = set(random.sample(chosen_positions, target_size))
        train_positions = [pos for pos in chosen_positions if pos not in pred_positions]

        train_movies = [movie_dataset[movie_ids[pos]] for pos in train_positions]
        train_scores = [float(rating_vals[pos]) for pos in train_positions]
        pred_movies = [movie_dataset[movie_ids[pos]] for pos in pred_positions]
        pred_scores = [float(rating_vals[pos]) for pos in pred_positions]

        train_movies_batch.append(train_movies)
        train_scores_batch.append(train_scores)
        pred_movies_batch.append(pred_movies)
        pred_scores_batch.append(pred_scores)

    return (train_movies_batch, train_scores_batch, pred_movies_batch, pred_scores_batch)


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
    """Split a prepared user_ratings_dict into train/val/test dicts."""
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

    for user_id, user_data in user_ratings_dict.items():
        movie_ids = user_data.get("movie_ids", [])
        rating_key = "rating_val" if "rating_val" in user_data else "rating_vals"
        rating_vals = user_data.get(rating_key, [])
        if len(movie_ids) != len(rating_vals):
            raise ValueError(f"movie_ids and {rating_key} length mismatch for user {user_id}.")

        indices = list(range(len(movie_ids)))
        rng.shuffle(indices)

        test_count = int(len(indices) * test_ratio)
        val_count = int(len(indices) * val_ratio)
        test_idx = set(indices[:test_count])
        val_idx = set(indices[test_count : test_count + val_count])

        train_movie_ids: list[int] = []
        train_rating_vals: list[int | str] = []
        val_movie_ids: list[int] = []
        val_rating_vals: list[int | str] = []
        test_movie_ids: list[int] = []
        test_rating_vals: list[int | str] = []

        for idx, movie_id in enumerate(movie_ids):
            rating_val = rating_vals[idx]
            if idx in test_idx:
                test_movie_ids.append(movie_id)
                test_rating_vals.append(rating_val)
            elif idx in val_idx:
                val_movie_ids.append(movie_id)
                val_rating_vals.append(rating_val)
            else:
                train_movie_ids.append(movie_id)
                train_rating_vals.append(rating_val)

        train_dict[user_id] = {"movie_ids": train_movie_ids, rating_key: train_rating_vals}
        val_dict[user_id] = {"movie_ids": val_movie_ids, rating_key: val_rating_vals}
        test_dict[user_id] = {"movie_ids": test_movie_ids, rating_key: test_rating_vals}

    return train_dict, val_dict, test_dict

if __name__ == '__main__':
    # Tests
    movie_dataset = torch.load('/home/jcrows/torch_env/LettMePick/data/2026-batch/tests/sub-movie-dataset.pt', weights_only=False)
    users_ratings_dict = torch.load('/home/jcrows/torch_env/LettMePick/data/2026-batch/tests/sub-users-ratings.pt', weights_only=False)

    movie_dataset = prepare_dataset(users_ratings_dict, movie_dataset)
    import ipdb; ipdb.set_trace()
