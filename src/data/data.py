import random
from collections import defaultdict
from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from pandas.core.frame import DataFrame
from tqdm import tqdm

from src.data.movie_embedder import FeatureConfig, MovieEmbedder

"""
TODO:
    - Re-implement get_train batch with cross attension design in mind
"""

def get_user_ratings_dict(
    raw_ratings_data: DataFrame,
    movie_ids: list[str],
    normalization_function: Callable,
) -> dict[str, list[tuple[int, float]]]:
    id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    user_ratings_dict: dict[str, list[tuple[int, float]]] = defaultdict(list)
    total_count = 0
    missing_count = 0

    rows = raw_ratings_data[["user_id", "movie_id", "rating_val"]].itertuples(index=False, name=None)
    for user_id, movie_id, rating_val in tqdm(rows, total=len(raw_ratings_data), desc="Building user ratings dict"):
        total_count += 1
        movie_idx = id_to_idx.get(movie_id)
        if movie_idx is None:
            missing_count += 1
            continue

        rating_norm = normalization_function(rating_val)
        user_ratings_dict[user_id].append((int(movie_idx), rating_norm))

    if missing_count:
        print(f"Warning: Dropping {missing_count} ratings with unknown movie ids out of {total_count} total.")

    return dict(user_ratings_dict)


def iter_user_ratings_shards(
    raw_ratings_data: DataFrame,
    movie_ids: list[str],
    normalization_function: Callable,
    max_users_per_shard: int = 100_000,
):
    """Yield user->ratings shards to keep peak memory bounded.

    Each yielded shard is a dict[user_id] -> list[(movie_idx, norm_rating)], plus
    per-shard missing/total counts so the caller can log warnings.
    """
    id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    shard: dict[str, list[tuple[int, torch.Tensor]]] = defaultdict(list)
    shard_missing = 0
    shard_total = 0

    rows = raw_ratings_data[["user_id", "movie_id", "rating_val"]].itertuples(index=False, name=None)
    for user_id, movie_id, rating_val in tqdm(rows, total=len(raw_ratings_data), desc="Building user ratings shards"):
        shard_total += 1
        movie_idx = id_to_idx.get(movie_id)
        if movie_idx is None:
            shard_missing += 1
            continue

        rating_norm = normalization_function(rating_val)
        shard[user_id].append((int(movie_idx), rating_norm))

        if len(shard) >= max_users_per_shard:
            yield dict(shard), shard_missing, shard_total
            shard = defaultdict(list)
            shard_missing = 0
            shard_total = 0

    if shard:
        yield dict(shard), shard_missing, shard_total


def get_train_batch(
    user_ratings_dict: dict[str, list[tuple[int, torch.Tensor]]],
    movie_embeddings: torch.Tensor,
    batch_size: int = 1,
    context_size: int = 64,
    target_size: int = 8,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates a training batch."""
    if not user_ratings_dict:
        raise ValueError("user_ratings_dict is empty; cannot create batch.")

    users = list(user_ratings_dict.keys())
    if batch_size <= len(users):
        batch_users = random.sample(users, batch_size)
    else:
        batch_users = random.choices(users, k=batch_size)

    user_lengths = [len(user_ratings_dict[u]) for u in batch_users]
    if any(length <= target_size for length in user_lengths):
        raise ValueError("At least one user has too few ratings for the requested target_size.")

    user_context_length = min(context_size + target_size, min(user_lengths))
    if user_context_length <= target_size:
        raise ValueError("Context length must be greater than target_size for all users.")

    train_embeddings: list[torch.Tensor] = []
    train_scores: list[torch.Tensor] = []
    pred_embeddings: list[torch.Tensor] = []
    pred_scores: list[torch.Tensor] = []

    for user_id in batch_users:
        user_ratings = user_ratings_dict[user_id]

        movies_in_context = random.sample(user_ratings, user_context_length)
        pred_movies = random.sample(movies_in_context, target_size)
        train_movies = [movie for movie in movies_in_context if movie not in pred_movies]

        train_movies_ids = torch.tensor([m[0] for m in train_movies], dtype=torch.long)
        train_movies_scores = torch.stack([m[1] for m in train_movies]).to(dtype=torch.float)
        pred_movies_ids = torch.tensor([m[0] for m in pred_movies], dtype=torch.long)
        pred_movies_scores = torch.stack([m[1] for m in pred_movies]).to(dtype=torch.float)

        train_embeddings.append(movie_embeddings[train_movies_ids])
        train_scores.append(train_movies_scores)
        pred_embeddings.append(movie_embeddings[pred_movies_ids])
        pred_scores.append(pred_movies_scores)

    device = torch.device(device)
    train_movie_embeddings = torch.stack(train_embeddings).to(device)
    train_movies_scores = torch.stack(train_scores).to(device)
    pred_movie_embeddings = torch.stack(pred_embeddings).to(device)
    pred_movies_scores = torch.stack(pred_scores).to(device)

    return (train_movie_embeddings, train_movies_scores, pred_movie_embeddings, pred_movies_scores)


if __name__ == "__main__":
    ratings_data = pd.read_csv('data/ratings_subset.csv', lineterminator='\n')
    movie_dataset = torch.load('data/subset_movie_dataset-id-pc-yr.pt')

    user_ratings_dict = get_user_ratings_dict(
        ratings_data,
        movie_dataset['ids'],
        lambda x: torch.tensor(x / 10.0, dtype=torch.float)
    )

    get_train_batch(user_ratings_dict, movie_dataset['data'])
