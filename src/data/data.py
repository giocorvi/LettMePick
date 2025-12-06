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
    context_size: int = 64,
    target_size: int = 8,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """ Creates a training batch.
    
    Args:
        user_ratings_dict: user_id -> list[(movie_id, normalized_rating)]
    """
    # TODO: Support batch_size > 1. batch_size as input, remove unsqueeze.
    _, _, embedding_dim = movie_embeddings.shape
    random_user = random.choice(list(user_ratings_dict.keys()))
    user_context_length = min(context_size + target_size, len(user_ratings_dict[random_user]))
    movies_in_context = random.sample(user_ratings_dict[random_user], user_context_length)
    pred_movies = random.sample(movies_in_context, target_size)
    train_movies = [movie for movie in movies_in_context if movie not in pred_movies]

    train_movies_ids = torch.zeros((user_context_length - target_size), dtype=torch.long)
    train_movies_scores = torch.zeros((user_context_length - target_size))
    pred_movies_ids = torch.zeros((target_size), dtype=torch.long)
    pred_movies_scores = torch.zeros((target_size))

    for i, m in enumerate(train_movies):
        train_movies_ids[i] = m[0]
        train_movies_scores[i] = m[1]

    for i, m in enumerate(pred_movies):
        pred_movies_ids[i] = m[0]
        pred_movies_scores[i] = m[1]
    
    train_movie_embeddings = movie_embeddings[train_movies_ids]
    pred_movie_embeddings = movie_embeddings[pred_movies_ids]

    device = torch.device(device)
    train_movie_embeddings = train_movie_embeddings.to(device).unsqueeze(0)
    train_movies_scores = train_movies_scores.to(device).unsqueeze(0)
    pred_movie_embeddings = pred_movie_embeddings.to(device).unsqueeze(0)
    pred_movies_scores = pred_movies_scores.to(device).unsqueeze(0)

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
