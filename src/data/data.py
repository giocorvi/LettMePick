import random
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
) -> dict[str, list[tuple[int, torch.Tensor]]]:
    id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
    ratings_with_idx = raw_ratings_data.assign(
        movie_idx=raw_ratings_data["movie_id"].map(id_to_idx),
        rating_norm=raw_ratings_data["rating_val"].apply(normalization_function),
    )

    if ratings_with_idx["movie_idx"].isnull().any():
        missing_ids = ratings_with_idx.loc[ratings_with_idx["movie_idx"].isnull(), "movie_id"].unique()
        raise KeyError(f"Missing movie ids in provided list: {missing_ids}")

    user_ratings_dict = {}
    grouped = ratings_with_idx.groupby("user_id", sort=False)
    for user_id, rows in tqdm(grouped, total=grouped.ngroups, desc="Building user ratings dict"):
        user_ratings_dict[user_id] = list(zip(rows["movie_idx"].tolist(), rows["rating_norm"].tolist()))

    return user_ratings_dict

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
