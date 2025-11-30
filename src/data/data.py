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
    user_ratings_dict = {}
    for d in raw_ratings_data.itertuples(index=True):
        indx = movie_ids.index(d.movie_id)
        user_ratings_dict.setdefault(d.user_id, []).append((indx, normalization_function(d.rating_val)))
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
    _, _, embedding_dim = movie_embeddings.shape
    random_user = random.choice(list(user_ratings_dict.keys()))
    user_context_length = min(context_size + target_size, len(user_ratings_dict[random_user]))
    movies_in_context = random.sample(user_ratings_dict[random_user], user_context_length)
    pred_movies = random.sample(movies_in_context, target_size)
    train_movies = [movie for movie in movies_in_context if movie not in pred_movies]

    train_movies_ids = torch.zeros((user_context_length - target_size), dtype=torch.long)
    train_movies_scores = torch.zeros((user_context_length - target_size, 1, 1))
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
    train_movie_embeddings = train_movie_embeddings.to(device)
    train_movies_scores = train_movies_scores.to(device)
    pred_movie_embeddings = pred_movie_embeddings.to(device)
    pred_movies_scores = pred_movies_scores.to(device)

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
