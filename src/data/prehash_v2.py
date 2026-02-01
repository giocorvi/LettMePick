from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.data_v2.data import Movie


@dataclass(frozen=True)
class PrehashedMovie:
    id_idx: int
    year: float
    actors_idx: list[int]
    genres_idx: list[int]
    directors_idx: list[int]


def _hash_to_bucket(
    value: str | int,
    num_buckets: int,
    cache: dict[tuple[str | int, int], int],
) -> int:
    cache_key = (value, num_buckets)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    if isinstance(value, int):
        bucket = value % num_buckets
    else:
        digest = hashlib.sha256(str(value).encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:8], "little") % num_buckets
    cache[cache_key] = bucket
    return bucket


def prehash_movies(
    movies: Sequence[Movie],
    num_id_buckets: int,
    num_actor_buckets: int,
    num_genre_buckets: int,
    num_director_buckets: int,
) -> list[PrehashedMovie]:
    """Precompute hashed indices for Movie instances."""
    cache: dict[tuple[str | int, int], int] = {}
    prehashed: list[PrehashedMovie] = []

    for movie in movies:
        id_idx = _hash_to_bucket(movie.id, num_id_buckets, cache)
        actors_idx = [
            _hash_to_bucket(actor, num_actor_buckets, cache)
            for actor in (movie.actors or [])
        ]
        genres_idx = [
            _hash_to_bucket(genre, num_genre_buckets, cache)
            for genre in (movie.genres or [])
        ]
        directors_idx = [
            _hash_to_bucket(director, num_director_buckets, cache)
            for director in (movie.directors or [])
        ]
        prehashed.append(
            PrehashedMovie(
                id_idx=id_idx,
                year=float(movie.year),
                actors_idx=actors_idx,
                genres_idx=genres_idx,
                directors_idx=directors_idx,
            )
        )

    return prehashed


def collate_prehashed_batch(
    movies_batch: Sequence[Sequence[PrehashedMovie]],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Collate prehashed movies into padded index tensors + masks."""
    if not movies_batch:
        raise ValueError("movies_batch is empty; cannot collate.")

    sizes = [len(movies) for movies in movies_batch]
    if any(size == 0 for size in sizes):
        raise ValueError("Each batch entry must contain at least one movie.")
    if len(set(sizes)) != 1:
        raise ValueError("All batch entries must contain the same number of movies.")

    flat_movies = [movie for movies in movies_batch for movie in movies]
    batch_size = len(flat_movies)

    max_actors = max((len(movie.actors_idx) for movie in flat_movies), default=0)
    max_genres = max((len(movie.genres_idx) for movie in flat_movies), default=0)
    max_directors = max((len(movie.directors_idx) for movie in flat_movies), default=0)

    device = torch.device(device)
    id_idx = torch.empty(batch_size, dtype=torch.long, device=device)
    year = torch.empty(batch_size, dtype=torch.float32, device=device)

    actors_idx = torch.zeros((batch_size, max_actors), dtype=torch.long, device=device)
    actors_mask = torch.zeros((batch_size, max_actors), dtype=torch.bool, device=device)
    genres_idx = torch.zeros((batch_size, max_genres), dtype=torch.long, device=device)
    genres_mask = torch.zeros((batch_size, max_genres), dtype=torch.bool, device=device)
    directors_idx = torch.zeros((batch_size, max_directors), dtype=torch.long, device=device)
    directors_mask = torch.zeros((batch_size, max_directors), dtype=torch.bool, device=device)

    for i, movie in enumerate(flat_movies):
        id_idx[i] = movie.id_idx
        year[i] = movie.year

        for j, actor_idx in enumerate(movie.actors_idx):
            actors_idx[i, j] = actor_idx
            actors_mask[i, j] = True

        for j, genre_idx in enumerate(movie.genres_idx):
            genres_idx[i, j] = genre_idx
            genres_mask[i, j] = True

        for j, director_idx in enumerate(movie.directors_idx):
            directors_idx[i, j] = director_idx
            directors_mask[i, j] = True

    return {
        "id_idx": id_idx,
        "year": year,
        "actors_idx": actors_idx,
        "actors_mask": actors_mask,
        "genres_idx": genres_idx,
        "genres_mask": genres_mask,
        "directors_idx": directors_idx,
        "directors_mask": directors_mask,
    }
