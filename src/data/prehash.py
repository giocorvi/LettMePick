from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from src.data.data import Movie


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
    """ Map a feature value to a deterministic hash bucket.

    Args:
        value: Feature value to hash.
        num_buckets: Number of available hash buckets.
        cache: Per-call cache of previously computed bucket assignments.

    Returns:
        The zero-based bucket index.
    """
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
    """ Convert movie metadata into deterministic hashed indices.

    Args:
        movies: Movies to encode.
        num_id_buckets: Number of movie-identifier buckets.
        num_actor_buckets: Number of actor buckets.
        num_genre_buckets: Number of genre buckets.
        num_director_buckets: Number of director buckets.

    Returns:
        Hashed metadata for each movie in input order.
    """
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


def build_movie_bank(
    movies: Sequence[Movie],
    num_id_buckets: int,
    num_actor_buckets: int,
    num_genre_buckets: int,
    num_director_buckets: int,
) -> dict[str, torch.Tensor]:
    """ Hash movies directly into a tensor bank without an intermediate object list.

    Args:
        movies: Normalized movies in bank-row order.
        num_id_buckets: Number of movie-identifier buckets.
        num_actor_buckets: Number of actor buckets.
        num_genre_buckets: Number of genre buckets.
        num_director_buckets: Number of director buckets.

    Returns:
        A movie tensor bank suitable for indexed collation.
    """
    if not movies:
        raise ValueError("movies is empty; cannot build movie bank.")

    count = len(movies)
    max_actors = max((len(movie.actors or []) for movie in movies), default=0)
    max_genres = max((len(movie.genres or []) for movie in movies), default=0)
    max_directors = max((len(movie.directors or []) for movie in movies), default=0)
    bank: dict[str, torch.Tensor] = {
        "id_idx": torch.empty(count, dtype=torch.long),
        "year": torch.empty(count, dtype=torch.float32),
        "actors_idx": torch.zeros((count, max_actors), dtype=torch.long),
        "actors_mask": torch.zeros((count, max_actors), dtype=torch.bool),
        "genres_idx": torch.zeros((count, max_genres), dtype=torch.long),
        "genres_mask": torch.zeros((count, max_genres), dtype=torch.bool),
        "directors_idx": torch.zeros((count, max_directors), dtype=torch.long),
        "directors_mask": torch.zeros((count, max_directors), dtype=torch.bool),
    }
    @lru_cache(maxsize=100_000)
    def hash_value(value: str | int, buckets: int) -> int:
        """ Hash one value while bounding preparation-time cache memory.

        Args:
            value: Feature value to hash.
            buckets: Number of available buckets.

        Returns:
            The zero-based bucket index.
        """
        return _hash_to_bucket(value, buckets, {})

    for index, movie in enumerate(movies):
        bank["id_idx"][index] = hash_value(movie.id, num_id_buckets)
        bank["year"][index] = movie.year
        features = (
            ("actors", movie.actors or [], num_actor_buckets),
            ("genres", movie.genres or [], num_genre_buckets),
            ("directors", movie.directors or [], num_director_buckets),
        )
        for name, values, buckets in features:
            if not values:
                continue
            hashed = [hash_value(value, buckets) for value in values]
            end = len(hashed)
            bank[f"{name}_idx"][index, :end] = torch.as_tensor(hashed, dtype=torch.long)
            bank[f"{name}_mask"][index, :end] = True

    return bank


def build_prehashed_bank(
    prehashed: Sequence[PrehashedMovie],
) -> dict[str, torch.Tensor]:
    """ Pack prehashed movies into fixed-width tensors and masks.

    Args:
        prehashed: Hashed movie records to pack.

    Returns:
        A movie tensor bank suitable for indexed collation.
    """
    if not prehashed:
        raise ValueError("prehashed is empty; cannot build prehashed bank.")

    max_actors = max((len(movie.actors_idx) for movie in prehashed), default=0)
    max_genres = max((len(movie.genres_idx) for movie in prehashed), default=0)
    max_directors = max((len(movie.directors_idx) for movie in prehashed), default=0)

    count = len(prehashed)
    bank: dict[str, torch.Tensor] = {
        "id_idx": torch.empty(count, dtype=torch.long),
        "year": torch.empty(count, dtype=torch.float32),
        "actors_idx": torch.zeros((count, max_actors), dtype=torch.long),
        "actors_mask": torch.zeros((count, max_actors), dtype=torch.bool),
        "genres_idx": torch.zeros((count, max_genres), dtype=torch.long),
        "genres_mask": torch.zeros((count, max_genres), dtype=torch.bool),
        "directors_idx": torch.zeros((count, max_directors), dtype=torch.long),
        "directors_mask": torch.zeros((count, max_directors), dtype=torch.bool),
    }

    for i, movie in enumerate(prehashed):
        bank["id_idx"][i] = movie.id_idx
        bank["year"][i] = movie.year

        if movie.actors_idx:
            end = len(movie.actors_idx)
            bank["actors_idx"][i, :end] = torch.as_tensor(movie.actors_idx, dtype=torch.long)
            bank["actors_mask"][i, :end] = True

        if movie.genres_idx:
            end = len(movie.genres_idx)
            bank["genres_idx"][i, :end] = torch.as_tensor(movie.genres_idx, dtype=torch.long)
            bank["genres_mask"][i, :end] = True

        if movie.directors_idx:
            end = len(movie.directors_idx)
            bank["directors_idx"][i, :end] = torch.as_tensor(movie.directors_idx, dtype=torch.long)
            bank["directors_mask"][i, :end] = True

    return bank


def collate_prehashed_bank(
    movie_indices_batch: Sequence[Sequence[int]],
    prehashed_bank: dict[str, torch.Tensor],
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """ Select and flatten equally sized movie sequences from a tensor bank.

    Args:
        movie_indices_batch: Movie-bank indices grouped by batch item.
        prehashed_bank: Source movie feature tensors and masks.
        device: Device receiving the selected tensors.

    Returns:
        Collated movie tensors with a flattened batch-and-sequence dimension.
    """
    if not movie_indices_batch:
        raise ValueError("movie_indices_batch is empty; cannot collate.")

    sizes = [len(indices) for indices in movie_indices_batch]
    if any(size == 0 for size in sizes):
        raise ValueError("Each batch entry must contain at least one movie.")
    if len(set(sizes)) != 1:
        raise ValueError("All batch entries must contain the same number of movies.")

    required_keys = (
        "id_idx",
        "year",
        "actors_idx",
        "actors_mask",
        "genres_idx",
        "genres_mask",
        "directors_idx",
        "directors_mask",
    )
    for key in required_keys:
        if key not in prehashed_bank:
            raise KeyError(f"prehashed_bank is missing required key: '{key}'.")

    flat_indices = [idx for indices in movie_indices_batch for idx in indices]
    bank_device = prehashed_bank["id_idx"].device
    idx_tensor = torch.as_tensor(flat_indices, dtype=torch.long, device=bank_device)

    selected = {key: value.index_select(0, idx_tensor) for key, value in prehashed_bank.items()}

    target_device = torch.device(device)
    if target_device != bank_device:
        selected = {key: value.to(target_device) for key, value in selected.items()}

    return selected
