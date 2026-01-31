"""Movie ecoder models."""

from __future__ import annotations

import hashlib
from typing import Sequence

import torch

from src.data_v2.data import Movie


class SimpleMovieEncoder(torch.nn.Module):
    """Encodes an embedded movie into a single combined representation."""
    def __init__(
        self,
        embedding_dim: int,
        num_features: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
    ):
        super().__init__()

        input_size = embedding_dim * num_features

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        layers = []
        current_dim = input_size
        for layer_idx in range(num_layers):
            next_dim = output_size if layer_idx == num_layers - 1 else hidden_size
            linear = torch.nn.Linear(current_dim, next_dim)

            if layer_idx == num_layers - 1:
                layers.append(torch.nn.Sequential(linear))
            else:
                layers.append(torch.nn.Sequential(linear, torch.nn.ReLU()))

            current_dim = next_dim

        self.model = torch.nn.ModuleList(layers)
    
    def forward(self, movie_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            movie_embeddings: tensor shaped (batch, context_size, num_features, embedding_dim)

        Returns:
            Tensor shaped (batch, context_size, output_size)
        """
        batch_size, context_size = movie_embeddings.shape[0], movie_embeddings.shape[1]

        # collapse features/embedding dims -> (batch * context, num_features * embedding_dim)
        x = movie_embeddings.view(batch_size * context_size, -1)

        # sequentially map flattened embeddings into output space
        for layer in self.model:
            x = layer(x)

        return x.reshape((batch_size, context_size, -1))


class MovieEncoder(torch.nn.Module):
    """Encodes an embedded movie into a single combined representation."""
    def __init__(
        self,
        feature_size: int,
        prefix_size: int,
        num_id_buckets: int,
        num_actor_buckets: int,
        num_genre_buckets: int,
        num_director_buckets: int,
    ):
        super().__init__()

        if feature_size <= 0:
            raise ValueError("feature_size must be positive")
        if prefix_size <= 0:
            raise ValueError("prefix_size must be positive")

        self.feature_size = feature_size
        self.prefix_size = prefix_size

        self.id_embedding = torch.nn.Embedding(
            self._validate_bucket_count(num_id_buckets, "num_id_buckets"), feature_size
        )
        self.actor_embedding = torch.nn.Embedding(
            self._validate_bucket_count(num_actor_buckets, "num_actor_buckets"), feature_size
        )
        self.genre_embedding = torch.nn.Embedding(
            self._validate_bucket_count(num_genre_buckets, "num_genre_buckets"), feature_size
        )
        self.director_embedding = torch.nn.Embedding(
            self._validate_bucket_count(num_director_buckets, "num_director_buckets"), feature_size
        )

        self.id_prefix = torch.nn.Parameter(torch.zeros(prefix_size))
        self.actor_prefix = torch.nn.Parameter(torch.zeros(prefix_size))
        self.genre_prefix = torch.nn.Parameter(torch.zeros(prefix_size))
        self.director_prefix = torch.nn.Parameter(torch.zeros(prefix_size))
        self.year_prefix = torch.nn.Parameter(torch.zeros(prefix_size))
        self.cls_token = torch.nn.Parameter(torch.zeros(prefix_size + feature_size))
        self._hash_cache: dict[tuple[str | int, int], int] = {}

    
    def forward(self, movies: Movie | Sequence[Movie]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            movies: Movie instance or a batch of Movie instances

        Returns:
            Tensor shaped (num_features, feature_size + prefix_size) for a single movie,
            or (batch, num_features, feature_size + prefix_size) for a batch.
        """
        if isinstance(movies, Movie):
            collated = self.collate_movies([movies])
            embeddings, mask = self._encode_collated(collated)
            return embeddings[0], mask[0]
        if not movies:
            raise ValueError("movies batch is empty; cannot encode.")
        collated = self.collate_movies(movies)
        return self._encode_collated(collated)

    def collate_movies(self, movies: Sequence[Movie]) -> dict[str, torch.Tensor]:
        """Collate a batch of Movie objects into padded index tensors + masks."""
        if not movies:
            raise ValueError("movies batch is empty; cannot collate.")

        device = self.id_embedding.weight.device
        batch_size = len(movies)

        max_actors = max(len(movie.actors or []) for movie in movies)
        max_genres = max(len(movie.genres or []) for movie in movies)
        max_directors = max(len(movie.directors or []) for movie in movies)

        id_idx = torch.empty(batch_size, dtype=torch.long, device=device)
        year = torch.empty(batch_size, dtype=torch.float32, device=device)

        actors_idx = torch.zeros((batch_size, max_actors), dtype=torch.long, device=device)
        actors_mask = torch.zeros((batch_size, max_actors), dtype=torch.bool, device=device)
        genres_idx = torch.zeros((batch_size, max_genres), dtype=torch.long, device=device)
        genres_mask = torch.zeros((batch_size, max_genres), dtype=torch.bool, device=device)
        directors_idx = torch.zeros(
            (batch_size, max_directors), dtype=torch.long, device=device
        )
        directors_mask = torch.zeros(
            (batch_size, max_directors), dtype=torch.bool, device=device
        )

        for i, movie in enumerate(movies):
            id_idx[i] = self._hash_to_bucket(movie.id, self.id_embedding.num_embeddings)
            year[i] = float(movie.year)

            actors = movie.actors or []
            for j, actor in enumerate(actors):
                actors_idx[i, j] = self._hash_to_bucket(
                    actor, self.actor_embedding.num_embeddings
                )
                actors_mask[i, j] = True

            genres = movie.genres or []
            for j, genre in enumerate(genres):
                genres_idx[i, j] = self._hash_to_bucket(
                    genre, self.genre_embedding.num_embeddings
                )
                genres_mask[i, j] = True

            directors = movie.directors or []
            for j, director in enumerate(directors):
                directors_idx[i, j] = self._hash_to_bucket(
                    director, self.director_embedding.num_embeddings
                )
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

    def _encode_collated(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        device = self.id_embedding.weight.device

        id_idx = batch["id_idx"]
        year = batch["year"]
        actors_idx = batch["actors_idx"]
        actors_mask = batch["actors_mask"]
        genres_idx = batch["genres_idx"]
        genres_mask = batch["genres_mask"]
        directors_idx = batch["directors_idx"]
        directors_mask = batch["directors_mask"]

        batch_size = id_idx.shape[0]
        token_dim = self.prefix_size + self.feature_size

        def _with_prefix(
            embedded: torch.Tensor,
            prefix: torch.Tensor,
            mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if embedded.ndim == 2:
                embedded = embedded.unsqueeze(1)
            prefixed = torch.cat(
                [prefix.expand(embedded.size(0), embedded.size(1), -1), embedded],
                dim=2,
            )
            if mask is not None:
                prefixed = prefixed * mask.unsqueeze(-1)
            return prefixed

        cls = self.cls_token.expand(batch_size, 1, token_dim)

        id_emb = self.id_embedding(id_idx)
        id_feat = _with_prefix(id_emb, self.id_prefix)

        actors_emb = (
            self.actor_embedding(actors_idx)
            if actors_idx.numel()
            else actors_idx.new_empty((batch_size, 0, self.feature_size))
        )
        actors_feat = _with_prefix(actors_emb, self.actor_prefix, actors_mask)

        genres_emb = (
            self.genre_embedding(genres_idx)
            if genres_idx.numel()
            else genres_idx.new_empty((batch_size, 0, self.feature_size))
        )
        genres_feat = _with_prefix(genres_emb, self.genre_prefix, genres_mask)

        directors_emb = (
            self.director_embedding(directors_idx)
            if directors_idx.numel()
            else directors_idx.new_empty((batch_size, 0, self.feature_size))
        )
        directors_feat = _with_prefix(directors_emb, self.director_prefix, directors_mask)

        min_year = 1850.0
        max_year = 2030.0
        year_norm = (year - min_year) / (max_year - min_year)
        year_norm = torch.clamp(year_norm, 0.0, 1.0)
        year_vec = year_norm.unsqueeze(1).expand(-1, self.feature_size)
        year_feat = _with_prefix(year_vec, self.year_prefix)

        embeddings = torch.cat(
            [cls, id_feat, actors_feat, genres_feat, directors_feat, year_feat], dim=1
        )
        mask = torch.cat(
            [
                torch.ones((batch_size, 1), device=device, dtype=torch.bool),
                torch.ones((batch_size, 1), device=device, dtype=torch.bool),
                actors_mask,
                genres_mask,
                directors_mask,
                torch.ones((batch_size, 1), device=device, dtype=torch.bool),
            ],
            dim=1,
        )
        return embeddings, mask


    def _validate_bucket_count(self, count: int, name: str) -> int:
        if count <= 0:
            raise ValueError(f"{name} must be positive")
        return count

    def _hash_to_bucket(self, value: str | int, num_buckets: int) -> int:
        cache_key = (value, num_buckets)
        cached = self._hash_cache.get(cache_key)
        if cached is not None:
            return cached
        if isinstance(value, int):
            bucket = value % num_buckets
        else:
            digest = hashlib.sha256(str(value).encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "little") % num_buckets
        self._hash_cache[cache_key] = bucket
        return bucket
