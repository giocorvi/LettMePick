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
        device: torch.device | str = "cpu",
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
        self.device = torch.device(device)
        self.to(self.device)

    
    def forward(self, movie: Movie):
        """
        Args:
            movie: Movie instance

        Returns:
            Tensor shaped (num_features, feature_size + prefix_size)
        """
        device = self.device

        id_vecs = self._embed_category(
            movie.id, self.id_embedding, self.id_prefix, device
        )
        actor_vecs = self._embed_category(
            movie.actors, self.actor_embedding, self.actor_prefix, device
        )
        genre_vecs = self._embed_category(
            movie.genres, self.genre_embedding, self.genre_prefix, device
        )
        director_vecs = self._embed_category(
            movie.directors, self.director_embedding, self.director_prefix, device
        )
        year_vec = self._embed_year(movie.year, self.year_prefix, device)

        feature_tensors = [
            self.cls_token.to(device),
            *id_vecs,
            *actor_vecs,
            *genre_vecs,
            *director_vecs,
            year_vec,
        ]
        return torch.stack(feature_tensors, dim=0).to(device)

    def _embed_category(
        self,
        values: Sequence[str] | str | None,
        embedding: torch.nn.Embedding,
        prefix: torch.Tensor,
        device: torch.device,
    ) -> list[torch.Tensor]:
        if values is None:
            values = []
        elif isinstance(values, str):
            values = [values]

        num_buckets = embedding.num_embeddings
        ids = [self._hash_to_bucket(value, num_buckets) for value in values]
        if not ids:
            return []

        tensor_ids = torch.as_tensor(ids, dtype=torch.long, device=device)
        embedded = embedding(tensor_ids)
        return [self._concat_prefix(prefix, vec) for vec in embedded]

    def _embed_year(
        self,
        year: int | float,
        prefix: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        min_year = 1850.0
        max_year = 2030.0
        normalized = (float(year) - min_year) / (max_year - min_year)
        normalized = float(max(0.0, min(1.0, normalized)))
        year_vec = prefix.new_full((self.feature_size,), normalized).to(device)
        return self._concat_prefix(prefix, year_vec)

    def _validate_bucket_count(self, count: int, name: str) -> int:
        if count <= 0:
            raise ValueError(f"{name} must be positive")
        return count

    def _hash_to_bucket(self, value: str | int, num_buckets: int) -> int:
        if isinstance(value, int):
            return value % num_buckets
        digest = hashlib.sha256(str(value).encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "little") % num_buckets

    def _concat_prefix(self, prefix: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return torch.cat([prefix, vec], dim=0)
