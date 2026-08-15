""" Movie encoder for hashed metadata features."""

from __future__ import annotations

import torch


class MovieEncoder(torch.nn.Module):
    """ Encode a movie's hashed metadata as a masked token sequence."""
    def __init__(
        self,
        feature_size: int,
        prefix_size: int,
        num_id_buckets: int,
        num_actor_buckets: int,
        num_genre_buckets: int,
        num_director_buckets: int,
    ):
        """ Initialize feature embeddings and learnable prefix tokens.

        Args:
            feature_size: Width of each embedded feature value.
            prefix_size: Width of the learned feature-type prefix.
            num_id_buckets: Number of movie-identifier buckets.
            num_actor_buckets: Number of actor buckets.
            num_genre_buckets: Number of genre buckets.
            num_director_buckets: Number of director buckets.

        Returns:
            None.
        """
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

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """ Encode a collated movie batch as tokens and an attention mask.

        Args:
            batch: Hashed feature tensors and masks for flattened movies.

        Returns:
            Encoded movie tokens and their validity mask.
        """
        return self._encode_collated(batch)

    def _encode_collated(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """ Assemble prefixed tokens from collated hashed features.

        Args:
            batch: Hashed feature tensors and masks for flattened movies.

        Returns:
            Encoded feature tokens and their validity mask.
        """
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
            """ Prepend a feature-type vector and apply an optional mask.

            Args:
                embedded: Embedded values for one feature type.
                prefix: Learned prefix identifying the feature type.
                mask: Optional validity mask for padded values.

            Returns:
                Prefixed feature tokens with invalid positions zeroed.
            """
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
        """ Validate and return a positive embedding bucket count.

        Args:
            count: Candidate bucket count.
            name: Parameter name used in validation errors.

        Returns:
            The validated bucket count.
        """
        if count <= 0:
            raise ValueError(f"{name} must be positive")
        return count
