""" Movie encoder for hashed metadata features."""

from __future__ import annotations

import torch

from .self_gpt import MaskedSelfAttentionBlock


class MovieEncoder(torch.nn.Module):
    """ Encode hashed movie metadata into fixed-width movie embeddings."""

    _NUM_FEATURE_TYPES = 5
    _ID_TYPE = 0
    _ACTOR_TYPE = 1
    _GENRE_TYPE = 2
    _DIRECTOR_TYPE = 3
    _YEAR_TYPE = 4
    _MIN_YEAR = 1850.0
    _YEAR_SPAN = 180.0

    def __init__(
        self,
        feature_size: int,
        model_embed_dim: int,
        num_attention_heads: int,
        num_attention_blocks: int,
        num_id_buckets: int,
        num_actor_buckets: int,
        num_genre_buckets: int,
        num_director_buckets: int,
        use_feature_type_embeddings: bool = True,
    ):
        """ Initialize metadata embeddings and movie-level attention.

        Args:
            feature_size: Width of metadata tokens used by movie attention.
            model_embed_dim: Width of the final movie embedding.
            num_attention_heads: Number of movie-attention heads.
            num_attention_blocks: Number of movie-attention blocks.
            num_id_buckets: Number of movie-identifier buckets.
            num_actor_buckets: Number of actor buckets.
            num_genre_buckets: Number of genre buckets.
            num_director_buckets: Number of director buckets.
            use_feature_type_embeddings: Whether to add learned feature-type embeddings.
        """
        super().__init__()

        if feature_size <= 1:
            raise ValueError("feature_size must be greater than one.")
        if model_embed_dim <= 0:
            raise ValueError("model_embed_dim must be positive.")
        if num_attention_heads <= 0:
            raise ValueError("num_attention_heads must be positive.")
        if num_attention_blocks <= 0:
            raise ValueError("num_attention_blocks must be positive.")
        if feature_size % num_attention_heads != 0:
            raise ValueError("feature_size must be divisible by num_attention_heads.")

        self.feature_size = feature_size
        self.model_embed_dim = model_embed_dim
        self.use_feature_type_embeddings = use_feature_type_embeddings

        # Each metadata family uses its own hash space and learned value table.
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
            self._validate_bucket_count(num_director_buckets, "num_director_buckets"),
            feature_size,
        )
        self.year_projection = torch.nn.Linear(1, feature_size)
        self.feature_type_embedding = (
            torch.nn.Embedding(self._NUM_FEATURE_TYPES, feature_size)
            if use_feature_type_embeddings
            else None
        )
        self.cls_token = torch.nn.Parameter(torch.zeros(feature_size))

        # Movie-level attention aggregates the metadata set into the CLS state.
        self.attention_blocks = torch.nn.ModuleList(
            MaskedSelfAttentionBlock(
                embed_dim=feature_size,
                num_heads=num_attention_heads,
            )
            for _ in range(num_attention_blocks)
        )
        self.output_norm = torch.nn.LayerNorm(feature_size)
        self.output_projection = torch.nn.Linear(feature_size, model_embed_dim)

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        batch_size: int,
        sequence_size: int,
    ) -> torch.Tensor:
        """ Encode a flattened collated batch and restore its movie sequence.

        Args:
            batch: Hashed feature tensors for flattened movies.
            batch_size: Number of user batches represented by the input.
            sequence_size: Number of movies represented for each user.

        Returns:
            Movie embeddings shaped ``[batch, movies, model_embed_dim]``.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if sequence_size <= 0:
            raise ValueError("sequence_size must be positive.")

        tokens, mask = self._encode_tokens(batch)
        expected_count = batch_size * sequence_size
        if tokens.shape[0] != expected_count:
            raise ValueError(
                "Flattened movie count must equal batch_size * sequence_size, got "
                f"{tokens.shape[0]} and {batch_size} * {sequence_size}."
            )

        # Summarize valid metadata, then stabilize and project the CLS representation.
        for block in self.attention_blocks:
            tokens = block(tokens, mask)

        cls_embeddings = self.output_norm(tokens[:, 0, :])
        movie_embeddings = self.output_projection(cls_embeddings)
        return movie_embeddings.reshape(batch_size, sequence_size, self.model_embed_dim)

    def _encode_tokens(
        self,
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ Assemble metadata tokens and their validity mask.

        Args:
            batch: Hashed feature tensors for flattened movies.

        Returns:
            Metadata tokens and their boolean validity mask.
        """
        flat_count = self._validate_batch(batch)
        id_idx = batch["id_idx"]
        year = batch["year"]
        actors_idx = batch["actors_idx"]
        actors_mask = batch["actors_mask"]
        genres_idx = batch["genres_idx"]
        genres_mask = batch["genres_mask"]
        directors_idx = batch["directors_idx"]
        directors_mask = batch["directors_mask"]

        # Categorical metadata becomes an unordered set of type-aware tokens.
        cls = self.cls_token.expand(flat_count, 1, self.feature_size)
        id_tokens = self._with_feature_type(self.id_embedding(id_idx), self._ID_TYPE)
        actor_tokens = self._with_feature_type(
            self._embed_optional(self.actor_embedding, actors_idx, flat_count),
            self._ACTOR_TYPE,
            actors_mask,
        )
        genre_tokens = self._with_feature_type(
            self._embed_optional(self.genre_embedding, genres_idx, flat_count),
            self._GENRE_TYPE,
            genres_mask,
        )
        director_tokens = self._with_feature_type(
            self._embed_optional(self.director_embedding, directors_idx, flat_count),
            self._DIRECTOR_TYPE,
            directors_mask,
        )

        # Keep year continuous and allow the learned projection to extrapolate.
        normalized_year = (year - self._MIN_YEAR) / self._YEAR_SPAN
        normalized_year = normalized_year.to(dtype=self.year_projection.weight.dtype)
        year_tokens = self._with_feature_type(
            self.year_projection(normalized_year.unsqueeze(-1)),
            self._YEAR_TYPE,
        )

        tokens = torch.cat([cls, id_tokens, actor_tokens, genre_tokens, director_tokens, year_tokens], dim=1)

        # CLS, movie ID, and year are always present; list features retain padding masks.
        always_valid = torch.ones(
            (flat_count, 1),
            device=id_idx.device,
            dtype=torch.bool,
        )
        mask = torch.cat(
            [
                always_valid,
                always_valid,
                actors_mask,
                genres_mask,
                directors_mask,
                always_valid,
            ],
            dim=1,
        )
        return tokens, mask

    def _with_feature_type(
        self,
        embedded: torch.Tensor,
        feature_type: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ Add an optional feature-type embedding and apply padding masks.

        Args:
            embedded: Embedded values for one feature type.
            feature_type: Integer identifying the feature type.
            mask: Optional validity mask for padded values.

        Returns:
            Feature tokens with invalid positions zeroed.
        """
        if embedded.ndim == 2:
            embedded = embedded.unsqueeze(1)
        if self.feature_type_embedding is not None:
            embedded = embedded + self.feature_type_embedding.weight[feature_type]
        if mask is not None:
            embedded = embedded * mask.unsqueeze(-1)
        return embedded

    def _embed_optional(
        self,
        embedding: torch.nn.Embedding,
        indices: torch.Tensor,
        flat_count: int,
    ) -> torch.Tensor:
        """ Embed an optional padded feature tensor, including zero-width inputs.

        Args:
            embedding: Embedding table for the feature type.
            indices: Hashed feature indices shaped ``[movies, values]``.
            flat_count: Number of flattened movies.

        Returns:
            Embedded feature values shaped ``[movies, values, feature_size]``.
        """
        if indices.numel():
            return embedding(indices)
        return embedding.weight.new_empty((flat_count, 0, self.feature_size))

    def _set_embeddings_trainable(self, trainable: bool) -> None:
        """ Set gradient state for metadata embedding components.

        Args:
            trainable: Whether embedding components should receive gradients.
        """
        modules: tuple[torch.nn.Module, ...] = (
            self.id_embedding,
            self.actor_embedding,
            self.genre_embedding,
            self.director_embedding,
            self.year_projection,
        )
        for module in modules:
            for parameter in module.parameters():
                parameter.requires_grad_(trainable)
        if self.feature_type_embedding is not None:
            for parameter in self.feature_type_embedding.parameters():
                parameter.requires_grad_(trainable)
        self.cls_token.requires_grad_(trainable)

    def freeze_embeddings(self) -> None:
        """ Freeze metadata embedding and token-construction parameters."""
        self._set_embeddings_trainable(False)

    def unfreeze_embeddings(self) -> None:
        """ Unfreeze metadata embedding and token-construction parameters."""
        self._set_embeddings_trainable(True)

    def freeze(self) -> None:
        """ Freeze every parameter in the movie encoder."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def unfreeze(self) -> None:
        """ Unfreeze every parameter in the movie encoder."""
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def _validate_batch(self, batch: dict[str, torch.Tensor]) -> int:
        """ Validate a flattened collated movie batch.

        Args:
            batch: Candidate hashed feature tensors and masks.

        Returns:
            The number of flattened movies.
        """
        required_shapes = {
            "id_idx": (1, torch.long),
            "year": (1, torch.float32),
            "actors_idx": (2, torch.long),
            "actors_mask": (2, torch.bool),
            "genres_idx": (2, torch.long),
            "genres_mask": (2, torch.bool),
            "directors_idx": (2, torch.long),
            "directors_mask": (2, torch.bool),
        }
        if not isinstance(batch, dict):
            raise TypeError(f"batch must be a dict, got {type(batch)}.")

        for key, (ndim, dtype) in required_shapes.items():
            if key not in batch:
                raise KeyError(f"batch is missing required key: '{key}'.")
            if not isinstance(batch[key], torch.Tensor):
                raise TypeError(f"batch['{key}'] must be a torch.Tensor.")
            if batch[key].ndim != ndim:
                raise ValueError(f"batch['{key}'] must be {ndim}D, got {batch[key].shape}.")
            if batch[key].dtype != dtype:
                raise TypeError(f"batch['{key}'] must have dtype {dtype}, got {batch[key].dtype}.")

        flat_count = int(batch["id_idx"].shape[0])
        if flat_count <= 0:
            raise ValueError("batch is empty; cannot encode movies.")

        device = batch["id_idx"].device
        if device != self.id_embedding.weight.device:
            raise ValueError(
                f"batch must be on encoder device {self.id_embedding.weight.device}, got {device}."
            )
        for key in required_shapes:
            if int(batch[key].shape[0]) != flat_count:
                raise ValueError(
                    f"batch['{key}'] first dim must match id_idx ({flat_count}), "
                    f"got {batch[key].shape[0]}."
                )
            if batch[key].device != device:
                raise ValueError(f"batch['{key}'] must be on device {device}.")

        for indices_key, mask_key in (
            ("actors_idx", "actors_mask"),
            ("genres_idx", "genres_mask"),
            ("directors_idx", "directors_mask"),
        ):
            if batch[indices_key].shape != batch[mask_key].shape:
                raise ValueError(
                    f"batch['{indices_key}'] and batch['{mask_key}'] must have equal shapes."
                )
        return flat_count

    def _validate_bucket_count(self, count: int, name: str) -> int:
        """ Validate and return a positive embedding bucket count.

        Args:
            count: Candidate bucket count.
            name: Parameter name used in validation errors.

        Returns:
            The validated bucket count.
        """
        if count <= 0:
            raise ValueError(f"{name} must be positive.")
        return count
