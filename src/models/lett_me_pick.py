""" Attention-based personalized movie rating model."""
from collections.abc import Iterable, Iterator

import torch

from .movie_encoder import MovieEncoder
from .cross_gpt import CrossAttentionBlock
from .self_gpt import SelfAttentionBlock

class LettMePick(torch.nn.Module):
    def __init__(
        self,
        feature_size: int,
        num_id_buckets: int,
        num_actor_buckets: int,
        num_genre_buckets: int,
        num_director_buckets: int,
        model_embed_dim: int,
        num_attention_heads: int,
        num_movie_attention_blocks: int = 1,
        num_self_attention_blocks: int = 1,
        num_cross_attention_blocks: int = 1,
        use_feature_type_embeddings: bool = True,
    ):
        """ Initialize the movie encoder and rating-conditioned attention layers.

        Args:
            feature_size: Size of per-movie feature vector.
            num_id_buckets: Number of ID buckets for the 'movie-id' feature.
            num_actor_buckets: Number of actor buckets for the 'cast' feature.
            num_genre_buckets: Number of genre buckets for the 'genre' feature.
            num_director_buckets: Number of director buckets for the 'director' feature.
            model_embed_dim: Embedding dimension used throughout attention blocks.
            num_attention_heads: Number of heads in each attention block.
            num_movie_attention_blocks: Count of masked self-attention blocks for movie tokens.
            num_self_attention_blocks: Count of self-attention blocks over the context set.
            num_cross_attention_blocks: Count of cross-attention blocks from query to context.
            use_feature_type_embeddings: Whether to add movie feature-type embeddings.
        """
        super().__init__()

        block_counts = {
            "num_movie_attention_blocks": num_movie_attention_blocks,
            "num_self_attention_blocks": num_self_attention_blocks,
            "num_cross_attention_blocks": num_cross_attention_blocks,
        }
        for name, count in block_counts.items():
            if count <= 0:
                raise ValueError(f"{name} must be positive.")

        # The movie encoder owns metadata tokenization, aggregation, and projection.
        self.encoder = MovieEncoder(
            feature_size=feature_size,
            model_embed_dim=model_embed_dim,
            num_attention_heads=num_attention_heads,
            num_attention_blocks=num_movie_attention_blocks,
            num_id_buckets=num_id_buckets,
            num_actor_buckets=num_actor_buckets,
            num_genre_buckets=num_genre_buckets,
            num_director_buckets=num_director_buckets,
            use_feature_type_embeddings=use_feature_type_embeddings,
        )
        self.score_encoder = torch.nn.Sequential(
            torch.nn.Linear(1, model_embed_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(model_embed_dim, model_embed_dim),
        )

        self.self_attention_blocks = torch.nn.ModuleList(
            SelfAttentionBlock(embed_dim=model_embed_dim, num_heads=num_attention_heads)
            for _ in range(num_self_attention_blocks)
        )
        self.cross_attention_blocks = torch.nn.ModuleList(
            CrossAttentionBlock(embed_dim=model_embed_dim, num_heads=num_attention_heads)
            for _ in range(num_cross_attention_blocks)
        )

        self.final_block = torch.nn.Sequential(
            torch.nn.Linear(model_embed_dim, model_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(model_embed_dim, 1),
            torch.nn.Sigmoid(),
        )

    def _condition_context(self, movie_embeddings: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        """ Add learned score embeddings to rated movie representations.

        Args:
            movie_embeddings: Movie embeddings shaped ``[batch, movies, embedding]``.
            scores: Ratings shaped ``[batch, movies]`` or ``[batch, movies, 1]``.

        Returns:
            Rating-conditioned movie embeddings.
        """
        if movie_embeddings.ndim != 3:
            raise ValueError(
                "movie_embeddings must have shape [B, T, D], got "
                f"{movie_embeddings.shape}."
            )
        if scores.ndim == 2:
            scores = scores.unsqueeze(-1)
        elif scores.ndim != 3 or scores.shape[-1] != 1:
            raise ValueError(
                f"scores must have shape [B, T] or [B, T, 1], got {scores.shape}."
            )
        if scores.shape[:2] != movie_embeddings.shape[:2]:
            raise ValueError(
                "scores batch and movie dimensions must match movie_embeddings, got "
                f"{scores.shape[:2]} and {movie_embeddings.shape[:2]}."
            )
        if scores.device != movie_embeddings.device:
            raise ValueError(
                f"scores must be on device {movie_embeddings.device}, got {scores.device}."
            )
        if not scores.is_floating_point():
            raise TypeError(f"scores must have a floating-point dtype, got {scores.dtype}.")

        score_embeddings = self.score_encoder(scores.to(dtype=movie_embeddings.dtype))
        return movie_embeddings + score_embeddings


    def forward(
        self,
        context_movies: dict[str, torch.Tensor],
        query_movies: dict[str, torch.Tensor],
        context_scores: torch.Tensor,
    ) -> torch.Tensor:
        """ Predict ratings for query movies from a user's rated context.

        Args:
            context_movies: Collated hashed features for rated movies.
            query_movies: Collated hashed features for movies to score.
            context_scores: Known ratings for the context movies.

        Returns:
            Predicted query ratings in the range ``[0, 1]``.
        """
        # Recover explicit sequence shapes while the data contract remains flattened.
        batch_size, context_size = self._context_shape(context_scores)
        query_size = self._sequence_size(query_movies, batch_size, "query_movies")

        context = self.encoder(context_movies, batch_size=batch_size, sequence_size=context_size)
        query = self.encoder(query_movies, batch_size=batch_size, sequence_size=query_size)

        # Ratings condition known history only; query movies remain score-free.
        context = self._condition_context(context, context_scores)

        # Build a user-history representation before queries retrieve from it.
        for block in self.self_attention_blocks:
            context = block(context)

        for block in self.cross_attention_blocks:
            query = block(query, context)

        predictions = self.final_block(query.flatten(start_dim=0, end_dim=1)).reshape((batch_size, query_size))
        
        return predictions


    def _context_shape(self, context_scores: torch.Tensor) -> tuple[int, int]:
        """ Validate context scores and return explicit batch dimensions.

        Args:
            context_scores: Known context ratings shaped ``[B, T]`` or ``[B, T, 1]``.

        Returns:
            The batch size and context sequence size.
        """
        if context_scores.ndim not in (2, 3):
            raise ValueError(
                "context_scores must have shape [B, T] or [B, T, 1], got "
                f"{context_scores.shape}."
            )
        if context_scores.ndim == 3 and context_scores.shape[-1] != 1:
            raise ValueError(
                "context_scores must have a singleton final dimension when 3D, got "
                f"{context_scores.shape}."
            )
        batch_size, sequence_size = map(int, context_scores.shape[:2])
        if batch_size <= 0 or sequence_size <= 0:
            raise ValueError("context_scores batch and sequence dimensions must be positive.")
        return batch_size, sequence_size

    def _sequence_size(
        self,
        movies_batch: dict[str, torch.Tensor],
        batch_size: int,
        name: str,
    ) -> int:
        """ Infer one explicit movie sequence size from a flattened batch.

        Args:
            movies_batch: Collated hashed features for flattened movies.
            batch_size: Number of user batches represented by the input.
            name: Input name used in validation errors.

        Returns:
            The number of movies represented for each user.
        """
        if "id_idx" not in movies_batch:
            raise KeyError(f"{name} is missing required key: 'id_idx'.")
        flat_count = int(movies_batch["id_idx"].shape[0])
        if flat_count <= 0:
            raise ValueError(f"{name} is empty; cannot infer sequence size.")
        if flat_count % batch_size != 0:
            raise ValueError(
                f"{name} flat size ({flat_count}) must be divisible by batch_size "
                f"({batch_size})."
            )
        return flat_count // batch_size

    def freeze_embeddings(self) -> None:
        """ Prevent metadata embedding components from receiving gradients."""
        self.encoder.freeze_embeddings()

    def unfreeze_embeddings(self) -> None:
        """ Allow metadata embedding components to receive gradients."""
        self.encoder.unfreeze_embeddings()

    def freeze_encoder(self) -> None:
        """ Prevent the complete movie encoder from receiving gradients."""
        self.encoder.freeze()

    def unfreeze_encoder(self) -> None:
        """ Allow the complete movie encoder to receive gradients."""
        self.encoder.unfreeze()

    def context_cached_inference(
        self,
        context_movies: dict[str, torch.Tensor],
        context_scores: torch.Tensor,
        query_batches: Iterable[dict[str, torch.Tensor]],
    ) -> Iterator[torch.Tensor]:
        """ Score query chunks while reusing one encoded user context.

        Args:
            context_movies: Collated hashed features for rated movies.
            context_scores: Known ratings for the context movies.
            query_batches: Query chunks with flattened batch and sequence dimensions.

        Returns:
            An iterator yielding predicted ratings for each query chunk.
        """
        self.eval()

        with torch.inference_mode():
            context = self.encode_context(context_movies, context_scores)
            for query_batch in query_batches:
                yield self.score_encoded_context(context, query_batch)

    def encode_context(
        self,
        context_movies: dict[str, torch.Tensor],
        context_scores: torch.Tensor,
    ) -> torch.Tensor:
        """ Encode and rating-condition a context for reuse across query batches.

        Args:
            context_movies: Collated hashed features for rated movies.
            context_scores: Known ratings for the context movies.

        Returns:
            Context embeddings after rating conditioning and self-attention.
        """
        batch_size, context_size = self._context_shape(context_scores)
        context = self.encoder(
            context_movies,
            batch_size=batch_size,
            sequence_size=context_size,
        )
        context = self._condition_context(context, context_scores)
        for block in self.self_attention_blocks:
            context = block(context)
        return context

    def score_encoded_context(
        self,
        encoded_context: torch.Tensor,
        query_movies: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """ Score one query batch against a previously encoded context.

        Args:
            encoded_context: Context embeddings shaped ``[batch, movies, embedding]``.
            query_movies: Collated hashed features for query movies.

        Returns:
            Predicted ratings in the range ``[0, 1]``.
        """
        if encoded_context.ndim != 3:
            raise ValueError(
                "encoded_context must have shape [B, T, D], got "
                f"{encoded_context.shape}."
            )
        batch_size = int(encoded_context.shape[0])
        query_size = self._sequence_size(query_movies, batch_size, "query_movies")
        query = self.encoder(
            query_movies,
            batch_size=batch_size,
            sequence_size=query_size,
        )
        for block in self.cross_attention_blocks:
            query = block(query, encoded_context)
        return self.final_block(
            query.reshape(batch_size * query_size, -1)
        ).reshape((batch_size, query_size))
