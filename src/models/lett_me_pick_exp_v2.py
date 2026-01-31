""" Different score features fusion.

Instead of concateneting the movie features with the scores, the ladder are projected 
onto the same dimension and summed together (see `self.score_features_fusion`) 
"""
import torch
import torch.nn.functional as F

from .movie_encoder import MovieEncoder
from .cross_gpt import CrossAttentionBlock
from .self_gpt import MaskedSelfAttentionBlock, SelfAttentionBlock
from ..data_v2.data import Movie

class LettMePick_exp(torch.nn.Module):
    def __init__(
        self,
        feature_size: int,
        prefix_size: int,
        num_id_buckets: int,
        num_actor_buckets: int,
        num_genre_buckets: int,
        num_director_buckets: int,
        model_embed_dim: int,
        num_attention_heads: int,
        num_movie_attention_blocks: int = 1,
        num_self_attention_blocks: int = 1,
        num_cross_attention_blocks: int = 1,
        device: torch.device | str = "cpu",
    ):
        """Initialize the model and configure encoders, attention blocks, and fusion layers.

        Args:
            feature_size: Size of per-movie feature vector.
            prefix_size: Size of prefix token features added by the encoder.
            num_id_buckets: Number of ID buckets for the 'movie-id' feature.
            num_actor_buckets: Number of actor buckets for the 'cast' feature.
            num_genre_buckets: Number of genre buckets for the 'genre' feature.
            num_director_buckets: Number of director buckets for the 'director' feature.
            model_embed_dim: Embedding dimension used throughout attention blocks.
            num_attention_heads: Number of heads in each attention block.
            num_movie_attention_blocks: Count of masked self-attention blocks for movie tokens.
            num_self_attention_blocks: Count of self-attention blocks over the context set.
            num_cross_attention_blocks: Count of cross-attention blocks from query to context.
            device: Device for model parameters and intermediate tensors.
        """
        super().__init__()

        self.device = torch.device(device)

        self.encoder = MovieEncoder(
            feature_size=feature_size,
            prefix_size=prefix_size,
            num_id_buckets=num_id_buckets,
            num_actor_buckets=num_actor_buckets,
            num_genre_buckets=num_genre_buckets,
            num_director_buckets=num_director_buckets,
            device=self.device,
        )

        movie_token_dim = feature_size + prefix_size
        self.movie_attention_blocks = torch.nn.ModuleList(
            MaskedSelfAttentionBlock(embed_dim=movie_token_dim, num_heads=num_attention_heads)
            for _ in range(num_movie_attention_blocks)
        )

        self.movie_projection = torch.nn.Linear(movie_token_dim, model_embed_dim)

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

        # Fusion (add dropout??)
        self.fusion_block_a = torch.nn.Sequential(
            torch.nn.Linear(1, model_embed_dim),
            torch.nn.LayerNorm(model_embed_dim),
        )
        self.fusion_block_proj = torch.nn.Linear(model_embed_dim, model_embed_dim)
        self.to(self.device)


    def score_features_fusion(self, features: torch.Tensor, scores: torch.Tensor):
        assert features.ndim == 3 # B, T, D
        B, T, _ = features.shape
        if scores.ndim in [2, 3]: # [B, T] or [B, T, 1]
            features = features + self.fusion_block_a(scores.view(-1, 1)).view(B, T, -1)
            return self.fusion_block_proj(features)
        else:
            raise IndexError(f"Scores has shape: {scores.shape}")

    def forward(
        self,
        context_movies: list[list[Movie]],
        query_movies: list[list[Movie]],
        context_scores: torch.Tensor,
    ) -> torch.Tensor:
        
        context_scores = context_scores.to(self.device)

        context = self._encode_movies(context_movies)
        query = self._encode_movies(query_movies)

        context = self.score_features_fusion(context, context_scores)

        for block in self.self_attention_blocks:
            context = block(context)

        for block in self.cross_attention_blocks:
            query = block(query, context)

        batch_size, context_size = context.shape[0], query.shape[1]
        predictions = self.final_block(query.view(batch_size * context_size, -1)).reshape((batch_size, context_size))
        
        return predictions

    def _encode_movies(self, movies_batch: list[list[Movie]]) -> torch.Tensor:
        if not movies_batch:
            raise ValueError("movies_batch is empty; cannot encode movies.")

        sizes = [len(movies) for movies in movies_batch]
        if any(size == 0 for size in sizes):
            raise ValueError("Each batch entry must contain at least one movie.")
        if len(set(sizes)) != 1:
            raise ValueError("All batch entries must contain the same number of movies.")

        flat_movies = [movie for movies in movies_batch for movie in movies]
        tokens_list = [self.encoder(movie).to(self.device) for movie in flat_movies]
        token_dim = tokens_list[0].shape[1]
        max_len = max(tokens.shape[0] for tokens in tokens_list)

        batch_tokens = torch.zeros(
            (len(tokens_list), max_len, token_dim),
            device=self.device,
            dtype=tokens_list[0].dtype,
        )
        mask = torch.zeros((len(tokens_list), max_len), device=self.device, dtype=torch.bool)

        for idx, tokens in enumerate(tokens_list):
            length = tokens.shape[0]
            batch_tokens[idx, :length] = tokens
            mask[idx, :length] = True

        x = batch_tokens
        for block in self.movie_attention_blocks:
            x = block(x, mask)

        cls_embeddings = x[:, 0, :]
        movie_embeddings = self.movie_projection(cls_embeddings)
        batch_size = len(movies_batch)
        context_size = sizes[0]
        return movie_embeddings.reshape(batch_size, context_size, -1)

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute regression loss for relevance scores in [0, 1]."""
        return F.mse_loss(predictions, targets, reduction=reduction)
