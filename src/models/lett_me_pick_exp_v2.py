""" Different score features fusion.

Instead of concateneting the movie features with the scores, the ladder are projected 
onto the same dimension and summed together (see `self.score_features_fusion`) 
"""
import torch
import torch.nn.functional as F

from .movie_encoder import MovieEncoder
from .cross_gpt import CrossAttentionBlock
from .self_gpt import MaskedSelfAttentionBlock, SelfAttentionBlock

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
        mrl_margin: float = 0.1,
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
            mrl_margin: The margin used for the MarginRanking loss.
            device: Device for model parameters and intermediate tensors.
        """
        super().__init__()

        self.encoder = MovieEncoder(
            feature_size=feature_size,
            prefix_size=prefix_size,
            num_id_buckets=num_id_buckets,
            num_actor_buckets=num_actor_buckets,
            num_genre_buckets=num_genre_buckets,
            num_director_buckets=num_director_buckets,
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

        self.loss = torch.nn.MarginRankingLoss(margin=mrl_margin, reduction='none')

        # Fusion (add dropout??)
        self.fusion_block_a = torch.nn.Sequential(
            torch.nn.Linear(1, model_embed_dim),
            torch.nn.LayerNorm(model_embed_dim),
        )
        self.fusion_block_proj = torch.nn.Linear(model_embed_dim, model_embed_dim)


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
        context_movies: dict[str, torch.Tensor],
        query_movies: dict[str, torch.Tensor],
        context_scores: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = context_scores.shape[0]
        context = self._encode_movies(context_movies, batch_size=batch_size)
        query = self._encode_movies(query_movies, batch_size=batch_size)

        context = self.score_features_fusion(context, context_scores)

        for block in self.self_attention_blocks:
            context = block(context)

        for block in self.cross_attention_blocks:
            query = block(query, context)

        batch_size, query_size = context.shape[0], query.shape[1]
        predictions = self.final_block(query.view(batch_size * query_size, -1)).reshape((batch_size, query_size))
        
        return predictions

    def _encode_movies(
        self,
        movies_batch: dict[str, torch.Tensor],
        batch_size: int | None = None,
    ) -> torch.Tensor:
        if batch_size is None:
            raise ValueError("batch_size is required when passing collated tensors.")
        if "id_idx" not in movies_batch:
            raise KeyError("movies_batch is missing required key: 'id_idx'.")
        flat_count = int(movies_batch["id_idx"].shape[0])
        if flat_count == 0:
            raise ValueError("movies_batch is empty; cannot encode movies.")
        if flat_count % batch_size != 0:
            raise ValueError(
                "movies_batch size is not divisible by batch_size; cannot infer sequence length."
            )
        context_size = flat_count // batch_size
        x, mask = self.encoder(movies_batch)
        for block in self.movie_attention_blocks:
            x = block(x, mask)
        cls_embeddings = x[:, 0, :]
        movie_embeddings = self.movie_projection(cls_embeddings)
        return movie_embeddings.reshape(batch_size, context_size, -1)

    def freeze_encoder(self) -> None:
        """Freeze movie encoder parameters to avoid gradient computation."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        """Unfreeze movie encoder parameters so they receive gradients."""
        for param in self.encoder.parameters():
            param.requires_grad = True

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Masked Margin Ranking Loss"""
        # NOTE: Might be nice at some point to track the percentage of "active" pairs.
        query_size = predictions.shape[1]

        combinations = torch.combinations(torch.arange(query_size, device=predictions.device))

        mrl_targets = (targets[:, combinations[:, 0]] - targets[:, combinations[:, 1]]).sign()
        preds_a, preds_b = predictions[:, combinations[:, 0]], predictions[:, combinations[:, 1]]

        mask = mrl_targets != 0.0

        denom = mask.sum().clamp_min(1.0)

        masked_loss = (self.loss(preds_a, preds_b, mrl_targets) * mask).sum() / denom

        return masked_loss
    
    def compute_mse_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute regression loss for relevance scores in [0, 1]."""
        return F.mse_loss(predictions, targets, reduction=reduction)
