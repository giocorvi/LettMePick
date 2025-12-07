import torch
import torch.nn.functional as F

from .movie_encoder import SimpleMovieEncoder
from .cross_gpt import CrossAttentionBlock
from .self_gpt import SelfAttentionBlock

class LettMePick(torch.nn.Module):
    def __init__(
        self,
        data_embed_dim: int,
        num_features: int,
        model_embed_dim: int,
        num_attention_heads: int,
        num_self_attention_blocks: int = 1,
        num_cross_attention_blocks: int = 1,
        encoder_hidden_size: int | None = None,
        encoder_num_layers: int = 2,
    ):
        super().__init__()

        hidden_size = encoder_hidden_size or model_embed_dim
        self.encoder = SimpleMovieEncoder(
            embedding_dim=data_embed_dim,
            num_features=num_features,
            output_size=model_embed_dim - 1,
            hidden_size=hidden_size,
            num_layers=encoder_num_layers,
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

    def score_features_fusion(self, features: torch.Tensor, scores: torch.Tensor):
        assert features.ndim == 3
        if scores.ndim == 2:
            return torch.cat((features, scores.unsqueeze(-1)), dim=-1)
        elif scores.ndim == 3:
            return torch.cat((features, scores), dim=-1)
        else:
            raise IndexError(f"Scores has shape: {scores.shape}")

    def forward(
        self,
        context_embed: torch.Tensor,
        query_embed: torch.Tensor,
        context_scores: torch.Tensor,
    ) -> torch.Tensor:
        
        context = self.encoder(context_embed)
        query = self.encoder(query_embed)

        context = self.score_features_fusion(context, context_scores)
        query = self.score_features_fusion(query, torch.zeros_like(query[:, :, 0]))

        for block in self.self_attention_blocks:
            context = block(context)

        for block in self.cross_attention_blocks:
            query = block(query, context)

        batch_size, context_size = context.shape[0], query.shape[1]
        predictions = self.final_block(query.view(batch_size * context_size, -1)).reshape((batch_size, context_size))
        
        return predictions

    def compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute regression loss for relevance scores in [0, 1]."""
        return F.mse_loss(predictions, targets, reduction=reduction)
