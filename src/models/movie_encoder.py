"""Movie ecoder models."""

from __future__ import annotations

import torch


class SimpleMovieEncoder(torch.nn.Module):
    """Encodes an embedded movie into a single combined representation."""
    def __init__(
        self,
        embedding_dim: int,
        num_features: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
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
