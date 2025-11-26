"""Movie feature embedder utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration for a single movie feature."""

    name: str
    cardinality: int | None = None
    value_range: tuple[float, float] | None = None
    pooling: str = "mean"  # how to combine multi-value lookups

    def __post_init__(self) -> None:
        has_cardinality = self.cardinality is not None
        has_range = self.value_range is not None
        if has_cardinality == has_range:
            raise ValueError(
                f"{self.name} must define exactly one of cardinality or value_range"
            )

        if has_cardinality and self.cardinality is not None:
            if self.cardinality <= 0:
                raise ValueError(f"{self.name} cardinality must be positive")
            if self.pooling not in {"mean", "sum"}:
                raise ValueError("pooling must be either 'mean' or 'sum'")

        if has_range and self.value_range is not None:
            min_val, max_val = self.value_range
            if min_val >= max_val:
                raise ValueError(f"{self.name} value_range min must be < max")

    @property
    def is_numeric(self) -> bool:
        return self.value_range is not None


class MovieEmbedder(nn.Module):
    """Embeds movie metadata features into a common dense space."""

    def __init__(
        self,
        embedding_dim: int,
        feature_configs: Sequence[FeatureConfig],
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        self.embedding_dim = embedding_dim
        self.feature_configs: Dict[str, FeatureConfig] = {
            cfg.name: cfg for cfg in feature_configs
        }
        if not self.feature_configs:
            raise ValueError("At least one feature must be configured")

        self.embeddings = nn.ModuleDict(
            {
                cfg.name: nn.Embedding(cfg.cardinality, embedding_dim)
                for cfg in feature_configs
                if not cfg.is_numeric
            }
        )

    @classmethod
    def from_feature_sizes(
        cls,
        embedding_dim: int,
        **feature_sizes: int,
    ) -> "MovieEmbedder":
        """Convenience constructor accepting a mapping of feature sizes."""
        configs = [
            FeatureConfig(name=name, cardinality=size)
            for name, size in feature_sizes.items()
        ]
        return cls(
            embedding_dim=embedding_dim,
            feature_configs=configs,
        )

    def forward(self, features: Sequence[Mapping[str, Tensor]]) -> Tensor:
        """
        Embed the provided features.

        Args:
            features: sequence where each element is a mapping from feature name
                to LongTensor of ids for a single movie. Each tensor should be
                shape (n_values,) for multi-value lookups (e.g. multiple genres)
                or scalar/shape (1,) for single values.
        """
        if not isinstance(features, Sequence):
            raise TypeError("features must be a sequence of per-movie feature maps")
        if not features:
            raise ValueError("features must contain at least one movie")

        batch_size = len(features)
        num_features = len(self.feature_configs)
        if self.embeddings:
            template_weight = next(iter(self.embeddings.values())).weight
            movie_embeddings = template_weight.new_zeros(
                (batch_size, num_features, self.embedding_dim)
            )
        else:
            movie_embeddings = torch.zeros(
                (batch_size, num_features, self.embedding_dim),
                dtype=torch.float32,
            )

        for batch_idx, movie_features in enumerate(features):
            if not isinstance(movie_features, Mapping):
                raise TypeError("Each feature entry must be a mapping of name -> Tensor")
            for feature_idx, (name, config) in enumerate(self.feature_configs.items()):
                if name not in movie_features:
                    continue
                feature_value = movie_features[name]
                if config.is_numeric:
                    numeric_embedding = self._numeric_embedding(
                        feature_value, config, movie_embeddings
                    )
                    if numeric_embedding is None:
                        continue
                    movie_embeddings[batch_idx, feature_idx] = numeric_embedding
                    continue

                inputs = torch.as_tensor(feature_value).long().unsqueeze(0)
                layer = self.embeddings[name]
                embedded = layer(inputs)
                embedded = self._pool_feature(name, config, embedded).squeeze(0)
                movie_embeddings[batch_idx, feature_idx] = embedded

        return movie_embeddings

    def _pool_feature(
        self, feature_name: str, config: FeatureConfig, embedded: Tensor
    ) -> Tensor:
        """Pool embeddings for features with multiple lookups per movie."""
        if embedded.dim() == 3:
            if config.pooling == "sum":
                return embedded.sum(dim=1)
            return embedded.mean(dim=1)
        if embedded.dim() == 2:
            return embedded
        raise ValueError(
            f"Unexpected embedding tensor shape for {feature_name}: {embedded.shape}"
        )

    def _numeric_embedding(
        self, value: Tensor | float | int, config: FeatureConfig, target: Tensor
    ) -> Tensor | None:
        """Convert a numeric value to a dense embedding."""
        scalar = self._extract_scalar(value)
        if scalar is None or config.value_range is None:
            return None

        min_val, max_val = config.value_range
        normalized = (scalar - min_val) / (max_val - min_val)
        normalized = float(
            max(min(normalized, 1.0 - 1e-6), 1e-6)
        )  # keep within (0, 1)
        return target.new_full((self.embedding_dim,), normalized)

    @staticmethod
    def _extract_scalar(value: Tensor | float | int) -> float | None:
        if isinstance(value, Tensor):
            if value.numel() == 0:
                return None
            return float(value.view(-1)[0].item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
