""" Training objectives for movie-rating models."""

from .recommendation import mean_squared_error_loss, pairwise_margin_ranking_loss

__all__ = ["mean_squared_error_loss", "pairwise_margin_ranking_loss"]
