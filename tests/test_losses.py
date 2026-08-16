""" Tests for standalone recommendation losses."""

import torch

from src.losses import mean_squared_error_loss, pairwise_margin_ranking_loss


def test_mean_squared_error_loss_is_finite() -> None:
    """ Verify rating regression produces a finite scalar loss."""
    loss = mean_squared_error_loss(
        torch.tensor([[0.8, 0.4]]),
        torch.tensor([[0.9, 0.1]]),
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_pairwise_margin_ranking_loss_is_finite() -> None:
    """ Verify relative rating supervision produces a finite scalar loss."""
    loss = pairwise_margin_ranking_loss(
        torch.tensor([[0.8, 0.4]]),
        torch.tensor([[0.9, 0.1]]),
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_pairwise_margin_ranking_loss_applies_margin() -> None:
    """ Verify ranking loss penalizes separation below the requested margin."""
    loss = pairwise_margin_ranking_loss(
        torch.tensor([[0.6, 0.5]]),
        torch.tensor([[1.0, 0.0]]),
        margin=0.2,
    )

    torch.testing.assert_close(loss, torch.tensor(0.1))


def test_pairwise_margin_ranking_loss_ignores_tied_targets() -> None:
    """ Verify target ties do not contribute a ranking preference."""
    loss = pairwise_margin_ranking_loss(
        torch.tensor([[0.9, 0.1]]),
        torch.tensor([[0.5, 0.5]]),
    )

    torch.testing.assert_close(loss, torch.tensor(0.0))
