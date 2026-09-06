""" Tests for standalone recommendation losses."""

import pytest
import torch

from src.losses import mean_squared_error_loss, pairwise_huber_loss, pairwise_margin_ranking_loss


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


@pytest.mark.parametrize(
    ("error", "expected_loss", "expected_gradient"),
    [(0.05, 0.0025, 0.1), (0.1, 0.01, 0.2), (0.4, 0.07, 0.2), (-0.4, 0.07, -0.2)],
)
def test_pairwise_huber_penalties_and_gradients(
    error: float, expected_loss: float, expected_gradient: float
) -> None:
    """ Verify tied targets contribute quadratic and bounded-gradient penalties.

    Args:
        error: Predicted difference for equal target ratings.
        expected_loss: Analytic twice-Huber penalty.
        expected_gradient: Analytic derivative for the first prediction.
    """
    predictions = torch.tensor([[0.5 + error, 0.5]], dtype=torch.float64, requires_grad=True)
    targets = torch.full_like(predictions, 0.5)
    loss = pairwise_huber_loss(predictions, targets, delta=0.1)
    loss.backward()

    assert loss.dtype == predictions.dtype
    assert loss.device == predictions.device
    torch.testing.assert_close(loss, torch.tensor(expected_loss, dtype=loss.dtype))
    torch.testing.assert_close(
        predictions.grad,
        torch.tensor([[expected_gradient, -expected_gradient]], dtype=predictions.dtype),
    )


def test_pairwise_huber_learns_gaps_and_ignores_user_offsets() -> None:
    """ Verify correct ordering alone is insufficient but exact gaps give zero loss."""
    targets = torch.tensor([[0.2, 0.5, 0.9], [0.1, 0.1, 0.4]])
    predictions = torch.tensor([[0.645, 0.655, 0.665], [0.3, 0.3, 0.3]])
    offsets = torch.tensor([[0.2], [-0.1]])

    loss = pairwise_huber_loss(predictions, targets)
    assert loss > 0
    torch.testing.assert_close(pairwise_huber_loss(predictions + offsets, targets), loss)
    torch.testing.assert_close(pairwise_huber_loss(targets + offsets, targets), torch.tensor(0.0))
    order = torch.tensor([2, 0, 1])
    torch.testing.assert_close(pairwise_huber_loss(predictions[:, order], targets[:, order]), loss)


def test_pairwise_huber_averages_pairs_then_users() -> None:
    """ Verify all pairs, including zero-error pairs, receive equal within-user weight."""
    predictions = torch.tensor([[0.2, 0.0, 0.0], [0.4, 0.4, 0.4]])
    loss = pairwise_huber_loss(predictions, torch.zeros_like(predictions))
    # User penalties: (0.03 + 0.03 + 0) / 3 = 0.02, and zero.
    torch.testing.assert_close(loss, torch.tensor(0.01))


def test_pairwise_huber_single_query_has_zero_gradient() -> None:
    """ Verify batches without pairs can participate in backward propagation."""
    predictions = torch.tensor([[0.3], [0.7]], requires_grad=True)
    loss = pairwise_huber_loss(predictions, torch.zeros_like(predictions))
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.0))
    torch.testing.assert_close(predictions.grad, torch.zeros_like(predictions))


@pytest.mark.parametrize("delta", [0.0, -0.1, float("nan"), float("inf")])
def test_pairwise_huber_rejects_invalid_delta(delta: float) -> None:
    """ Reject invalid error thresholds.

    Args:
        delta: Invalid Huber threshold.
    """
    with pytest.raises(ValueError, match="delta must be finite and greater than zero"):
        pairwise_huber_loss(torch.zeros(1, 2), torch.zeros(1, 2), delta=delta)


@pytest.mark.parametrize(
    ("prediction_shape", "target_shape"),
    [((2,), (2,)), ((1, 2), (2, 2)), ((0, 2), (0, 2)), ((1, 0), (1, 0))],
)
def test_pairwise_huber_rejects_invalid_shapes(
    prediction_shape: tuple[int, ...], target_shape: tuple[int, ...]
) -> None:
    """ Reject broadcasting and empty inputs.

    Args:
        prediction_shape: Invalid prediction shape or one mismatching targets.
        target_shape: Shape of targets.
    """
    with pytest.raises(ValueError):
        pairwise_huber_loss(torch.zeros(prediction_shape), torch.zeros(target_shape))


def test_pairwise_huber_rejects_incompatible_dtypes() -> None:
    """ Reject integer ratings and mismatched floating-point dtypes."""
    with pytest.raises(TypeError, match="floating-point"):
        pairwise_huber_loss(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.long))
    with pytest.raises(ValueError, match="device and dtype"):
        pairwise_huber_loss(torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.float64))
