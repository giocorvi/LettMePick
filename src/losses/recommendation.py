""" Loss functions for rating prediction and relative ranking."""

import math

import torch
import torch.nn.functional as F


def pairwise_huber_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 0.1,
) -> torch.Tensor:
    """ Regress within-user rating differences with twice the Huber loss.

    Args:
        predictions: Floating-point predicted ratings shaped ``[batch, queries]``.
        targets: Ratings with the same shape, device, and dtype as predictions.
        delta: Positive finite difference-error threshold between quadratic and
            linear penalties, in rating units; this is not a ranking margin.

    Returns:
        Twice the Huber loss, averaged over all unordered pairs (including target
        ties) within each user, then over users. The quadratic region equals
        squared error. A single query returns a differentiable zero.
    """
    if not math.isfinite(delta) or delta <= 0:
        raise ValueError("delta must be finite and greater than zero.")
    if predictions.ndim != 2 or targets.shape != predictions.shape:
        raise ValueError("predictions and targets must have matching [batch, queries] shapes.")
    if predictions.shape[0] == 0 or predictions.shape[1] == 0:
        raise ValueError("predictions and targets must have non-empty batch and query dimensions.")
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise TypeError("predictions and targets must have floating-point dtypes.")
    if predictions.device != targets.device or predictions.dtype != targets.dtype:
        raise ValueError("predictions and targets must share a device and dtype.")

    query_size = predictions.shape[1]
    if query_size == 1:
        return predictions.sum() * 0.0
    first, second = torch.triu_indices(
        query_size, query_size, offset=1, device=predictions.device
    )
    predicted_differences = predictions[:, first] - predictions[:, second]
    target_differences = targets[:, first] - targets[:, second]
    losses = 2.0 * F.huber_loss(
        predicted_differences, target_differences, delta=delta, reduction="none"
    )
    return losses.mean(dim=1).mean()


def pairwise_margin_ranking_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """ Compute masked pairwise margin-ranking loss.

    Args:
        predictions: Predicted ratings shaped ``[batch, queries]``.
        targets: Target ratings with the same shape as ``predictions``.
        margin: Minimum desired separation between differently ranked predictions.

    Returns:
        Mean loss across target pairs with unequal ratings.
    """
    query_size = predictions.shape[1]
    combinations = torch.combinations(
        torch.arange(query_size, device=predictions.device)
    )

    target_order = (
        targets[:, combinations[:, 0]] - targets[:, combinations[:, 1]]
    ).sign()
    predictions_a = predictions[:, combinations[:, 0]]
    predictions_b = predictions[:, combinations[:, 1]]

    # Equal targets do not express a ranking preference and should not affect loss.
    active_pairs = target_order != 0.0
    pair_losses = F.margin_ranking_loss(
        predictions_a,
        predictions_b,
        target_order,
        margin=margin,
        reduction="none",
    )
    active_count = active_pairs.sum().clamp_min(1)
    return (pair_losses * active_pairs).sum() / active_count


def mean_squared_error_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """ Compute mean-squared error for predicted ratings.

    Args:
        predictions: Predicted ratings.
        targets: Target ratings with the same shape as ``predictions``.
        reduction: PyTorch reduction mode applied to element losses.

    Returns:
        Reduced mean-squared error tensor.
    """
    return F.mse_loss(predictions, targets, reduction=reduction)
