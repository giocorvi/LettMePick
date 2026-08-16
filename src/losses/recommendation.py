""" Loss functions for rating prediction and relative ranking."""

import torch
import torch.nn.functional as F


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
