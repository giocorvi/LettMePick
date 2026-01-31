"""Model definitions for LettMePick."""

import torch

def log_parameter_counts(model: torch.nn.Module) -> None:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(
        f"Model parameters (total/trainable/frozen): "
        f"{total_params:,} / {trainable_params:,} / {frozen_params:,}"
    )