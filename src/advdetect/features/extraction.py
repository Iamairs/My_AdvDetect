from __future__ import annotations

import torch
import torch.nn.functional as F


def pooled_features(model, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Keep gradients; callers choose no_grad for ordinary inference."""
    logits, maps = model.forward_features(images)
    return logits, [F.adaptive_avg_pool2d(feature, 1).flatten(1) for feature in maps]
