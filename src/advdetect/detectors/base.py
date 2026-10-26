from __future__ import annotations

from typing import Protocol

import torch


class ScoringDetector(Protocol):
    """Minimal contract; fitting and auxiliary resources remain method-specific."""

    def score(self, images: torch.Tensor) -> torch.Tensor:
        """Return shape [batch], larger means more anomalous; preserve input gradients."""
        ...
