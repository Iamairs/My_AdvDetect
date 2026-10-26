from __future__ import annotations

import torch


class MSPDetector:
    """Simple confidence control, not a robust adversarial defense."""

    def __init__(self, model):
        self.model = model

    def score(self, images: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.model(images).softmax(1).amax(1)

    def fit(self, fit_loader, selection_loader, device):
        return {"requires_fitting": False}

    def state(self) -> dict:
        return {}

    def restore(self, state: dict) -> None:
        if state:
            raise ValueError("MSP has no fitted state")
