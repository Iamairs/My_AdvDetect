"""JPEG-based Adversarial Data detector (JAD).

The detector follows the paper's score: the ratio between the feature
distance at the last and first selected network layers after JPEG
sanitization.  JPEG is intentionally kept outside the classifier so inputs
remain in [0, 1].  The standard evaluation path is non-differentiable;
adaptive attacks are not silently routed through this implementation.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torchvision.io import decode_jpeg, encode_jpeg


def jpeg_batch(images: torch.Tensor, quality: int) -> torch.Tensor:
    """Encode/decode a BCHW float tensor on CPU and return BCHW float tensor."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("JAD expects images in BCHW RGB format")
    if not 1 <= int(quality) <= 100:
        raise ValueError("JPEG quality must be in [1, 100]")
    cpu = images.detach().clamp(0, 1).mul(255).round().to(torch.uint8).cpu()
    result = []
    for image in cpu:
        encoded = encode_jpeg(image, quality=int(quality))
        result.append(decode_jpeg(encoded).to(torch.float32).div(255))
    return torch.stack(result, dim=0)


class JADDetector:
    """JAD score with explicit first/last layer bindings."""

    def __init__(self, model: nn.Module, config: dict, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.first_layer = str(config.get("first_layer", "backbone.conv1"))
        self.last_layer = str(config.get("last_layer", "backbone.layer4.2"))
        self.quality = int(config.get("jpeg_quality", 75))
        self.eps = float(config.get("denominator_eps", 1e-8))
        modules = dict(model.named_modules())
        if self.first_layer not in modules or self.last_layer not in modules:
            raise ValueError(
                f"JAD layer not found: first={self.first_layer!r}, last={self.last_layer!r}"
            )

    def _features(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        captured: dict[str, torch.Tensor] = {}
        handles = []
        modules = dict(self.model.named_modules())

        def capture(name: str) -> Callable:
            def hook(_module, _inputs, output):
                if not torch.is_tensor(output):
                    raise TypeError(f"JAD hook output is not a tensor: {name}")
                captured[name] = output
            return hook

        handles.append(modules[self.first_layer].register_forward_hook(capture(self.first_layer)))
        handles.append(modules[self.last_layer].register_forward_hook(capture(self.last_layer)))
        try:
            _ = self.model(images)
        finally:
            for handle in handles:
                handle.remove()
        if self.first_layer not in captured or self.last_layer not in captured:
            raise RuntimeError("JAD hooks did not capture both layers")
        first = captured[self.first_layer].flatten(1)
        last = captured[self.last_layer].flatten(1)
        return first, last

    def score(self, images: torch.Tensor) -> torch.Tensor:
        if images.device != self.device:
            images = images.to(self.device)
        clean_first, clean_last = self._features(images)
        sanitized = jpeg_batch(images, self.quality).to(self.device)
        jpeg_first, jpeg_last = self._features(sanitized)
        d1 = (clean_first - jpeg_first).norm(p=2, dim=1)
        dn = (clean_last - jpeg_last).norm(p=2, dim=1)
        return dn / d1.clamp_min(self.eps)

    def fit(self, fit_loader, selection_loader, device):
        # JAD has no learned detector parameters.  The loaders are accepted so
        # it can use the same Experiment lifecycle as learned detectors.
        return {"requires_fitting": False, "selection": self.state(),
                "checkpoint_policy": "fixed_jad_configuration"}

    def state(self) -> dict:
        return {"first_layer": self.first_layer, "last_layer": self.last_layer,
                "jpeg_quality": self.quality, "denominator_eps": self.eps}

    def restore(self, state: dict) -> None:
        if state != self.state():
            raise ValueError("Saved JAD configuration differs")

