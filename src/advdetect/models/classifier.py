from __future__ import annotations

import torch

from .backbones import build_classifier
from ..utils.config import resolve_path


def load_classifier(config: dict, device: torch.device) -> tuple[torch.nn.Module, dict]:
    cfg = config["model"]
    payload = torch.load(resolve_path(cfg["checkpoint"]), map_location="cpu", weights_only=True)
    if payload.get("format") != "ccde_classifier_v1":
        raise ValueError("Expected ccde_classifier_v1 checkpoint")
    source = payload.get("config", {}).get("classifier", {})
    for key in ("arch", "num_classes", "normalize_input", "input_mean", "input_std", "activation"):
        if key in source and source[key] != cfg.get(key):
            raise ValueError(f"Model setting differs from checkpoint: {key}")
    model = build_classifier(
        arch=cfg["arch"], in_channels=3, num_classes=int(cfg.get("num_classes", 10)),
        small_input=True, normalize_input=cfg["normalize_input"],
        input_mean=tuple(cfg["input_mean"]), input_std=tuple(cfg["input_std"]),
        activation=cfg.get("activation", "relu"),
    )
    model.load_state_dict(payload["model"], strict=True)
    model.to(device).eval()
    # Fixed parameters still permit gradients with respect to images.
    model.requires_grad_(False)
    return model, {"format": payload["format"], "epoch": payload.get("epoch"),
                   "source_metrics": payload.get("metrics", {})}
