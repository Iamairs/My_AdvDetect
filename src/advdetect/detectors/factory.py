from __future__ import annotations

from pathlib import Path

import torch

from .baselines.msp import MSPDetector
from .baselines.lr import LayerRegressionDetector
from .proposed.jad import JADDetector


def build_detector(model, config: dict, device):
    name = config["name"]
    if name == "msp":
        return MSPDetector(model)
    if name == "lr":
        return LayerRegressionDetector(model, config, device)
    if name == "jad":
        return JADDetector(model, config, device)
    raise ValueError(f"Unimplemented detector: {name}; register it in detectors/factory.py")


def save_detector(path: Path, detector, config: dict, binding: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "advdetect_detector_v1", "config": config,
                "binding": binding, "state": detector.state()}, path)


def load_detector(path: Path, model, device, binding: dict):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != "advdetect_detector_v1":
        raise ValueError("Unsupported detector checkpoint")
    if payload["binding"] != binding:
        raise ValueError("Detector checkpoint belongs to different model/data/splits; refit it")
    detector = build_detector(model, payload["config"], device)
    detector.restore(payload["state"])
    return detector, payload["config"]
