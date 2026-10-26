"""Library-based attacks, following Iamairs/CCDE-ID_AdvDetect's factory."""
from __future__ import annotations

import torch
import torchattacks


def build_attack(model, config: dict):
    name = config["name"].lower()
    epsilon = float(config["epsilon"])
    if not 0 < epsilon <= 1:
        raise ValueError("epsilon must be in (0, 1] for raw [0, 1] inputs")
    if config.get("norm", "Linf") != "Linf":
        raise ValueError("The initial experiment protocol currently supports Linf only")
    if name == "pgd":
        steps, alpha = int(config["steps"]), float(config["step_size"])
        if steps < 1 or alpha <= 0:
            raise ValueError("PGD requires positive steps and step_size")
        return torchattacks.PGD(model, eps=epsilon, alpha=alpha, steps=steps,
                                random_start=bool(config.get("random_start", True)))
    if name == "autoattack":
        if config.get("version", "standard") != "standard":
            raise ValueError("Use AutoAttack standard for the initial reproduction")
        return torchattacks.AutoAttack(model, norm="Linf", eps=epsilon,
                                      version="standard", n_classes=int(config.get("n_classes", 10)),
                                      seed=int(config.get("seed", 42)), verbose=bool(config.get("verbose", False)))
    raise ValueError(f"Only standard PGD and AutoAttack are enabled: {name}")
