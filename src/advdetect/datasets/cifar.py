from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Resize, ToTensor

from ..utils.config import resolve_path
from ..utils.artifacts import stable_hash


class IndexedSubset(Dataset):
    def __init__(self, source, indices: list[int], split: str):
        self.source = source
        self.indices = list(indices)
        self.split = split

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.source[self.indices[index]]
        return image, label, self.indices[index]


def stratified_partition(labels, first_size: int, seed: int) -> tuple[list[int], list[int]]:
    labels = np.asarray(labels)
    if not 0 < first_size < len(labels):
        raise ValueError("Split size must leave both partitions nonempty")
    classes, counts = np.unique(labels, return_counts=True)
    ideal = counts * first_size / len(labels)
    allocation = np.floor(ideal).astype(int)
    for i in np.argsort(-(ideal - allocation), kind="stable")[:first_size - allocation.sum()]:
        allocation[i] += 1
    rng = np.random.default_rng(seed)
    first, second = [], []
    for cls, count in zip(classes, allocation):
        indices = rng.permutation(np.flatnonzero(labels == cls)).tolist()
        first.extend(indices[:count])
        second.extend(indices[count:])
    rng.shuffle(first)
    rng.shuffle(second)
    return first, second


def _fingerprint(dataset) -> str:
    digest = hashlib.sha256()
    digest.update(dataset.data.tobytes())
    digest.update(np.asarray(dataset.targets, dtype="<i8").tobytes())
    return digest.hexdigest()


def build_data(config: dict) -> tuple[dict[str, IndexedSubset], dict]:
    cfg = config["dataset"]
    if cfg["name"] not in {"cifar10", "cifar100"}:
        raise ValueError(f"Unsupported CIFAR dataset: {cfg['name']}")
    root = resolve_path(cfg["root"])
    if root.name == "cifar-10-batches-py":
        root = root.parent
    dataset_cls = CIFAR10 if cfg["name"] == "cifar10" else CIFAR100
    resize = cfg.get("resize")
    transform = ToTensor() if resize is None else Compose([Resize((int(resize), int(resize))), ToTensor()])
    train = dataset_cls(str(root), train=True, download=bool(cfg.get("download", False)), transform=transform)
    test = dataset_cls(str(root), train=False, download=bool(cfg.get("download", False)), transform=transform)
    split_cfg = config["split"]
    seed = int(split_cfg["seed"])
    fit, train_remainder = stratified_partition(train.targets, int(split_cfg["fit_size"]), seed)
    selection_size = int(split_cfg.get("selection_size", len(train_remainder) - int(split_cfg["calibration_size"])))
    if selection_size <= 0 or selection_size >= len(train_remainder):
        raise ValueError("selection_size must leave a nonempty calibration split")
    remainder_labels = np.asarray(train.targets)[train_remainder]
    selection_local, calibration_local = stratified_partition(remainder_labels, selection_size, seed + 1)
    selection = [train_remainder[i] for i in selection_local]
    calibration = [train_remainder[i] for i in calibration_local]
    evaluation = list(range(len(test)))
    indices = {"fit": fit, "selection": selection, "calibration": calibration, "test": evaluation}
    sources = {"fit": train, "selection": train, "calibration": train, "test": test}
    limits = config.get("limits", {})
    for name in indices:
        limit = limits.get(name)
        if limit is not None:
            if int(limit) < 1:
                raise ValueError(f"limits.{name} must be positive or null")
            indices[name] = indices[name][:int(limit)]
    fingerprints = {"train": _fingerprint(train), "test": _fingerprint(test)}
    manifest = {"dataset": cfg["name"], "seed": seed, "data_sha256": fingerprints,
                "splits": {name: {"source": "train" if name in ("fit", "selection", "calibration") else "test",
                                  "size": len(ids), "indices": ids, "sha256": stable_hash(ids)}
                           for name, ids in indices.items()}}
    return {name: IndexedSubset(sources[name], ids, name) for name, ids in indices.items()}, manifest


def make_loader(dataset, config: dict, shuffle: bool = False, seed: int | None = None):
    return DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=shuffle,
                      num_workers=int(config.get("num_workers", 0)), drop_last=False,
                      generator=torch.Generator().manual_seed(int(seed if seed is not None else config["seed"])))
