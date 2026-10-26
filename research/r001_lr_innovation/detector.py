from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from advdetect.datasets.cifar import stratified_partition


@dataclass
class ResidualBatch:
    residual: torch.Tensor
    target_feature: torch.Tensor
    logits: torch.Tensor
    predicted_class: torch.Tensor
    competitor_class: torch.Tensor
    margin: torch.Tensor
    confidence: torch.Tensor


class LayerRegressor(nn.Module):
    """Research-owned MLP matching LR's two-hidden-layer regressor shape."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LRInnovationDetector:
    """独立实现的 LR 回归器、out-of-fold 残差统计与创新分数。"""

    def __init__(self, model: nn.Module, config: dict, device: torch.device):
        self.model, self.config, self.device = model, config, device
        self.selected_layers = list(config["selected_layers"])
        self.hidden_dim = int(config.get("hidden_dim", 2048))
        self.target_dim = int(config.get("target_dim", 512))
        self.margin_bins = int(config.get("margin_bins", 10))
        self.min_group_samples = int(config.get("min_group_samples", 64))
        self.shrinkage = float(config.get("shrinkage", 0.1))
        self.ridge = float(config.get("ridge", 1e-5))
        self.folds = int(config.get("folds", 5))
        if self.folds < 2:
            raise ValueError("crossfit folds must be at least 2")
        self._slices: list[tuple[int, int, int]] | None = None
        self.layer_paths: list[str] = []
        self.input_dim: int | None = None
        self.regressor: LayerRegressor | None = None
        self.stats: dict | None = None
        self.fitted = False
        self._infer_slices()

    def _infer_slices(self) -> None:
        modules = dict(self.model.named_modules())
        self.layer_paths = [name if name in modules else name.removeprefix("backbone.")
                            for name in self.selected_layers]
        missing = [name for name, path in zip(self.selected_layers, self.layer_paths) if path not in modules]
        if missing:
            raise ValueError(f"LR innovation layers not found: {missing}")
        captures: dict[str, torch.Tensor] = {}
        handles = [modules[path].register_forward_hook(
            lambda _m, _i, out, layer=name: captures.__setitem__(layer, out))
            for name, path in zip(self.selected_layers, self.layer_paths)]
        shape = tuple(self.config.get("input_size", [1, 3, 32, 32]))
        try:
            with torch.no_grad():
                self.model(torch.zeros(shape, device=self.device))
        finally:
            for handle in handles:
                handle.remove()
        self._slices = []
        for name in self.selected_layers:
            output = captures[name]
            flat_dim = int(output.flatten(1).shape[1])
            start, end = int(flat_dim * 0.2), int(flat_dim * 0.8)
            self._slices.append((flat_dim, start, end))
        self.input_dim = sum(end - start for _, start, end in self._slices)

    def _features(self, images: torch.Tensor, return_logits: bool = False):
        if self._slices is None:
            raise RuntimeError("Feature slices are not initialized")
        modules = dict(self.model.named_modules())
        captured: dict[str, torch.Tensor] = {}
        handles = [modules[path].register_forward_hook(
            lambda _m, _i, out, layer=name: captured.__setitem__(layer, out))
            for name, path in zip(self.selected_layers, self.layer_paths)]
        try:
            logits, maps = self.model.forward_features(images)
            parts = []
            for name, (_, start, end) in zip(self.selected_layers, self._slices):
                if name not in captured:
                    raise RuntimeError(f"Selected layer produced no activation: {name}")
                parts.append(captured[name].flatten(1)[:, start:end])
            vector = torch.cat(parts, dim=1)
            backbone = getattr(self.model, "backbone", self.model)
            pool = getattr(backbone, "pool", None) or getattr(backbone, "global_pool", None)
            if pool is None:
                raise AttributeError("Target backbone must expose pool or global_pool")
            target = pool(maps[-1]).flatten(1)
        finally:
            for handle in handles:
                handle.remove()
        return (vector, target, logits) if return_logits else (vector, target)

    @staticmethod
    def _new_regressor(input_dim: int, output_dim: int, hidden_dim: int, device):
        return LayerRegressor(input_dim, output_dim, hidden_dim).to(device)

    def _train_one(self, fit_loader, selection_loader, epochs: int, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        regressor = self._new_regressor(self.input_dim, self.target_dim, self.hidden_dim, self.device)
        optimizer = torch.optim.Adam(regressor.parameters(), lr=float(self.config.get("lr", 3e-4)))
        history = []
        best_state, best_loss, best_epoch = None, float("inf"), None
        for epoch in range(epochs):
            regressor.train()
            total, count = 0.0, 0
            for images, labels, _ in fit_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    vector, target, logits = self._features(images, True)
                    keep = logits.argmax(1).eq(labels) if self.config.get("correct_only", True) else torch.ones_like(labels, dtype=torch.bool)
                    vector, target = vector[keep], target[keep]
                if not len(vector):
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss = (regressor(vector) - target).square().mean()
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * len(vector)
                count += len(vector)
            selection_loss, selection_count = self._selection_loss(regressor, selection_loader)
            row = {"epoch": epoch + 1, "fit_mse": total / max(count, 1), "fit_samples": count,
                   "selection_mse": selection_loss, "selection_samples": selection_count}
            history.append(row)
            if selection_loss < best_loss:
                best_loss, best_epoch = selection_loss, epoch + 1
                best_state = deepcopy(regressor.state_dict())
        if best_state is None:
            raise RuntimeError("LR innovation training had no usable clean-correct samples")
        regressor.load_state_dict(best_state)
        return regressor.eval().requires_grad_(False), history, best_epoch, best_loss

    def _selection_loss(self, regressor, loader):
        regressor.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                vector, target, logits = self._features(images, True)
                keep = logits.argmax(1).eq(labels) if self.config.get("correct_only", True) else torch.ones_like(labels, dtype=torch.bool)
                if keep.any():
                    total += float((regressor(vector[keep]) - target[keep]).square().mean(1).sum())
                    count += int(keep.sum())
        return total / count if count else float("inf"), count

    def fit(self, fit_dataset, selection_loader, loader_factory, seed: int = 42) -> dict:
        """5-fold OOF statistics, shared epoch selection, then full-fit regressor."""
        epochs = int(self.config.get("epochs", 5))
        labels = [fit_dataset.source.targets[i] for i in fit_dataset.indices]
        fold_indices: list[list[int]] = []
        # Repeated stratified partitions make each held-out fold class-balanced.
        remaining = list(range(len(fit_dataset)))
        for fold in range(self.folds - 1):
            size = len(fit_dataset) // self.folds
            before = remaining
            pos, rest_pos = stratified_partition([labels[i] for i in before], size, seed + fold)
            fold_indices.append([before[i] for i in pos])
            remaining = [before[i] for i in rest_pos]
        fold_indices.append(remaining)
        self.fold_indices = fold_indices

        # Select one common epoch by mean selection MSE across fold models.
        fold_histories = []
        fold_best_epochs = []
        residuals, classes, margins = [], [], []
        for fold, held_out in enumerate(fold_indices):
            train_ids = [i for i in range(len(fit_dataset)) if i not in set(held_out)]
            fold_dataset = torch.utils.data.Subset(fit_dataset, train_ids)
            fold_loader = loader_factory(fold_dataset, shuffle=True, seed=seed + fold)
            model, history, best_epoch, _ = self._train_one(fold_loader, selection_loader, epochs, seed + fold)
            fold_histories.append(history)
            fold_best_epochs.append(best_epoch)
            held_loader = loader_factory(torch.utils.data.Subset(fit_dataset, held_out), shuffle=False, seed=seed)
            r, c, m = self._collect_residuals(model, held_loader)
            residuals.append(r); classes.append(c); margins.append(m)
            del model
        self._fit_statistics(torch.cat(residuals), torch.cat(classes), torch.cat(margins))

        mean_by_epoch = [float(np.mean([h[e]["selection_mse"] for h in fold_histories])) for e in range(epochs)]
        selected_epoch = int(np.argmin(mean_by_epoch)) + 1
        full_loader = loader_factory(fit_dataset, shuffle=True, seed=seed + 500)
        self.regressor, final_history, _, _ = self._train_fixed_epochs(full_loader, selected_epoch, seed + 500)
        self.fitted = True
        return {"protocol": "5fold_out_of_fold", "folds": self.folds,
                "fold_sizes": [len(x) for x in fold_indices], "fold_selection_histories": fold_histories,
                "fold_indices": fold_indices,
                "fold_best_epochs": fold_best_epochs, "mean_selection_mse_by_epoch": mean_by_epoch,
                "selected_epoch": selected_epoch, "final_fit_history": final_history,
                "statistics_samples": int(len(torch.cat(residuals))),
                "selected_layers": self.selected_layers, "layer_paths": self.layer_paths,
                "input_dim": self.input_dim,
                "target_dim": self.target_dim}

    def _train_fixed_epochs(self, loader, epochs, seed):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if self.device.type == "cuda": torch.cuda.manual_seed_all(seed)
        regressor = self._new_regressor(self.input_dim, self.target_dim, self.hidden_dim, self.device)
        optimizer = torch.optim.Adam(regressor.parameters(), lr=float(self.config.get("lr", 3e-4)))
        history = []
        for epoch in range(epochs):
            regressor.train(); total = count = 0
            for images, labels, _ in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                with torch.no_grad():
                    vector, target, logits = self._features(images, True)
                    keep = logits.argmax(1).eq(labels) if self.config.get("correct_only", True) else torch.ones_like(labels, dtype=torch.bool)
                    vector, target = vector[keep], target[keep]
                if not len(vector): continue
                optimizer.zero_grad(set_to_none=True)
                loss = (regressor(vector) - target).square().mean(); loss.backward(); optimizer.step()
                total += float(loss.detach()) * len(vector); count += len(vector)
            history.append({"epoch": epoch + 1, "fit_mse": total / max(count, 1), "fit_samples": count})
        return regressor.eval().requires_grad_(False), history, epochs, None

    def _collect_residuals(self, regressor, loader):
        all_r, all_c, all_m = [], [], []
        regressor.eval()
        with torch.no_grad():
            for images, labels, _ in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                vector, target, logits = self._features(images, True)
                predicted = logits.argmax(1)
                keep = predicted.eq(labels) if self.config.get("correct_only", True) else torch.ones_like(labels, dtype=torch.bool)
                if not keep.any(): continue
                top = logits.topk(min(2, logits.shape[1]), dim=1).values
                margin = top[:, 0] - top[:, 1] if logits.shape[1] > 1 else top[:, 0]
                residual = target - regressor(vector)
                all_r.append(residual[keep].cpu()); all_c.append(predicted[keep].cpu()); all_m.append(margin[keep].cpu())
        if not all_r:
            raise RuntimeError("No clean-correct samples available for residual statistics")
        return torch.cat(all_r), torch.cat(all_c), torch.cat(all_m)

    def _fit_statistics(self, residuals, classes, margins):
        r = residuals.double()
        global_mean, global_var = r.mean(0), r.var(0, unbiased=False).clamp_min(self.ridge)
        edges = torch.quantile(margins.double(), torch.linspace(0, 1, self.margin_bins + 1)[1:-1])
        class_stats = {}
        for cls in torch.unique(classes).tolist():
            mask_c = classes.eq(cls)
            rc = r[mask_c]
            mu_c, var_c = rc.mean(0), rc.var(0, unbiased=False).clamp_min(self.ridge)
            bins = torch.bucketize(margins.double(), edges)
            bin_stats = {}
            for b in range(self.margin_bins):
                mask = mask_c & bins.eq(b)
                n = int(mask.sum())
                if n < self.min_group_samples:
                    continue
                rb = r[mask]
                alpha = n / (n + self.shrinkage * self.min_group_samples)
                mu = alpha * rb.mean(0) + (1 - alpha) * mu_c
                var = alpha * rb.var(0, unbiased=False) + (1 - alpha) * var_c
                bin_stats[str(b)] = {"count": n, "mean": mu.float(), "variance": var.clamp_min(self.ridge).float()}
            class_stats[str(cls)] = {"count": int(mask_c.sum()), "mean": mu_c.float(),
                                     "variance": var_c.float(), "bins": bin_stats}
        self.stats = {"global_mean": global_mean.float(), "global_variance": global_var.float(),
                      "margin_edges": edges.float(), "classes": class_stats,
                      "samples": len(r), "feature_dim": r.shape[1]}

    def residual_batch(self, images: torch.Tensor) -> ResidualBatch:
        if not self.fitted or self.regressor is None:
            raise RuntimeError("Fit or restore LRInnovationDetector before scoring")
        vector, target, logits = self._features(images, True)
        top_values, top_indices = logits.topk(min(2, logits.shape[1]), dim=1)
        margin = top_values[:, 0] - top_values[:, 1] if logits.shape[1] > 1 else top_values[:, 0]
        return ResidualBatch(target - self.regressor(vector), target, logits, top_indices[:, 0],
                             top_indices[:, 1] if logits.shape[1] > 1 else top_indices[:, 0],
                             margin, logits.softmax(1).amax(1))

    def _context_stats(self, predicted, margin):
        global_mu = self.stats["global_mean"].to(margin.device)
        global_var = self.stats["global_variance"].to(margin.device)
        edges = self.stats["margin_edges"].to(margin.device)
        bins = torch.bucketize(margin.detach(), edges)
        means, variances = [], []
        for cls, bin_id in zip(predicted.detach().cpu().tolist(), bins.cpu().tolist()):
            entry = self.stats["classes"].get(str(cls))
            if entry is None:
                means.append(global_mu); variances.append(global_var); continue
            mu = entry["mean"].to(margin.device); var = entry["variance"].to(margin.device)
            group = entry["bins"].get(str(bin_id))
            if group is not None:
                mu = group["mean"].to(margin.device); var = group["variance"].to(margin.device)
            means.append(mu); variances.append(var)
        return torch.stack(means), torch.stack(variances)

    def score_all(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = self.residual_batch(images)
        r = batch.residual
        global_mu = self.stats["global_mean"].to(r.device)
        global_var = self.stats["global_variance"].to(r.device)
        cond_mu, cond_var = self._context_stats(batch.predicted_class, batch.margin)
        centered = r - global_mu
        conditional = r - cond_mu
        scores = {
            "mse": r.square().mean(1),
            "centered_mse": centered.square().mean(1),
            "diag_whiten": (centered.square() / (global_var + self.ridge)).mean(1),
            "conditional_diag": (conditional.square() / (cond_var + self.ridge)).mean(1),
        }
        head = getattr(getattr(self.model, "backbone", self.model), "fc", None)
        if head is None or not isinstance(head, nn.Linear) or head.weight.shape[1] != r.shape[1]:
            scores["directional_two_sided"] = torch.full_like(scores["mse"], float("nan"))
            scores["directional_one_sided"] = torch.full_like(scores["mse"], float("nan"))
            return scores
        idx = batch.predicted_class
        comp = batch.competitor_class
        direction = head.weight[idx] - head.weight[comp]
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-12)
        projected = (direction * conditional).sum(1)
        projected_var = (direction.square() * cond_var).sum(1).clamp_min(self.ridge)
        scores["directional_two_sided"] = projected.square() / projected_var
        scores["directional_one_sided"] = -projected / projected_var.sqrt()
        return scores

    def score_differentiable(self, images: torch.Tensor, mode: str) -> torch.Tensor:
        scores = self.score_all(images)
        if mode not in scores:
            raise ValueError(f"Unknown LR innovation score mode: {mode}")
        return scores[mode]

    def state(self) -> dict:
        if not self.fitted or self.regressor is None or self.stats is None:
            raise RuntimeError("Cannot serialize an unfitted detector")
        return {"selected_layers": self.selected_layers, "input_dim": self.input_dim,
                "target_dim": self.target_dim, "slices": self._slices,
                "regressor": {k: v.detach().cpu() for k, v in self.regressor.state_dict().items()},
                "stats": self.stats, "config": self.config}

    def restore(self, state: dict) -> None:
        if state["selected_layers"] != self.selected_layers or state["input_dim"] != self.input_dim:
            raise ValueError("Saved LR innovation feature configuration differs")
        if state.get("layer_paths", self.layer_paths) != self.layer_paths:
            raise ValueError("Saved LR innovation layer paths differ")
        self.regressor = self._new_regressor(self.input_dim, self.target_dim, self.hidden_dim, self.device)
        self.regressor.load_state_dict(state["regressor"])
        self.regressor.eval().requires_grad_(False)
        self._slices = [tuple(x) for x in state["slices"]]
        self.stats = state["stats"]
        self.fitted = True
