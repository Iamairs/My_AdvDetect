"""Official Layer Regression architecture/selection, adapted to CIFAR ResNet18.

Vendored upstream code stays unchanged. This adapter provides streaming training,
reproducible selection, gradient-preserving hooks and workspace serialization.
"""
from __future__ import annotations

import importlib.util
import random
from copy import deepcopy
from time import perf_counter

import torch

from ...utils.config import resolve_path
from ...datasets.cifar import make_loader


def upstream():
    path = resolve_path("third_party/layer_regression/lr_modules.py")
    spec = importlib.util.spec_from_file_location("advdetect_upstream_lr", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class LayerRegressionDetector:
    def __init__(self, model, config: dict, device: torch.device):
        self.model, self.config, self.device = model, config, device
        official = upstream()
        input_size = tuple(config.get("input_size", [1, 3, 32, 32]))
        automator = official.LRAutomator(model, input_size=input_size)
        previous = random.getstate()
        random.seed(int(config.get("layer_seed", 42)))
        try:
            self.layer_config = automator.generate_config(
                num_layers=3, fixed_layers=config.get("selected_layers"))
        finally:
            random.setstate(previous)
        if len(self.layer_config["selected_names"]) != 3:
            raise ValueError("LR must capture exactly three layers")
        # The paper gives explicit ResNet50 slices in Appendix F.  The
        # upstream automatic slicer is retained for other adaptations.
        if config.get("paper_resnet50_slices", False):
            self.layer_config["slices"] = [
                lambda x: x[:, :5, :28, :28].flatten(1),
                lambda x: x[:, :50, :7, :7].flatten(1),
                lambda x: x[:, :10, :14, :14].flatten(1),
            ]
            self.layer_config["mlp_input_dim"] = 5 * 28 * 28 + 50 * 7 * 7 + 10 * 14 * 14
        target_dim = int(config.get("target_dim", 512))
        self.selection = {
            "selected_layers": self.layer_config["selected_names"],
            "input_dim": self.layer_config["mlp_input_dim"], "target_dim": target_dim,
            "hidden_dim": int(config.get("hidden_dim", 2048)),
            "slicing": ("paper_resnet50_appendix_f" if config.get("paper_resnet50_slices", False)
                        else "flatten_middle_60_percent"),
            "layer_seed": int(config.get("layer_seed", 42)),
        }
        self.regressor = official.LayerRegressionMLP(
            input_dim=self.selection["input_dim"], output_dim=target_dim,
            hidden_dim=self.selection["hidden_dim"]).to(device)
        self.fitted = False

    def feature_pair(self, images, return_logits=False):
        captured, handles = {}, []

        def capture(name):
            def hook(_module, _inputs, output):
                # Retain the tensor reference as upstream does, but preserve gradients.
                # In-place activation behavior therefore matches upstream numerically.
                captured[name] = output
            return hook

        modules = dict(self.model.named_modules())
        try:
            for name in self.layer_config["selected_names"]:
                handles.append(modules[name].register_forward_hook(capture(name)))
            logits, maps = self.model.forward_features(images)
            backbone = getattr(self.model, "backbone", self.model)
            pool = getattr(backbone, "pool", None) or getattr(backbone, "global_pool", None)
            if pool is None:
                raise AttributeError("Target backbone must expose pool or global_pool")
            if hasattr(backbone, "forward_head"):
                target = backbone.forward_head(maps[-1], pre_logits=True)
            else:
                target = pool(maps[-1]).flatten(1)
            vector = torch.cat([slicer(captured[name]) for name, slicer in
                                zip(self.layer_config["selected_names"], self.layer_config["slices"])], 1)
        finally:
            for handle in handles:
                handle.remove()
        return (vector, target, logits) if return_logits else (vector, target)

    @torch.no_grad()
    def selection_loss(self, loader, correct_only):
        total, count = 0.0, 0
        for images, labels, _ in loader:
            v, target, logits = self.feature_pair(images.to(self.device), return_logits=True)
            mask = logits.argmax(1).eq(labels.to(self.device)) if correct_only else torch.ones(len(images), device=self.device, dtype=torch.bool)
            if mask.any():
                total += (self.regressor(v[mask]) - target[mask]).square().mean(1).sum().item()
                count += mask.sum().item()
        return total / count if count else None, count

    def fit(self, fit_loader, selection_loader, device):
        epochs = int(self.config.get("epochs", 1))
        batch_size = int(self.config.get("fit_batch_size", 16))
        if min(epochs, batch_size) < 1:
            raise ValueError("epochs and fit_batch_size must be positive")
        correct_only = bool(self.config.get("correct_only", True))
        loader_cfg = {"batch_size": batch_size, "seed": int(self.config.get("shuffle_seed", 42)), "num_workers": 0}
        loader = make_loader(fit_loader.dataset, loader_cfg, shuffle=True)
        fused = bool(self.config.get("fused_adam", True)) and device.type == "cuda"
        optimizer = torch.optim.Adam(self.regressor.parameters(), lr=float(self.config.get("lr", 3e-4)),
                                     fused=fused, foreach=False if not fused else None)
        self.regressor.requires_grad_(True)
        history = []
        best_state = None
        best_selection = float("inf")
        best_epoch = None
        started = perf_counter()
        print(f"LR selection: {self.selection}", flush=True)
        for epoch in range(epochs):
            self.regressor.train()
            total_loss, count = 0.0, 0
            for batch, (images, labels, _) in enumerate(loader):
                with torch.no_grad():
                    vector, target, logits = self.feature_pair(images.to(device), return_logits=True)
                    mask = logits.argmax(1).eq(labels.to(device)) if correct_only else torch.ones(len(images), device=device, dtype=torch.bool)
                    vector, target = vector[mask], target[mask]
                if not len(vector):
                    continue
                optimizer.zero_grad(set_to_none=True)
                loss = (self.regressor(vector) - target).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite LR fitting loss")
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(vector)
                count += len(vector)
                if batch % 100 == 0:
                    print(f"LR epoch {epoch+1}/{epochs} batch {batch}/{len(loader)} MSE={loss.item():.6f} elapsed={perf_counter()-started:.1f}s", flush=True)
            if not count:
                raise ValueError("No correctly classified clean fitting samples")
            self.regressor.eval()
            validation, selected_count = self.selection_loss(selection_loader, correct_only)
            if validation is None:
                raise ValueError("No correctly classified clean selection samples")
            history.append({"epoch": epoch + 1, "fit_mse": total_loss / count, "fit_samples": count,
                            "selection_mse": validation, "selection_samples": selected_count})
            if validation < best_selection:
                best_selection = validation
                best_epoch = epoch + 1
                best_state = deepcopy(self.regressor.state_dict())
            print(f"LR epoch completed: {history[-1]}", flush=True)
        if best_state is None:
            raise RuntimeError("LR did not produce a valid selection checkpoint")
        self.regressor.load_state_dict(best_state)
        self.regressor.eval().requires_grad_(False)
        self.fitted = True
        return {"requires_fitting": True, "history": history, "selection": self.selection,
                "checkpoint_policy": "best_selection_mse", "best_epoch": best_epoch,
                "best_selection_mse": best_selection, "fused_adam": fused,
                "elapsed_seconds": perf_counter() - started,
                "implementation": "official_algorithm_cifar10_resnet18_adaptation"}

    def residual(self, images):
        if not self.fitted:
            raise RuntimeError("Fit or restore LR before scoring")
        vector, target = self.feature_pair(images)
        return target - self.regressor(vector)

    def score(self, images):
        return self.residual(images).square().mean(1)

    def state(self):
        if not self.fitted:
            raise RuntimeError("Cannot save unfitted LR")
        return {"selection": self.selection,
                "regressor": {k: v.detach().cpu() for k, v in self.regressor.state_dict().items()}}

    def restore(self, state):
        if self.selection != state["selection"]:
            raise ValueError("Saved LR layer/slicing configuration differs")
        self.regressor.load_state_dict(state["regressor"])
        self.regressor.eval().requires_grad_(False)
        self.fitted = True
