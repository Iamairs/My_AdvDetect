from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from ..attacks import build_attack
from ..datasets import build_data, make_loader
from ..datasets.cifar import IndexedSubset
from ..detectors import build_detector, load_detector, save_detector
from ..evaluation import calibrate_threshold, detection_metrics
from ..models import load_classifier
from ..utils.artifacts import choose_device, save_json, seed_everything, sha256_file, stable_hash
from ..utils.config import resolve_path


class Experiment:
    def __init__(self, config: dict, directory: Path):
        self.config, self.directory = config, directory
        seed_everything(int(config["seed"]))
        torch.set_num_threads(int(config.get("cpu_threads", 4)))
        self.device = choose_device(config["device"])
        self.data, self.manifest = build_data(config)
        save_json(directory / "splits.json", self.manifest)
        self.model, checkpoint_meta = load_classifier(config, self.device)
        self.model_sha256 = sha256_file(resolve_path(config["model"]["checkpoint"]))
        save_json(directory / "classifier.json", {**checkpoint_meta, "sha256": self.model_sha256})
        detector_code = Path(__file__).parents[1] / "detectors" / (
            "proposed/jad.py" if config["detector"]["name"] == "jad" else "baselines/lr.py")
        binding_files = [detector_code]
        if config["detector"]["name"] == "lr":
            binding_files.append(resolve_path("third_party/layer_regression/lr_modules.py"))
        self.binding = {
            "classifier_sha256": self.model_sha256,
            "classifier_code_sha256": sha256_file(Path(__file__).parents[1] / "models/backbones.py"),
            "detector_code_sha256": stable_hash({str(p): sha256_file(p) for p in binding_files}),
            "data_sha256": self.manifest["data_sha256"],
            "fit_indices_sha256": self.manifest["splits"]["fit"]["sha256"],
            "selection_indices_sha256": self.manifest["splits"]["selection"]["sha256"],
            "calibration_indices_sha256": self.manifest["splits"]["calibration"]["sha256"],
        }

    def loader(self, split: str):
        return make_loader(self.data[split], self.config)

    def inspect(self):
        images, labels, _ = next(iter(self.loader("test")))
        images = images.to(self.device).requires_grad_(True)
        logits = self.model(images)
        gradient = torch.autograd.grad(logits.square().mean(), images)[0]
        if not torch.isfinite(logits).all() or not torch.isfinite(gradient).all() or not gradient.abs().max() > 0:
            raise RuntimeError("Classifier outputs or input gradients are invalid")
        imported = resolve_path("docs/resource_manifest.json")
        resource_checks = []
        # The imported-resource manifest covers the original CIFAR-10 assets;
        # CIFAR-100 is downloaded from torchvision and has its own data hash.
        if imported.is_file() and self.config["dataset"]["name"] == "cifar10":
            for record in json.loads(imported.read_text(encoding="utf-8"))["files"]:
                path = resolve_path(record["destination"])
                resource_checks.append({"path": record["destination"],
                                        "matches": path.is_file() and sha256_file(path) == record["sha256"]})
        result = {"device": str(self.device), "batch_shape": list(images.shape),
                  "logits_shape": list(logits.shape), "input_gradient_nonzero": True,
                  "batch_accuracy": float(logits.argmax(1).cpu().eq(labels).float().mean()),
                  "splits": {k: len(v) for k, v in self.data.items()},
                  "imported_resources": resource_checks,
                  "ready": all(r["matches"] for r in resource_checks)}
        save_json(self.directory / "resources.json", result)
        if not result["ready"]:
            raise RuntimeError("Imported resource hashes differ; see resources.json")
        return result

    @torch.no_grad()
    def evaluate_classifier(self, split="test"):
        if split == "full-test":
            source = self.data["test"].source
            dataset = IndexedSubset(source, list(range(len(source))), "full-test")
            loader = make_loader(dataset, self.config)
        else:
            loader = self.loader(split)
        rows, correct = [], 0
        for images, labels, ids in loader:
            predictions = self.model(images.to(self.device)).argmax(1).cpu()
            correct += predictions.eq(labels).sum().item()
            rows.extend(zip(ids.tolist(), labels.tolist(), predictions.tolist()))
        result = {"split": split, "num_samples": len(rows), "accuracy": correct / len(rows)}
        self.write_csv("classifier_predictions.csv", ["sample_id", "label", "prediction"], rows)
        save_json(self.directory / "metrics.json", result)
        return result

    def fit(self):
        seed_everything(int(self.config["seed"]))
        cfg = self.config["detector"]
        detector = build_detector(self.model, cfg, self.device)
        history = detector.fit(self.loader("fit"), self.loader("selection"), self.device)
        path = resolve_path("checkpoints/detectors") / self.directory.name / "detector.pt"
        save_detector(path, detector, cfg, self.binding)
        save_json(self.directory / "fit.json", {**history, "checkpoint": str(path), "binding": self.binding})
        return detector, path

    def restore(self, path):
        path = resolve_path(path)
        detector, cfg = load_detector(path, self.model, self.device, self.binding)
        if cfg != self.config["detector"]:
            raise ValueError("Configured detector differs from checkpoint configuration")
        return detector, path

    @torch.no_grad()
    def score_loader(self, detector, split):
        scores, predictions, labels, ids = [], [], [], []
        for images, y, index in self.loader(split):
            images = images.to(self.device)
            scores.append(detector.score(images).cpu())
            predictions.append(self.model(images).argmax(1).cpu())
            labels.append(y)
            ids.append(index)
        return {"scores": torch.cat(scores), "predictions": torch.cat(predictions),
                "labels": torch.cat(labels), "indices": torch.cat(ids)}

    def calibrate(self, detector, checkpoint):
        seed_everything(int(self.config["seed"]) + 100)
        clean = self.score_loader(detector, "calibration")
        evaluation_cfg = self.config["evaluation"]
        raw_size = len(clean["scores"])
        correct = clean["predictions"].eq(clean["labels"])
        if bool(evaluation_cfg.get("calibrate_on_correct_only", True)):
            clean = {key: value[correct] for key, value in clean.items()}
        if not len(clean["scores"]):
            raise RuntimeError("Calibration split contains no correctly classified samples")
        result = calibrate_threshold(clean["scores"].numpy(), float(evaluation_cfg["target_fpr"]))
        result.update({"detector_sha256": sha256_file(checkpoint),
                       "calibration_indices_sha256": self.manifest["splits"]["calibration"]["sha256"],
                       "calibration_raw_size": raw_size,
                       "calibration_correct_only": bool(evaluation_cfg.get("calibrate_on_correct_only", True))})
        self.write_csv("calibration_scores.csv", ["sample_id", "label", "prediction", "score"],
                       zip(clean["indices"].tolist(), clean["labels"].tolist(),
                           clean["predictions"].tolist(), clean["scores"].tolist()))
        save_json(self.directory / "calibration.json", result)
        return result

    def generate(self):
        cfg = self.config["attack"]
        import torchattacks
        binding = {"classifier_sha256": self.model_sha256,
                   "classifier_code_sha256": sha256_file(Path(__file__).parents[1] / "models/backbones.py"),
                   "data_sha256": self.manifest["data_sha256"]["test"],
                   "split_sha256": self.manifest["splits"]["test"]["sha256"],
                   "attack": cfg, "seed": int(self.config["seed"]) + 200,
                   "batch_size": int(self.config["batch_size"]),
                   "torch_version": str(torch.__version__), "device": str(self.device),
                   "torchattacks_version": torchattacks.__version__,
                   "cache_format": "advdetect_attack_stream_v1", "storage_dtype": "float32",
                   "batch_seed_mode": "per_batch",
                   "workflow_sha256": sha256_file(Path(__file__)),
                   "implementation_sha256": sha256_file(Path(__file__).parents[1] / "attacks/standard/library.py")}
        cache = resolve_path("cache/attacks") / f"{stable_hash(binding)}.stream"
        manifest_path = cache / "manifest.json"
        adv_path = cache / "adv_images.float32.bin"
        labels_path = cache / "labels.int64.npy"
        indices_path = cache / "indices.int64.npy"
        total_samples = len(self.data["test"])
        resize = int(self.config["dataset"].get("resize", 32))
        shape = (total_samples, 3, resize, resize)
        batch_size = int(self.config["batch_size"])
        total_batches = (total_samples + batch_size - 1) // batch_size
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("binding") != binding:
                raise ValueError(f"Attack cache binding differs: {cache}")
            if tuple(manifest.get("shape", ())) != shape or manifest.get("dtype") != "float32":
                raise ValueError(f"Attack cache shape or dtype differs: {cache}")
            if manifest.get("status") == "complete":
                self._check_stream_cache_files(cache, manifest, labels_path, indices_path, adv_path)
                save_json(self.directory / "attack_cache.json", {"path": str(cache), "reused": True,
                                                                   "binding": binding, "format": manifest["format"]})
                return manifest, cache
            if manifest.get("status") != "writing":
                raise ValueError(f"Unknown attack cache status: {cache}")
            completed_batches = int(manifest.get("completed_batches", 0))
        else:
            cache.mkdir(parents=True, exist_ok=False)
            expected_labels = np.asarray(self.data["test"].source.targets, dtype=np.int64)[self.data["test"].indices]
            expected_indices = np.asarray(self.data["test"].indices, dtype=np.int64)
            np.save(labels_path, expected_labels)
            np.save(indices_path, expected_indices)
            stream = np.memmap(adv_path, mode="w+", dtype=np.float32, shape=shape)
            stream.flush()
            del stream
            completed_batches = 0
            manifest = {"format": "advdetect_attack_stream_v1", "binding": binding,
                        "status": "writing", "dtype": "float32", "shape": list(shape),
                        "batch_size": batch_size, "total_batches": total_batches,
                        "completed_batches": 0, "completed_samples": 0}
            save_json(manifest_path, manifest)
        self._check_stream_cache_files(cache, manifest, labels_path, indices_path, adv_path)
        stream = np.memmap(adv_path, mode="r+", dtype=np.float32, shape=shape)
        attack = build_attack(self.model, cfg)
        for batch, (images, y, index) in enumerate(self.loader("test")):
            if batch < completed_batches:
                continue
            seed_everything(int(self.config["seed"]) + 200 + batch)
            images, y_device = images.to(self.device), y.to(self.device)
            with torch.enable_grad():
                adv = attack(images, y_device)
            start = batch * batch_size
            end = start + len(y)
            stream[start:end] = adv.detach().cpu().float().numpy()
            stream.flush()
            manifest.update({"completed_batches": batch + 1, "completed_samples": end})
            save_json(manifest_path, manifest)
            print(f"Attack {cfg['name']}: {min((batch+1)*batch_size, total_samples)}/{total_samples}", flush=True)
        del stream
        manifest.update({"status": "complete", "completed_batches": total_batches,
                         "completed_samples": total_samples})
        save_json(manifest_path, manifest)
        self._check_stream_cache_files(cache, manifest, labels_path, indices_path, adv_path)
        save_json(self.directory / "attack_cache.json", {"path": str(cache), "reused": False,
                                                           "binding": binding, "format": manifest["format"]})
        return manifest, cache

    @staticmethod
    def _check_stream_cache_files(cache, manifest, labels_path, indices_path, adv_path):
        if not (labels_path.is_file() and indices_path.is_file() and adv_path.is_file()):
            raise ValueError(f"Attack cache files are incomplete: {cache}")
        shape = tuple(manifest["shape"])
        expected_bytes = int(np.prod(shape)) * np.dtype(np.float32).itemsize
        if adv_path.stat().st_size != expected_bytes:
            raise ValueError(f"Attack cache size differs: {cache}")
        labels = np.load(labels_path, mmap_mode="r")
        indices = np.load(indices_path, mmap_mode="r")
        if tuple(labels.shape) != (shape[0],) or tuple(indices.shape) != (shape[0],):
            raise ValueError(f"Attack cache labels or indices shape differs: {cache}")

    @torch.no_grad()
    def evaluate(self, detector, checkpoint):
        calibration = self.calibrate(detector, checkpoint)
        cache_info, cache = self.generate()
        # Attack generation explicitly enables autograd internally.
        seed_everything(int(self.config["seed"]) + 300)
        clean_scores, adv_scores, cp, ap = [], [], [], []
        labels, ids = [], []
        stream = np.memmap(cache / "adv_images.float32.bin", mode="r", dtype=np.float32,
                           shape=tuple(cache_info["shape"]))
        cached_labels = np.load(cache / "labels.int64.npy", mmap_mode="r")
        cached_indices = np.load(cache / "indices.int64.npy", mmap_mode="r")
        batch_size = int(self.config["batch_size"])
        offset = 0
        max_linf = 0.0
        for images, y, index in self.loader("test"):
            size = len(y)
            end = offset + size
            adv_np = np.array(stream[offset:end], copy=True)
            if not np.array_equal(cached_labels[offset:end], y.numpy()) or not np.array_equal(cached_indices[offset:end], index.numpy()):
                raise ValueError("Attack cache labels or sample IDs differ")
            if not np.isfinite(adv_np).all() or adv_np.min() < 0 or adv_np.max() > 1:
                raise ValueError("Invalid attack image values")
            max_linf = max(max_linf, float(np.abs(adv_np - images.numpy()).max()))
            images, adv = images.to(self.device).float(), torch.from_numpy(adv_np).to(self.device).float()
            clean_scores.append(detector.score(images).cpu())
            adv_scores.append(detector.score(adv).cpu())
            cp.append(self.model(images).argmax(1).cpu())
            ap.append(self.model(adv).argmax(1).cpu())
            labels.append(y)
            ids.append(index)
            offset = end
        del stream
        if offset != len(self.data["test"]):
            raise ValueError(f"Attack cache sample count differs: {offset} != {len(self.data['test'])}")
        epsilon = float(self.config["attack"]["epsilon"])
        if max_linf > epsilon + 1e-6:
            raise ValueError(f"Attack exceeds configured Linf budget: {max_linf} > {epsilon}")
        cs, ads, cp, ap, labels, ids = map(torch.cat, (clean_scores, adv_scores, cp, ap, labels, ids))
        metrics = detection_metrics(cs.numpy(), ads.numpy(), cp.numpy(), ap.numpy(), labels.numpy(), calibration)
        metrics.update({"detector": self.config["detector"]["name"], "attack": self.config["attack"]["name"],
                        "detector_checkpoint": str(checkpoint), "attack_cache": str(cache),
                        "max_linf": max_linf,
                        "exploratory": bool(self.config.get("exploratory", True))})
        tau = float("inf") if calibration["threshold"] is None else calibration["threshold"]
        rows = zip(ids.tolist(), labels.tolist(), cp.tolist(), ap.tolist(), cs.tolist(), ads.tolist(),
                   ((cp == labels) & (ap != labels)).tolist(), (ads > tau).tolist())
        self.write_csv("scores.csv", ["sample_id", "label", "clean_prediction", "adv_prediction",
                                     "clean_score", "adv_score", "attack_success", "adv_rejected"], rows)
        np.savez_compressed(self.directory / "scores.npz", indices=ids.numpy(), labels=labels.numpy(),
                            clean_predictions=cp.numpy(), adv_predictions=ap.numpy(),
                            clean_scores=cs.numpy(), adv_scores=ads.numpy())
        save_json(self.directory / "metrics.json", metrics)
        return metrics

    def write_csv(self, filename, columns, rows):
        with (self.directory / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows)
