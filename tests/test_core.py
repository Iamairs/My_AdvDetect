from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from advdetect.attacks import build_attack
from advdetect.datasets.cifar import stratified_partition
from advdetect.detectors.baselines.lr import LayerRegressionDetector, upstream
from advdetect.detectors.factory import load_detector, save_detector
from advdetect.evaluation.metrics import calibrate_threshold, detection_metrics
from advdetect.models.backbones import build_classifier
from advdetect.utils.config import load_config


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_splits_disjoint_reproducible_and_stratified(self):
        labels = np.repeat(np.arange(10), 11)
        a, b = stratified_partition(labels, 23, 42)
        self.assertFalse(set(a) & set(b))
        self.assertEqual(sorted(a + b), list(range(110)))
        self.assertEqual((a, b), stratified_partition(labels, 23, 42))
        counts = np.bincount(labels[a], minlength=10)
        self.assertLessEqual(counts.max() - counts.min(), 1)

    def test_lr_data_protocol_keeps_calibration_in_training_split(self):
        config = load_config("configs/experiments/cifar10_lr_full.yaml")
        from advdetect.datasets.cifar import build_data
        data, manifest = build_data({**config, "limits": {"fit": 100, "selection": 80, "calibration": 20, "test": 100}})
        self.assertEqual(manifest["splits"]["calibration"]["source"], "train")
        self.assertEqual(manifest["splits"]["test"]["source"], "test")
        self.assertFalse(set(manifest["splits"]["fit"]["indices"]) & set(manifest["splits"]["selection"]["indices"]))
        self.assertFalse(set(manifest["splits"]["fit"]["indices"]) & set(manifest["splits"]["calibration"]["indices"]))

    def test_configuration_inheritance_overrides_and_cycle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "base.yaml").write_text("a: {x: 1, y: 2}\n", encoding="utf-8")
            child = root / "child.yaml"
            child.write_text("extends: base.yaml\na: {x: 3}\n", encoding="utf-8")
            self.assertEqual(load_config(child, ["a.y=5"]), {"a": {"x": 3, "y": 5}})
            (root / "base.yaml").write_text("extends: child.yaml\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(child)

    def test_calibration_handles_small_samples_and_ties(self):
        self.assertIsNone(calibrate_threshold([1, 2, 3], 0.01)["threshold"])
        result = calibrate_threshold(np.arange(99), 0.05)
        self.assertEqual(result["threshold"], 94)
        self.assertEqual(calibrate_threshold(np.ones(100), 0.05)["empirical_calibration_fpr"], 0)

    def test_calibration_protocol_can_exclude_natural_errors(self):
        scores = np.asarray([0.1, 0.2, 0.3, 0.4, 10.0])
        predictions = np.asarray([0, 1, 0, 1, 2])
        labels = np.asarray([0, 1, 0, 1, 1])
        correct_scores = scores[predictions == labels]
        result = calibrate_threshold(correct_scores, 0.5)
        self.assertEqual(result["calibration_size"], 4)
        self.assertEqual(result["threshold"], 0.3)

    def test_joint_metric_excludes_natural_errors_from_attack_success(self):
        calibration = {"threshold": 0.5, "target_fpr": 0.05}
        result = detection_metrics([0, 0, 0], [0.1, 0.9, 0.1], [0, 1, 1], [1, 0, 0], [0, 1, 0], calibration)
        self.assertEqual(result["num_successful_attacks"], 2)
        self.assertEqual(result["joint_success_rate_on_clean_correct"], 0.5)
        self.assertEqual(result["joint_success_rate_all"], 1 / 3)
        self.assertEqual(result["accepted_wrong_rate_all"], 1 / 3)
        empty = detection_metrics([0], [0], [0], [0], [0], calibration)
        self.assertIsNone(empty["auroc_successful_only"])

    def test_library_pgd_obeys_budget_and_flips_known_linear_boundary(self):
        class Toy(nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = nn.Parameter(torch.zeros(1))

            def forward(self, x):
                z = (x.flatten(1).mean(1) - 0.5) * 10
                return torch.stack([-z, z], 1)
        model = Toy()
        images = torch.full((2, 1, 2, 2), 0.45)
        labels = torch.zeros(2, dtype=torch.long)
        cfg = {"name": "pgd", "epsilon": 0.1, "step_size": 0.05,
               "steps": 3, "random_start": True}
        adv = build_attack(model, cfg)(images, labels)
        self.assertLessEqual((adv - images).abs().max().item(), 0.100001)
        self.assertTrue(model(adv).argmax(1).ne(labels).all())
        self.assertTrue(torch.equal(images, torch.full_like(images, 0.45)))

    def test_autoattack_standard_keeps_all_four_library_components(self):
        import torchattacks
        model = nn.Sequential(nn.Flatten(), nn.Linear(4, 10))
        attack = build_attack(model, {"name": "autoattack", "epsilon": 8/255, "seed": 42})
        self.assertIsInstance(attack, torchattacks.AutoAttack)
        self.assertEqual([type(a).__name__ for a in attack._autoattack.attacks], ["APGD", "APGDT", "FAB", "Square"])

    def test_lr_restore_preserves_scores_and_input_gradients(self):
        torch.manual_seed(1)
        device = torch.device("cpu")
        model = build_classifier("cifar_resnet18", 3, 10).eval().requires_grad_(False)
        cfg = {"name": "lr", "hidden_dim": 8, "layer_seed": 42}
        detector = LayerRegressionDetector(model, cfg, device)
        detector.fitted = True  # Test serialization and differentiation, not fitting quality.
        x = torch.rand(2, 3, 32, 32, requires_grad=True)
        scores = detector.score(x)
        gradient = torch.autograd.grad(scores.sum(), x)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().max().item(), 0)
        official = upstream()
        with torch.no_grad():
            with official.UniversalActivationSaver(model, detector.layer_config) as saver:
                _, maps = model.forward_features(x)
                official_vector = saver.get_features()
                official_target = model.pool(maps[-1]).flatten(1)
            adapted_vector, adapted_target = detector.feature_pair(x)
            self.assertTrue(torch.equal(official_vector, adapted_vector))
            self.assertTrue(torch.equal(official_target, adapted_target))
            official_scores = (detector.regressor(official_vector) - official_target).square().mean(1)
            self.assertTrue(torch.allclose(scores, official_scores))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "detector.pt"
            save_detector(path, detector, cfg, {"model": "a"})
            restored, _ = load_detector(path, model, device, {"model": "a"})
            self.assertTrue(torch.allclose(scores, restored.score(x)))
            with self.assertRaises(ValueError):
                load_detector(path, model, device, {"model": "b"})

    def test_timm_resnet50_cifar100_paper_layer_mapping(self):
        model = build_classifier(
            "timm_resnet50", 3, 100, normalize_input=True,
            input_mean=(0.485, 0.456, 0.406), input_std=(0.229, 0.224, 0.225),
        ).eval()
        config = {
            "name": "lr", "hidden_dim": 8,
            "selected_layers": [
                "backbone.layer2.1.conv2",
                "backbone.layer3.0.conv2",
                "backbone.layer3.5.conv2",
            ],
            "input_size": [1, 3, 224, 224], "target_dim": 2048,
            "paper_resnet50_slices": True,
        }
        detector = LayerRegressionDetector(model, config, torch.device("cpu"))
        images = torch.rand(1, 3, 224, 224)
        vector, target, logits = detector.feature_pair(images, return_logits=True)
        self.assertEqual(tuple(vector.shape), (1, 8330))
        self.assertEqual(tuple(target.shape), (1, 2048))
        self.assertEqual(tuple(logits.shape), (1, 100))


if __name__ == "__main__":
    unittest.main()
