from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from advdetect.models.backbones import build_classifier
from research.r001_lr_innovation import LRInnovationDetector


class LRInnovationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_detector(self):
        model = build_classifier("cifar_resnet18", 3, 10).eval().requires_grad_(False)
        config = {
            "name": "lr_innovation",
            "selected_layers": [
                "backbone.layer1.1.conv2",
                "backbone.layer2.1.conv2",
                "backbone.layer3.1.conv2",
            ],
            "input_size": [1, 3, 32, 32], "target_dim": 512,
            "hidden_dim": 8, "folds": 5, "margin_bins": 2,
            "min_group_samples": 2, "ridge": 1e-5,
        }
        return model, LRInnovationDetector(model, config, torch.device("cpu"))

    def test_residual_and_score_shapes_with_input_gradient(self):
        model, detector = self.make_detector()
        detector.regressor = detector._new_regressor(detector.input_dim, 512, 8, torch.device("cpu"))
        detector.stats = {
            "global_mean": torch.zeros(512),
            "global_variance": torch.ones(512),
            "margin_edges": torch.tensor([0.1]),
            "classes": {},
        }
        detector.fitted = True
        x = torch.rand(2, 3, 32, 32, requires_grad=True)
        scores = detector.score_all(x)
        self.assertEqual(scores["mse"].shape, (2,))
        self.assertTrue(torch.isfinite(scores["conditional_diag"]).all())
        grad = torch.autograd.grad(scores["directional_two_sided"].sum(), x)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(float(grad.abs().max()), 0)

    def test_state_restore_keeps_independent_detector_scores(self):
        model, detector = self.make_detector()
        detector.regressor = detector._new_regressor(detector.input_dim, 512, 8, torch.device("cpu"))
        detector.stats = {
            "global_mean": torch.zeros(512), "global_variance": torch.ones(512),
            "margin_edges": torch.tensor([]), "classes": {},
        }
        detector.fitted = True
        x = torch.rand(1, 3, 32, 32)
        expected = detector.score_all(x)["mse"]
        restored = LRInnovationDetector(model, detector.config, torch.device("cpu"))
        restored.restore(detector.state())
        self.assertTrue(torch.equal(expected, restored.score_all(x)["mse"]))

    def test_diagonal_whitening_uses_per_dimension_variance(self):
        residual = torch.tensor([[2.0, 2.0]])
        variance = torch.tensor([4.0, 1.0])
        score = (residual.square() / variance).mean(1)
        self.assertAlmostEqual(float(score), 2.5)


if __name__ == "__main__":
    unittest.main()
