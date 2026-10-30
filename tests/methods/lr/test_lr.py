from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from advdetect.utils.config import load_config
from advdetect.models.backbones import build_classifier
from advdetect.detectors.baselines.lr import LayerRegressionDetector, upstream
from advdetect.detectors.factory import load_detector, save_detector


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)


    def test_lr_data_protocol_keeps_calibration_in_training_split(self):
        config = load_config("configs/experiments/baselines/lr/cifar10_lr_full.yaml")
        from advdetect.datasets.cifar import build_data
        data, manifest = build_data({**config, "limits": {"fit": 100, "selection": 80, "calibration": 20, "test": 100}})
        self.assertEqual(manifest["splits"]["calibration"]["source"], "train")
        self.assertEqual(manifest["splits"]["test"]["source"], "test")
        self.assertFalse(set(manifest["splits"]["fit"]["indices"]) & set(manifest["splits"]["selection"]["indices"]))
        self.assertFalse(set(manifest["splits"]["fit"]["indices"]) & set(manifest["splits"]["calibration"]["indices"]))


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


if __name__=="__main__":unittest.main()
