from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from advdetect.evaluation.metrics import calibrate_threshold, detection_metrics


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)


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


if __name__=="__main__":unittest.main()
