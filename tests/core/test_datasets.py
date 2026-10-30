from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from advdetect.datasets.cifar import stratified_partition


class ProtocolTests(unittest.TestCase):
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


if __name__=="__main__":unittest.main()
