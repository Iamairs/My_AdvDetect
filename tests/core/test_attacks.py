from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from torch import nn
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from advdetect.attacks import build_attack


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)


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


if __name__=="__main__":unittest.main()
