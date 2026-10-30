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


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)


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


if __name__=="__main__":unittest.main()
