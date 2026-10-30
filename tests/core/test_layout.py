"""目录迁移后的配置兼容与入口导入回归。"""
from pathlib import Path
import importlib.util
import sys
import unittest
import warnings

import yaml

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'pyproject.toml').is_file())
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from advdetect.utils.config import load_config


class LayoutTests(unittest.TestCase):
    def test_legacy_configs_match_new_paths_and_preserve_overrides(self):
        aliases = yaml.safe_load((ROOT/'configs/legacy_paths.yaml').read_text(encoding='utf-8'))
        for old, new in aliases.items():
            with self.subTest(old=old), warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always', FutureWarning)
                self.assertFalse((ROOT/old).exists())
                self.assertTrue((ROOT/new).is_file())
                self.assertEqual(load_config(old), load_config(new))
                self.assertEqual(len(caught), 1)
        old = 'configs/experiments/cifar10_mechanisms_p1.yaml'
        with self.assertWarns(FutureWarning):
            self.assertEqual(load_config(old, ['seed=7'])['seed'], 7)

    def test_every_config_inheritance_resolves(self):
        for path in (ROOT/'configs').rglob('*.yaml'):
            if path.name in {'legacy_paths.yaml', 'local.yaml'}:
                continue
            with self.subTest(config=path):
                self.assertIsInstance(load_config(path), dict)

    def test_relocated_scripts_import_without_running_experiments(self):
        scripts = [p for p in (ROOT/'scripts').rglob('*.py') if p.name!='__init__.py']
        self.assertGreater(len(scripts), 10)
        for i, path in enumerate(scripts):
            with self.subTest(script=path):
                spec = importlib.util.spec_from_file_location(f'_layout_import_{i}', path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, 'ROOT'):
                    self.assertEqual(module.ROOT, ROOT)
                self.assertTrue(callable(getattr(module, 'main', None)))


if __name__=='__main__':unittest.main()
