from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from research.r005_layer_contribution.family import LayerFamily
from research.r006_witness_geometry.geometry import decompose, nearest_reference, percentile, within_class_pair_distances


class WitnessGeometryTests(unittest.TestCase):
    def test_explicit_residuals_match_deployed_score_for_both_weight_modes(self):
        torch.set_num_threads(2); torch.manual_seed(26)
        model = nn.Module()
        for s in range(1, 5): setattr(model, f'layer{s}', nn.Identity())
        bank = [torch.randn(2, 4, d, dtype=torch.double) for d in [3, 4, 5]]
        family = LayerFamily(model, {'r': {'grid': 1, 'transforms': [], 'banks': {'shared': bank}}}, {},
            {'representations': ['r'], 'layer_sets': [[4], [2, 3, 4]], 'solver_steps': 1000, 'eta': .1}, torch.device('cpu'))
        z = {'r': [torch.randn(2, d, dtype=torch.double) for d in [3, 4, 5]]}
        y = torch.tensor([0, 1]); scores, _ = family.score_z(z, y)
        details = decompose(family, z, y)
        for name in family.variants:
            torch.testing.assert_close(details[name]['score'], scores[name])
            torch.testing.assert_close(details[name]['explicit_score'], scores[name])
            for l, item in details[name]['layers'].items():
                recon = torch.bmm(item['weights'][:, None], bank[l-2][y]).squeeze(1)
                torch.testing.assert_close(item['residual_l2'], (z['r'][l-2]-recon).norm(dim=1))
                torch.testing.assert_close(item['weights'].sum(1), torch.ones(2, dtype=torch.double))
                self.assertTrue(item['weights'].ge(0).all())

    def test_normal_scales_do_not_pair_an_image_with_itself_and_use_class_bank(self):
        z = torch.tensor([[0.], [2.], [100.], [103.]])
        classes = torch.tensor([0, 0, 1, 1])
        pair = within_class_pair_distances(z, classes.numpy(), 20, 41)
        np.testing.assert_equal(pair[0], np.full(20, 2.))
        np.testing.assert_equal(pair[1], np.full(20, 3.))
        bank = torch.tensor([[[1.], [7.]], [[101.], [200.]]])
        torch.testing.assert_close(nearest_reference(z, classes, bank), torch.tensor([1., 1., 1., 2.], dtype=torch.double))
        np.testing.assert_equal(percentile([1., 2., 2., 4.], [0., 2., 5.]), [0., 50., 100.])
        with self.assertRaises(ValueError): percentile([], [0.])


if __name__ == '__main__': unittest.main()
