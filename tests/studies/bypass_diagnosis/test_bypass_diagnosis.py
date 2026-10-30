from pathlib import Path
import sys
import unittest
import numpy as np
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from research.r004_bypass_diagnosis.details import projection_energy,weight_metrics,percentile,image_digest


class BypassDiagnosisTests(unittest.TestCase):
    def test_spatial_projection_energy_and_nested_pca(self):
        delta=torch.ones(1,2,2,2)
        full=projection_energy(delta,torch.eye(2))
        self.assertAlmostEqual(full['gap_retained_fraction'].item(),1)
        self.assertAlmostEqual(full['pca32_retained_of_gap_fraction'].item(),1)
        partial=projection_energy(delta,torch.eye(2)[:,:1])
        self.assertAlmostEqual(partial['pca32_retained_of_gap_fraction'].item(),.5)
        delta=torch.tensor([[[[1.,-1.],[-1.,1.]]]])
        erased=projection_energy(delta,torch.eye(1))
        self.assertEqual(erased['gap_retained_fraction'].item(),0)

    def test_effective_reference_count_and_tied_percentile(self):
        w=torch.tensor([[.25,.25,.25,.25],[1.,0.,0.,0.]])
        result=weight_metrics(w)
        torch.testing.assert_close(result['effective_reference_count'],torch.tensor([4.,1.]))
        self.assertAlmostEqual(percentile(2,[1,2,2,3]),.5)
        self.assertIsNone(percentile(0,[]))

    def test_image_hash_is_layout_independent(self):
        x=torch.arange(12).reshape(3,2,2).float()
        self.assertEqual(image_digest(x),image_digest(x.transpose(1,2).contiguous().transpose(1,2)))


if __name__=='__main__':
    unittest.main()
