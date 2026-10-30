from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from scipy.optimize import minimize

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from advdetect.models.backbones import build_classifier
from research.mechanisms.adaptive import joint_pgd
from research.mechanisms.common import StageBackbone, development_data, distance_profile, fit_profile, union_metrics
from research.r002_shared_reference.qp import simplex_projection, solve_simplex_qp
from research.r002_shared_reference.detector import SharedReferenceDetector
from research.r003_evidence_intervention.detector import EvidenceInterventionDetector, matched_drop


class MechanismTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_development_splits_do_not_open_final_test_or_expose_calibration(self):
        from advdetect.utils.config import load_config
        config = load_config('configs/experiments/studies/mechanisms/cifar10_mechanisms_p1.yaml')
        class Source:
            targets = np.repeat(np.arange(10), 5000).tolist()
        with patch('research.mechanisms.common.CIFAR10', return_value=Source()) as dataset:
            data, manifest = development_data(config)
            dataset.assert_called_once()
            self.assertTrue(dataset.call_args.kwargs['train'])
        self.assertNotIn('final_calibration_reserved', data)
        self.assertNotIn('test', data)
        split = manifest['splits']
        self.assertEqual(len(data['dev_evaluate']), 1000)
        self.assertFalse(set(split['reference']['indices']) & set(split['fit_diagnostic']['indices']))
        self.assertFalse(set(split['gate']['indices']) & set(split['profile']['indices']))
        self.assertEqual(set(split['gate']['indices']) | set(split['profile']['indices']), set(split['model']['indices']))
        all_ids = [split[k]['indices'] for k in ['model', 'dev_threshold', 'dev_evaluate', 'dev_select', 'final_calibration_reserved']]
        self.assertEqual(sum(map(len, all_ids)), len(set(sum(all_ids, []))))

    def test_simplex_qp_against_independent_scipy_and_gradient(self):
        torch.manual_seed(71)
        d = torch.randn(5, 4, dtype=torch.double)
        a = d.T@d/5 + .1*torch.eye(4, dtype=torch.double)
        x = torch.randn(1, 5, dtype=torch.double, requires_grad=True)
        def value(t):
            return solve_simplex_qp(a, t@d/5, t.square().mean(1), 600)[1]
        b = (x.detach()@d/5)[0].numpy()
        oracle = minimize(lambda w: w@a.numpy()@w-2*b@w+x.detach().square().mean().item(),
                          np.ones(4)/4, jac=lambda w: 2*a.numpy()@w-2*b,
                          bounds=[(0, None)]*4, constraints={'type': 'eq', 'fun': lambda w: w.sum()-1,
                          'jac': lambda w: np.ones(4)}, method='SLSQP', options={'ftol': 1e-13, 'maxiter': 1000})
        self.assertTrue(oracle.success)
        w, v, gap = solve_simplex_qp(a, x@d/5, x.square().mean(1), 600)
        self.assertAlmostEqual(v.item(), oracle.fun, places=8)
        self.assertLess(gap.item(), 1e-8)
        self.assertTrue(torch.all(w >= 0))
        self.assertAlmostEqual(w.sum().item(), 1., places=12)
        self.assertTrue(torch.autograd.gradcheck(value, (x,), atol=1e-5))
        p = simplex_projection(torch.tensor([[-1., .2, 2.]]))
        torch.testing.assert_close(p, torch.tensor([[0., 0., 1.]]))

    def test_shared_independent_order_and_column_identity(self):
        # Two layers prefer opposite reference identities: independent fits zero, shared cannot.
        model = nn.Identity()
        detector = object.__new__(SharedReferenceDetector)
        detector.cfg = {'eta': .001, 'solver_steps': 1000}
        detector.banks = {'shared': [torch.tensor([[[0.], [1.]]], dtype=torch.double),
                                      torch.tensor([[[1.], [0.]]], dtype=torch.double)]}
        detector._grams = {}
        z = [torch.zeros(1, 1, dtype=torch.double)]*2
        classes = torch.zeros(1, dtype=torch.long)
        shared = detector.objective(z, classes)[1]
        free = sum(detector.objective(z, classes, layers=[i])[1]/2 for i in range(2))
        self.assertGreater((shared-free).item(), .2)
        detector.features = lambda _: (torch.zeros(1, 2, dtype=torch.double), z)
        scores = detector.scores(torch.zeros(1, 1), modes=['shared', 'independent', 'last_layer'])
        torch.testing.assert_close(scores['shared'], shared)
        torch.testing.assert_close(scores['independent'], free)
        permuted = [d[:, [1, 0]] for d in detector.banks['shared']]
        detector.banks['both_permuted'] = permuted
        torch.testing.assert_close(shared, detector.objective(z, classes, bank='both_permuted')[1])

    def test_backbone_resume_interventions_and_bn_unchanged(self):
        torch.manual_seed(3)
        model = build_classifier('cifar_resnet18', 3, 10).eval().requires_grad_(False)
        view = StageBackbone(model)
        x = torch.rand(2, 3, 32, 32)
        before = {k: v.clone() for k, v in model.state_dict().items()}
        full, maps = view.forward(x)
        torch.testing.assert_close(full, model(x))
        for s in (2, 3, 4):
            h = maps[s-1]
            saved = h.clone(); mu = h.mean((0, 2, 3), keepdim=True)
            keep_all = mu+torch.ones_like(mu)*(h-mu)
            drop_zero = h-0*torch.ones_like(mu)*(h-mu)
            torch.testing.assert_close(view.resume(keep_all, s), full, atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(view.resume(drop_zero, s), full)
            view.resume(mu+.5*(h-mu), s)
            torch.testing.assert_close(h, saved)
        for k, v in model.state_dict().items():
            torch.testing.assert_close(v, before[k])

    def test_energy_matching_gradient_and_invalid_control(self):
        h = torch.tensor([[[[1., 2.]], [[3., 4.]]]], dtype=torch.double, requires_grad=True)
        mu = torch.zeros(1, 2, 1, 1, dtype=torch.double)
        m = torch.tensor([[[[.2]], [[.8]]]], dtype=torch.double)
        b = m.flip(1)
        def f(t):
            return matched_drop(t, mu, m, b, .25)[0]
        self.assertTrue(torch.autograd.gradcheck(f, (h,)))
        _, _, err, valid = matched_drop(h, mu, m, b, .25)
        self.assertLess(err.item(), 1e-7); self.assertTrue(valid.item())
        out, _, _, valid = matched_drop(h, mu, m, torch.zeros_like(b), .5)
        self.assertFalse(valid.item()); self.assertTrue(torch.isfinite(out).all())

    def test_evidence_whole_input_gradient_and_target_class(self):
        torch.manual_seed(12)
        model = build_classifier('cifar_resnet18', 3, 3).double().eval().requires_grad_(False)
        detector = EvidenceInterventionDetector(model, {'stages': [4], 'alpha': .5, 'random_controls': 2}, torch.device('cpu'), 3)
        detector.theta.append(nn.Parameter(torch.randn(3, 512, dtype=torch.double), requires_grad=False))
        detector.means = [torch.zeros(1, 512, 1, 1, dtype=torch.double)]
        detector.permutations = [torch.stack([torch.stack([torch.randperm(512) for _ in range(3)]) for _ in range(3)])]
        detector.profiles['paired'] = {'mean': torch.zeros(3, 2, dtype=torch.double),
                                       'precision': torch.eye(2, dtype=torch.double).repeat(3, 1, 1)}
        x = torch.rand(1, 3, 32, 32, dtype=torch.double, requires_grad=True)
        target = torch.tensor([1])
        score = detector.score(x, target)
        gradient = torch.autograd.grad(score.sum(), x)[0]
        self.assertGreater(gradient.abs().max().item(), 0)
        direction = torch.randn_like(x); direction /= direction.norm()
        step = 1e-5
        finite = (detector.score(x.detach()+step*direction, target)-detector.score(x.detach()-step*direction, target))/(2*step)
        torch.testing.assert_close(finite, (gradient*direction).sum().reshape(1), atol=1e-5, rtol=2e-3)

    def test_profile_small_classes_and_union_denominator(self):
        torch.manual_seed(2)
        values = torch.randn(20, 2)
        labels = torch.arange(20)%2
        profile = fit_profile(values, labels, labels, 2)
        self.assertEqual(profile['fallback'], ['diagonal', 'diagonal'])
        self.assertTrue((distance_profile(values, labels, profile) >= 0).all())
        m = union_metrics([0., 0., 1.], [0, 0, 0], [0, 1, 0], [True, False, False], .5)
        self.assertEqual(m['jasr0_union'], .5)
        self.assertEqual(m['wrong_acceptance_union_all_including_original'], 2/3)

    def test_joint_pgd_real_checks_and_success_retention(self):
        class Model(nn.Module):
            def forward(self, x):
                v = x.flatten(1).mean(1)-.5
                return torch.stack([-v, v], 1)
        class Detector:
            def score(self, x, classes=None, mode='main'):
                return (x.flatten(1).mean(1)-.55).square()
        x = torch.full((4, 1, 1, 1), .45)
        labels = torch.zeros(4, dtype=torch.long)
        observations = []
        with torch.no_grad():
            result = joint_pgd(Model(), Detector(), x, labels, .01,
                               {'epsilon': .2, 'step_size': .02, 'steps': 20, 'restarts': 2,
                                'targets': 'all', 'lambdas': [.1, 1], 'seed': 4}, 'main',
                               lambda candidate, tag: observations.append(candidate.clone()))
        self.assertTrue(result['success'].all())
        self.assertTrue((Model()(result['images']).argmax(1) != labels).all())
        self.assertTrue((Detector().score(result['images']) <= .01).all())
        self.assertLessEqual((result['images']-x).abs().max().item(), .200001)
        union = torch.zeros(4, dtype=torch.bool)
        for candidate in observations:
            union |= Model()(candidate).argmax(1).ne(labels) & Detector().score(candidate).le(.01)
        self.assertTrue(torch.equal(union, result['success']))


if __name__ == '__main__':
    unittest.main()
