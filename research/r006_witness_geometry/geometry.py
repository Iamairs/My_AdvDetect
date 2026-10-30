from __future__ import annotations

import numpy as np
import torch

from research.r002_shared_reference.qp import solve_simplex_qp


@torch.no_grad()
def decompose(family, z, classes, names=None):
    """原部署QP与权重；显式残差用float64复算，避免平方展开的抵消。"""
    names = list(family.variants) if names is None else list(names)
    keys = list(dict.fromkeys(k for name in names for k, _ in family.variants[name]['terms']))
    terms = [family.problem_terms(z, classes, k) for k in keys]
    a, b, c = [torch.cat([t[j] for t in terms]) for j in range(3)]
    w, score, gap = solve_simplex_qp(a, b, c, family.cfg['solver_steps'])
    ws = dict(zip(keys, w.split(len(classes))))
    scores = dict(zip(keys, score.split(len(classes))))
    gaps = dict(zip(keys, gap.split(len(classes))))
    result = {}
    for name in names:
        spec = family.variants[name]
        rep = family.representations[spec['representation']]
        layers = {}
        regularizer = torch.zeros(len(classes), dtype=torch.float64, device=classes.device)
        for key, factor in spec['terms']:
            problem = family.problems[key]
            weights = ws[key].double()
            regularizer += factor * float(family.cfg['eta']) * weights.square().sum(1)
            for layer in problem['layers']:
                bank = rep['banks'][problem['bank']][layer - 2][classes].double()
                reconstruction = torch.bmm(weights[:, None, :], bank).squeeze(1)
                value = z[spec['representation']][layer - 2].double()
                residual = (value - reconstruction).norm(dim=1)
                layers[layer] = {
                    'residual_l2': residual, 'residual_rms': residual / value.shape[1]**.5,
                    'mse': residual.square() / value.shape[1], 'reconstruction': reconstruction,
                    'weights': weights, 'bank': problem['bank'],
                    'objective_factor': factor / len(problem['layers'])}
        total = sum(factor * scores[key] for key, factor in spec['terms'])
        explicit = sum(t['objective_factor'] * t['mse'] for t in layers.values()) + regularizer
        result[name] = {'score': total, 'regularizer': regularizer, 'explicit_score': explicit,
                        'decomposition_absolute_error': (explicit-total.double()).abs(), 'layers': layers,
                        'max_relative_fw_gap': torch.stack([gaps[k]/(1+scores[k].abs()) for k, _ in spec['terms']]).amax(0)}
    return result


def describe(values):
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError('Reference distribution must be nonempty and finite')
    return {'n': len(x), 'mean': float(x.mean()),
            **{f'q{q:02d}': float(np.quantile(x, q/100)) for q in [1, 5, 25, 50, 75, 90, 95, 99]}}


def percentile(reference, values):
    """中秩经验CDF；并列值不武断放到最高端。"""
    ref = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    if not len(ref):
        raise ValueError('Empty normal comparison class')
    value = np.asarray(values, dtype=np.float64)
    return 50 * (np.searchsorted(ref, value, side='left') + np.searchsorted(ref, value, side='right')) / len(ref)


def within_class_pair_distances(z, classes, count, seed):
    """固定seed抽类内不同正常图片对；抽样次数不是独立图片数。"""
    rng = np.random.default_rng(seed)
    z = z.double().cpu()
    labels = np.asarray(classes)
    result = {}
    for c in np.unique(labels):
        indices = np.flatnonzero(labels == c)
        if len(indices) < 2:
            raise ValueError('Need two distinct normal images per class')
        i = rng.integers(len(indices), size=count)
        j = rng.integers(len(indices)-1, size=count)
        j += j >= i
        result[int(c)] = (z[indices[i]]-z[indices[j]]).norm(dim=1).numpy()
    return result


def nearest_reference(z, classes, bank):
    result = torch.empty(len(z), device=z.device, dtype=torch.float64)
    for c in classes.unique().tolist():
        mask = classes.eq(c)
        result[mask] = torch.cdist(z[mask].double(), bank[c].double()).amin(1)
    return result
