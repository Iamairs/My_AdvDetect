from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from research.mechanisms.common import StageBackbone, confidence_scores, pooled_features
from .qp import solve_simplex_qp


def fit_transform(features, dimension, floor=1e-5):
    x = features.double()
    mean = x.mean(0)
    if dimension is None:
        components = torch.eye(x.shape[1], dtype=x.dtype)
    else:
        _, _, vh = torch.linalg.svd(x - mean, full_matrices=False)
        components = vh[:min(int(dimension), x.shape[1])].T
    projected = (x-mean) @ components
    return {'mean': mean.float(), 'components': components.float(),
            'std': projected.std(0, unbiased=False).clamp_min(floor).float()}


def transform(features, params):
    return ((features - params['mean'].to(features)) @ params['components'].to(features)) / params['std'].to(features)


def choose_medoids(x, count, seed, iterations=10):
    """类内真实样本 k-medoids：最远点初始化，再最小化簇内距离和。"""
    if len(x) < count:
        raise ValueError(f'Reference candidates {len(x)} < requested {count}')
    dist = torch.cdist(x.double(), x.double()).square().numpy()
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(len(x)))]
    nearest = dist[:, chosen[0]].copy()
    for _ in range(1, count):
        nearest[chosen] = -1
        idx = int(nearest.argmax()); chosen.append(idx)
        nearest = np.minimum(nearest, dist[:, idx])
    for _ in range(iterations):
        assignment = dist[:, chosen].argmin(1)
        updated = []
        for j, old in enumerate(chosen):
            members = np.flatnonzero(assignment == j)
            updated.append(int(members[dist[np.ix_(members, members)].sum(0).argmin()]) if len(members) else old)
        if updated == chosen:
            break
        chosen = updated
    if len(set(chosen)) != count:
        raise RuntimeError('Medoid identities are not unique')
    return torch.tensor(chosen)


class SharedReferenceDetector:
    modes = ('shared', 'independent', 'shuffled', 'random_reference', 'last_layer', 'msp', 'margin')
    def __init__(self, model, config, device):
        self.model, self.cfg, self.device = model, config, device
        self.view = StageBackbone(model)
        self.stages = config.get('stages', [2, 3, 4])
        self.transforms, self.banks = [], {}
        self.reference_ids = {}
        self.permutations = []

    def fit(self, dataset, full_config):
        features, labels, ids, _ = pooled_features(self.model, dataset, full_config, self.stages, self.device)
        self.transforms = [fit_transform(x, self.cfg.get('pca_dim', 32)) for x in features]
        z = [transform(x, t) for x, t in zip(features, self.transforms)]
        joined = torch.cat(z, 1)
        rng = torch.Generator().manual_seed(int(full_config['seed']) + 20)
        k, nclasses = int(self.cfg['references_per_class']), int(full_config['model']['num_classes'])
        medoids, randoms = [], []
        for c in range(nclasses):
            candidates = torch.where(labels.eq(c))[0]
            chosen = choose_medoids(joined[candidates], k, int(full_config['seed']) + c)
            medoids.append(candidates[chosen])
            randoms.append(candidates[torch.randperm(len(candidates), generator=rng)[:k]])
        self.permutations = [torch.arange(k)] + [torch.randperm(k, generator=rng) for _ in self.stages[1:]]
        for name, selected in [('shared', medoids), ('random_reference', randoms)]:
            selected = torch.stack(selected)
            self.reference_ids[name] = ids[selected]
            self.banks[name] = [a[selected].to(self.device) for a in z]  # [classes, refs, dim]
        self.banks['shuffled'] = [bank[:, p].clone() for bank, p in zip(self.banks['shared'], self.permutations)]
        self._grams = {}
        return {'reference_ids': {k: v.tolist() for k, v in self.reference_ids.items()},
                'permutations': [v.tolist() for v in self.permutations],
                'layer_names': [f'backbone.layer{s} (post residual addition/ReLU)' for s in self.stages],
                'dimensions': [v.shape[1] for v in z]}

    def features(self, images):
        logits, maps = self.view.forward(images)
        grid = int(self.cfg.get('spatial_grid', 1))
        values = [F.adaptive_avg_pool2d(maps[s-1], grid).flatten(1) for s in self.stages]
        return logits, [transform(a, t) for a, t in zip(values, self.transforms)]

    def objective(self, z, classes, bank='shared', layers=None, steps=None, return_terms=False):
        layers = list(range(len(z))) if layers is None else layers
        key = (bank, tuple(layers), str(z[0].device), str(z[0].dtype))
        if key not in self._grams:
            ds = [self.banks[bank][j].to(z[0]) for j in layers]
            rho = torch.as_tensor(self.cfg.get('layer_weights', [1/len(z)]*len(z)), device=z[0].device, dtype=z[0].dtype)[layers]
            rho = rho / rho.sum()
            a = sum(r * torch.bmm(d, d.transpose(1, 2))/d.shape[2] for r, d in zip(rho, ds))
            a = a + float(self.cfg.get('eta', .001)) * torch.eye(a.shape[-1], device=a.device, dtype=a.dtype)
            self._grams[key] = a.detach(), rho, ds
        a, rho, ds = self._grams[key]
        ds = [d[classes] for d in ds]
        b = sum(r * torch.bmm(d, z[j].unsqueeze(-1)).squeeze(-1)/d.shape[-1]
                for r, d, j in zip(rho, ds, layers))
        c = sum(r * z[j].square().mean(1) for r, j in zip(rho, layers))
        if return_terms:
            return a[classes], b, c
        return solve_simplex_qp(a[classes], b, c, int(steps or self.cfg['solver_steps']))

    def scores(self, images, classes=None, modes=None, steps=None):
        logits, z = self.features(images)
        classes = logits.argmax(1) if classes is None else classes
        modes = self.modes if modes is None else modes
        result = confidence_scores(logits)
        terms, groups = [], {}
        for mode in modes:
            if mode in ('msp', 'margin'):
                continue
            if mode == 'independent':
                rho = self.cfg.get('layer_weights', [1/len(z)]*len(z))
                groups[mode] = []
                for j, r in enumerate(rho):
                    groups[mode].append((len(terms), r))
                    terms.append(self.objective(z, classes, layers=[j], return_terms=True))
            elif mode == 'last_layer':
                groups[mode] = [(len(terms), 1.)]
                terms.append(self.objective(z, classes, layers=[len(z)-1], return_terms=True))
            else:
                groups[mode] = [(len(terms), 1.)]
                terms.append(self.objective(z, classes, bank=mode, return_terms=True))
        if terms:
            # Independent QPs share a launch stream by concatenating the batch axis.
            # No objective/cross-layer weight is changed by this acceleration.
            a, b, c = [torch.cat([t[i] for t in terms]) for i in range(3)]
            values = solve_simplex_qp(a, b, c, int(steps or self.cfg['solver_steps']))[1].split(len(images))
            for mode, group in groups.items():
                result[mode] = sum(weight*values[index] for index, weight in group)
        return {mode: result[mode] for mode in modes}

    def score(self, images, classes=None, mode='shared'):
        return self.scores(images, classes, [mode])[mode]

    def state(self):
        return {'transforms': self.transforms, 'banks': {k: [v.cpu() for v in vals] for k, vals in self.banks.items()},
                'reference_ids': self.reference_ids, 'permutations': self.permutations}

    def restore(self, state):
        self.transforms = state['transforms']; self.reference_ids = state['reference_ids']
        self.permutations = state['permutations']
        self.banks = {k: [v.to(self.device) for v in vals] for k, vals in state['banks'].items()}
        self._grams = {}
