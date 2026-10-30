from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms import Compose, Resize, ToTensor

from advdetect.datasets.cifar import IndexedSubset, make_loader, stratified_partition
from advdetect.utils.artifacts import stable_hash
from advdetect.utils.config import resolve_path


def nested_split(ids, labels, size, seed):
    a, b = stratified_partition(np.asarray(labels)[ids], size, seed)
    return [ids[i] for i in a], [ids[i] for i in b]


def development_data(config):
    """只打开官方训练文件；最终 calibration/test 均不用于 P0/P1。"""
    cfg, plan = config['dataset'], config['mechanism_split']
    transform = ToTensor()
    if cfg.get('resize'):
        transform = Compose([Resize((cfg['resize'], cfg['resize'])), ToTensor()])
    cls = {'cifar10': CIFAR10, 'cifar100': CIFAR100}[cfg['name']]
    source = cls(str(resolve_path(cfg['root'])), train=True, download=False, transform=transform)
    seed = int(config['seed'])
    model, rest = stratified_partition(source.targets, plan['model'], seed)
    dev, calibration = nested_split(rest, source.targets, plan['dev'], seed + 1)
    gate, profile = nested_split(model, source.targets, plan['gate'], seed + 2)
    ref, rest_model = nested_split(model, source.targets, plan['reference'], seed + 3)
    fit, _ = nested_split(rest_model, source.targets, plan['fit_diagnostic'], seed + 4)
    threshold, remaining_dev = nested_split(dev, source.targets, plan['dev_threshold'], seed + 5)
    evaluate, select = nested_split(remaining_dev, source.targets, plan['dev_evaluate'], seed + 6)
    ids = dict(model=model, gate=gate, profile=profile, reference=ref, fit_diagnostic=fit,
               dev_threshold=threshold, dev_evaluate=evaluate, dev_select=select,
               final_calibration_reserved=calibration)
    for names in [('model', 'final_calibration_reserved', 'dev_threshold', 'dev_evaluate', 'dev_select'),
                  ('gate', 'profile'), ('reference', 'fit_diagnostic')]:
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                if set(ids[a]) & set(ids[b]):
                    raise AssertionError(f'Overlapping splits: {a}, {b}')
    manifest = {'source': 'official_training_only', 'classifier_training_indices': 'unknown',
                'scope': 'exploratory_P0_P1', 'final_calibration_used': False, 'official_test_used': False,
                'splits': {k: {'size': len(v), 'indices': v, 'sha256': stable_hash(v)} for k, v in ids.items()}}
    # These are reserved identities only; callers cannot request the final set by accident.
    return {k: IndexedSubset(source, v, k) for k, v in ids.items()
            if k not in ('model', 'final_calibration_reserved')}, manifest


class StageBackbone:
    """在完整 stage 输出处拆分，支持项目 CIFAR ResNet18 与 timm ResNet50。"""
    def __init__(self, model):
        self.model = model
        self.net = getattr(model, 'backbone', model)
        if not all(hasattr(self.net, f'layer{i}') for i in range(1, 5)):
            raise TypeError('A four-stage ResNet is required')

    def forward(self, images):
        h = self.model.normalize(images) if hasattr(self.model, 'normalize') else images
        h = self.net.bn1(self.net.conv1(h))
        h = (self.net.act1 if hasattr(self.net, 'act1') else self.net.relu)(h)
        if hasattr(self.net, 'maxpool'):
            h = self.net.maxpool(h)
        maps = []
        for i in range(1, 5):
            h = getattr(self.net, f'layer{i}')(h)
            maps.append(h)
        return self.head(h), maps

    def head(self, h):
        if hasattr(self.net, 'forward_head'):
            return self.net.forward_head(h)
        pool = self.net.pool if hasattr(self.net, 'pool') else self.net.avgpool
        return self.net.fc(pool(h).flatten(1))

    def resume(self, h, stage):
        if stage not in (1, 2, 3, 4):
            raise ValueError('stage must be 1..4')
        for i in range(stage + 1, 5):
            h = getattr(self.net, f'layer{i}')(h)
        return self.head(h)


def margin(logits, classes, temperature=1.):
    selected = logits.gather(1, classes[:, None]).squeeze(1)
    other = logits.masked_fill(F.one_hot(classes, logits.shape[1]).bool(), -torch.inf).amax(1)
    return (selected - other) / temperature


def confidence_scores(logits):
    classes = logits.argmax(1)
    return {'msp': -logits.softmax(1).amax(1), 'margin': -margin(logits, classes)}


@torch.no_grad()
def pooled_features(model, dataset, config, stages, device):
    view = StageBackbone(model)
    features = [[] for _ in stages]
    labels, ids, predictions = [], [], []
    for images, y, index in make_loader(dataset, config):
        logits, maps = view.forward(images.to(device))
        for chunks, stage in zip(features, stages):
            grid = int(config.get('shared', {}).get('spatial_grid', 1))
            chunks.append(F.adaptive_avg_pool2d(maps[stage - 1], grid).flatten(1).cpu())
        labels.append(y); ids.append(index); predictions.append(logits.argmax(1).cpu())
    return [torch.cat(v) for v in features], torch.cat(labels), torch.cat(ids), torch.cat(predictions)


def fit_profile(values, classes, labels, nclasses, beta=.1, ridge=1e-5):
    """正确正常类的均值/收缩协方差；缺类不静默制造统计量。"""
    means, precisions, counts, fallbacks = [], [], [], []
    good = classes.eq(labels)
    pooled = values[good].double()
    if len(pooled) < 2:
        raise ValueError('Insufficient correct profile data')
    pooled_cov = torch.cov(pooled.T).reshape(values.shape[1], values.shape[1])
    d = values.shape[1]
    for c in range(nclasses):
        v = values[good & classes.eq(c)].double()
        counts.append(len(v))
        if not len(v):
            raise ValueError(f'No correct profile examples for class {c}; change the design explicitly')
        means.append(v.mean(0))
        if len(v) < 2:
            cov = pooled_cov.diag().diag(); fallback = 'pooled_diagonal'
        else:
            cov = torch.cov(v.T).reshape(d, d)
            fallback = 'diagonal' if len(v) < max(100, 10 * d) else 'shrinkage'
            cov = cov.diag().diag() if fallback == 'diagonal' else (1-beta)*cov + beta*cov.diag().diag()
        cov = cov + ridge * torch.eye(d, dtype=torch.float64)
        precisions.append(torch.linalg.inv(cov)); fallbacks.append(fallback)
    return {'mean': torch.stack(means).float(), 'precision': torch.stack(precisions).float(),
            'counts': counts, 'fallback': fallbacks}


def distance_profile(values, classes, profile):
    mean = profile['mean'].to(values.device)[classes]
    precision = profile['precision'].to(values.device)[classes]
    delta = values - mean
    return torch.einsum('bi,bij,bj->b', delta, precision, delta)


def union_metrics(clean_score, labels, predictions, union, tau):
    clean_score = np.asarray(clean_score)
    y, cp = np.asarray(labels), np.asarray(predictions)
    correct = y == cp
    accepted = clean_score <= tau
    # delta=0 is a candidate for all originals, including natural errors.
    union = np.asarray(union, dtype=bool) | ((~correct) & accepted)
    return {'num_samples': len(y), 'num_clean_correct': int(correct.sum()),
            'clean_fpr_all_normal': float((~accepted).mean()), 'normal_coverage': float(accepted.mean()),
            'accepted_clean_error_rate': float(((~correct) & accepted).sum()/accepted.sum()) if accepted.any() else None,
            'jasr0_union': float(union[correct].mean()) if correct.any() else None,
            'wrong_acceptance_union_all_including_original': float(union.mean())}
