"""固定目标类的联合 Linf PGD。部署核验与优化目标分开。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .common import margin


def joint_pgd(model, detector, images, labels, threshold, config, mode, observer=None):
    """返回逐原图累计的真实成功及见证；observer 可对公共候选流作跨方法评价。

    用固定目标选择门控/参考，但每一步以真实 argmax 选择部署分数。
    不通过目标损失值推断成功；候选包括原图和所有迭代点。
    """
    if threshold is None:
        raise ValueError('A finite development threshold is required for a meaningful joint attack')
    epsilon, alpha = float(config['epsilon']), float(config['step_size'])
    if not 0 < epsilon <= 1 or alpha <= 0 or int(config['steps']) < 1:
        raise ValueError('Invalid Linf PGD configuration')
    original = images.detach()
    with torch.no_grad():
        logits = model(original)
        nclasses = logits.shape[1]
        order = logits.masked_fill(F.one_hot(labels, nclasses).bool(), -torch.inf).argsort(1, descending=True)
    target_count = nclasses-1 if config.get('targets', 'all') == 'all' else int(config['targets'])
    target_count = min(target_count, nclasses-1)
    successful = torch.zeros(len(images), dtype=torch.bool, device=images.device)
    best = original.clone()
    best_score = torch.full((len(images),), torch.inf, device=images.device)
    fallback_loss = torch.full_like(best_score, torch.inf)
    trace = []

    @torch.no_grad()
    def inspect(candidate, tag):
        prediction = model(candidate).argmax(1)
        score = detector.score(candidate, mode=mode)
        valid = ((candidate-original).flatten(1).abs().amax(1) <= epsilon+1e-6)
        valid &= candidate.flatten(1).amin(1) >= 0
        valid &= candidate.flatten(1).amax(1) <= 1
        hit = prediction.ne(labels) & score.le(threshold) & valid & torch.isfinite(score)
        improved = hit & ((~successful) | score.lt(best_score))
        best[improved] = candidate[improved]
        best_score[improved] = score[improved]
        successful.logical_or_(hit)
        if observer is not None:
            observer(candidate.detach(), tag)
        return int(successful.sum())

    inspect(original, {'kind': 'original'})
    generator = torch.Generator(device=images.device).manual_seed(int(config.get('seed', 42)))
    for target_rank in range(target_count):
        targets = order[:, target_rank]
        for weight in config.get('lambdas', [.01, .1, 1, 10, 100]):
            for restart in range(int(config.get('restarts', 5))):
                noise = (2*torch.rand(images.shape, generator=generator, device=images.device)-1)*epsilon
                candidate = (original+noise).clamp(0, 1)
                for step in range(int(config['steps'])+1):
                    tag = {'target_rank': target_rank, 'lambda': weight, 'restart': restart, 'step': step}
                    hits = inspect(candidate, tag)
                    if step == int(config['steps']):
                        trace.append({**tag, 'union_success_count': hits})
                        break
                    with torch.enable_grad():
                        candidate = candidate.detach().requires_grad_(True)
                        logits = model(candidate)
                        classification = F.relu(-margin(logits, targets)+float(config.get('kappa', .1)))
                        scores = detector.score(candidate, classes=targets, mode=mode)
                        loss = classification + float(weight)*F.relu(scores-threshold)/float(config.get('score_scale', 1.))
                        if not torch.isfinite(loss).all():
                            raise RuntimeError('Non-finite joint objective')
                        grad = torch.autograd.grad(loss.sum(), candidate)[0]
                    if not torch.isfinite(grad).all():
                        raise RuntimeError('Non-finite input gradient')
                    with torch.no_grad():
                        better = (~successful) & loss.lt(fallback_loss)
                        best[better] = candidate[better]
                        fallback_loss[better] = loss[better]
                        candidate = candidate-alpha*grad.sign()
                        candidate = torch.minimum(torch.maximum(candidate, original-epsilon), original+epsilon).clamp(0, 1)
    # The last update has already been inspected; previously successful witnesses survive later failures.
    return {'images': best.detach(), 'success': successful.detach(), 'trace': trace}
