"""按方法寻找尚未被公共候选攻破的原图；完整梯度、真实判定和成功累计。"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from research.mechanisms.common import margin


def attack_batch(family,name,images,labels,indices,ordinary,calibration,config):
    tau=float(calibration['threshold'])
    scale=max(float(calibration['score_scale']),1e-8)
    epsilon=float(config['epsilon']);alpha=float(config['step_size'])
    original=images.detach()
    with torch.no_grad():
        logits=family.model(original)
        order=logits.masked_fill(F.one_hot(labels,logits.shape[1]).bool(),-torch.inf).argsort(1,descending=True)
    generator=torch.Generator(device=images.device).manual_seed(int(config['seed'])+int(indices[0])*997)
    solved=torch.zeros(len(images),dtype=torch.bool,device=images.device)
    best=original.clone()
    best_score=torch.full((len(images),),torch.inf,device=images.device)
    frames=[];frame_indices=[];events=[]
    gradient_steps=0

    def inspect(candidate,tag):
        nonlocal solved
        active=~solved
        with torch.no_grad():
            logits,values,_=family.evaluate(candidate,[name])
            score=values[name]
            norm=(candidate-original).flatten(1).abs().amax(1)
            valid=norm.le(epsilon+1e-6)&candidate.flatten(1).amin(1).ge(0)&candidate.flatten(1).amax(1).le(1)
            hit=logits.argmax(1).ne(labels)&score.le(tau)&valid&torch.isfinite(score)
            improve=hit&score.lt(best_score)
            best[improve]=candidate[improve];best_score[improve]=score[improve]
            # Only changed/unresolved rows are archived; solved rows stay frozen.
            frames.append(candidate[active].detach().cpu())
            frame_indices.append(np.asarray(indices)[active.cpu().numpy()])
            solved|=hit
            events.append({**tag,'newly_solved':int((hit&active).sum()),'cumulative_solved':int(solved.sum()),
                           'batch_size':len(images)})

    targets=min(int(config['targets']),logits.shape[1]-1)
    for rank in range(targets):
        for weight in config['lambdas']:
            for restart in range(int(config['restarts'])):
                if solved.all():
                    break
                # One standard PGD warm start, then reproducible random starts.
                if rank==0 and weight==config['lambdas'][0] and restart==0:
                    candidate=ordinary.detach().clone()
                    start_kind='ordinary_pgd_warm_start'
                else:
                    noise=(2*torch.rand(images.shape,generator=generator,device=images.device)-1)*epsilon
                    candidate=(original+noise).clamp(0,1)
                    start_kind='uniform_random'
                candidate[solved]=best[solved]
                target=order[:,rank]
                for step in range(int(config['steps'])+1):
                    tag={'target_rank':rank,'lambda':float(weight),'restart':restart,'step':step,'start':start_kind}
                    inspect(candidate,tag)
                    if solved.all() or step==int(config['steps']):
                        break
                    with torch.enable_grad():
                        x=candidate.detach().requires_grad_(True)
                        logits,values,_=family.evaluate(x,[name],classes=target)
                        loss=F.relu(-margin(logits,target)+float(config['kappa']))
                        loss=loss+float(weight)*F.relu(values[name]-tau)/scale
                        grad=torch.autograd.grad(loss[~solved].sum(),x)[0]
                    if not torch.isfinite(grad).all():
                        raise RuntimeError('Invalid joint input gradient')
                    with torch.no_grad():
                        updated=x-alpha*grad.sign()
                        updated=torch.minimum(torch.maximum(updated,original-epsilon),original+epsilon).clamp(0,1)
                        updated[solved]=best[solved]
                        candidate=updated.detach()
                    gradient_steps+=1
    return {'images':torch.cat(frames),'indices':np.concatenate(frame_indices),
            'events':events,'own_success':solved.cpu(),'best':best.cpu(),'gradient_steps':gradient_steps}
