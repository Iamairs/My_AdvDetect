from __future__ import annotations

import numpy as np
import torch


class CandidatePool:
    """逐原图、逐方法累计同一候选集合；原图也进入集合。"""
    def __init__(self,images,labels,names,thresholds,epsilon):
        self.images=images.cpu()
        self.labels=labels.cpu().numpy()
        self.names=list(names)
        self.thresholds=np.asarray([thresholds[name]['threshold'] for name in names])
        self.epsilon=epsilon
        self.success=np.zeros((len(images),len(names)),dtype=bool)
        self.witness=self.images.unsqueeze(1).expand(-1,len(names),-1,-1,-1).clone()
        self.witness_score=np.full(self.success.shape,np.inf)
        self.candidate_counts=np.zeros(len(images),dtype=np.int64)

    def consume(self,images,indices,predictions,scores):
        images=images.detach().cpu();indices=np.asarray(indices,dtype=int)
        pred=np.asarray(predictions)
        values=np.stack([np.asarray(scores[name]) for name in self.names],1)
        if not np.isfinite(values).all() or not torch.isfinite(images).all():
            raise ValueError('Non-finite candidate/score')
        norms=(images-self.images[indices]).flatten(1).abs().amax(1)
        if norms.max()>self.epsilon+1e-6 or images.min()<0 or images.max()>1:
            raise ValueError('Candidate is outside budget or pixel bounds')
        hits=(pred!=self.labels[indices])[:,None]&(values<=self.thresholds[None])
        # Multiple candidates can share an original ID in a streaming batch.
        for row,index in enumerate(indices):
            improved=hits[row]&(values[row]<self.witness_score[index])
            self.witness[index,improved]=images[row]
            self.witness_score[index,improved]=values[row,improved]
            self.success[index]|=hits[row]
            self.candidate_counts[index]+=1

    def unresolved(self,name,clean_correct):
        return np.flatnonzero(np.asarray(clean_correct)&~self.success[:,self.names.index(name)])
