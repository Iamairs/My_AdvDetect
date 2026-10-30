from __future__ import annotations

from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

from research.mechanisms.common import StageBackbone
from research.r002_shared_reference.detector import transform, choose_medoids
from research.r002_shared_reference.qp import solve_simplex_qp


def variant_name(representation, layers, mode):
    return f'{representation}_L'+''.join(map(str,layers))+f'_{mode}'


class LayerFamily:
    """唯一QP批量计算；所有方法共享冻结主模型及同层变换。"""
    def __init__(self,model,representations,reference_ids,config,device):
        self.model=model
        self.view=StageBackbone(model)
        self.representations=representations
        self.reference_ids=reference_ids
        self.cfg=config
        self.device=device
        self._grams={}
        self.variants={}
        self.problems={}
        for representation in config['representations']:
            for layers in config['layer_sets']:
                layers=tuple(layers)
                shared_key=self.add_problem(representation,layers)
                self.variants[variant_name(representation,layers,'shared')]={
                    'representation':representation,'layers':list(layers),'mode':'shared','terms':[(shared_key,1.)]}
                if len(layers)>1:
                    terms=[(self.add_problem(representation,(layer,)),1/len(layers)) for layer in layers]
                    self.variants[variant_name(representation,layers,'independent')]={
                        'representation':representation,'layers':list(layers),'mode':'independent','terms':terms}
        for representation in config.get('reselect_layer4',[]):
            key=self.add_problem(representation,(4,),'reselected')
            self.variants[variant_name(representation,[4],'reselected')]={
                'representation':representation,'layers':[4],'mode':'reselected','terms':[(key,1.)]}

    def add_problem(self,representation,layers,bank='shared'):
        key=f'{representation}|'+','.join(map(str,layers))+'|'+bank
        self.problems[key]={'representation':representation,'layers':list(layers),'bank':bank}
        return key

    def project(self,raw):
        result={}
        for name,rep in self.representations.items():
            result[name]=[transform(v,t) for v,t in zip(raw[rep['grid']],rep['transforms'])]
        return result

    def features(self,images):
        logits,maps=self.view.forward(images)
        raw={grid:[F.adaptive_avg_pool2d(maps[s-1],grid).flatten(1) for s in [2,3,4]] for grid in [1,2]}
        return logits,self.project(raw)

    def problem_terms(self,z,classes,key):
        problem=self.problems[key]
        rep=self.representations[problem['representation']]
        layers=problem['layers']
        values=z[problem['representation']]
        cache_key=(key,str(values[0].dtype),str(values[0].device))
        if cache_key not in self._grams:
            bank=rep['banks'][problem['bank']]
            ds=[bank[s-2].to(values[0]) for s in layers]
            gram=sum(torch.bmm(d,d.transpose(1,2))/d.shape[2] for d in ds)/len(ds)
            gram=gram+float(self.cfg.get('eta',.001))*torch.eye(gram.shape[-1],device=gram.device,dtype=gram.dtype)
            self._grams[cache_key]=(gram.detach(),ds)
        gram,ds=self._grams[cache_key]
        linear=sum(torch.bmm(d[classes],values[s-2].unsqueeze(-1)).squeeze(-1)/d.shape[2]
                   for d,s in zip(ds,layers))/len(layers)
        constant=sum(values[s-2].square().mean(1) for s in layers)/len(layers)
        return gram[classes],linear,constant

    def score_z(self,z,classes,names=None,steps=None):
        names=list(self.variants) if names is None else names
        keys=list(dict.fromkeys(key for name in names for key,_ in self.variants[name]['terms']))
        terms=[self.problem_terms(z,classes,key) for key in keys]
        a,b,c=[torch.cat([t[j] for t in terms]) for j in range(3)]
        _,values,gaps=solve_simplex_qp(a,b,c,int(steps or self.cfg['solver_steps']))
        values=dict(zip(keys,values.split(len(classes))))
        gaps=dict(zip(keys,gaps.split(len(classes))))
        result={name:sum(weight*values[key] for key,weight in self.variants[name]['terms']) for name in names}
        certificates={key:gaps[key]/(1+values[key].abs()) for key in keys}
        return result,certificates

    def evaluate(self,images,names=None,classes=None,steps=None):
        logits,z=self.features(images)
        classes=logits.argmax(1) if classes is None else classes
        scores,gaps=self.score_z(z,classes,names,steps)
        return logits,scores,gaps

    def score(self,images,classes=None,mode=None):
        if mode is None:
            raise ValueError('Specify a layer-study variant name')
        return self.evaluate(images,[mode],classes)[1][mode]

    def state(self):
        return {'representations':{name:{'grid':rep['grid'],
                       'transforms':[{k:v.cpu() for k,v in t.items()} for t in rep['transforms']],
                       'banks':{k:[v.cpu() for v in bank] for k,bank in rep['banks'].items()}}
                       for name,rep in self.representations.items()},
                'reference_ids':self.reference_ids,'config':self.cfg}

    @classmethod
    def restore(cls,model,state,device):
        representations={}
        for name,rep in state['representations'].items():
            representations[name]={'grid':rep['grid'],
                'transforms':[{k:v.to(device) for k,v in t.items()} for t in rep['transforms']],
                'banks':{k:[v.to(device) for v in bank] for k,bank in rep['banks'].items()}}
        return cls(model,representations,state['reference_ids'],state['config'],device)


def family_from_saved(model,states,config,device):
    representations={};ids={}
    for name in config['representations']:
        saved=states[name]
        state=saved['state']
        representations[name]={'grid':saved['config']['spatial_grid'],
            'transforms':[{k:v.to(device) for k,v in t.items()} for t in state['transforms']],
            'banks':{'shared':[v.to(device) for v in state['banks']['shared']]}}
        ids[name]={'shared':state['reference_ids']['shared']}
    # Reselected banks are added later; placeholders never participate in scoring before that.
    for name in config.get('reselect_layer4',[]):
        representations[name]['banks']['reselected']=representations[name]['banks']['shared']
        ids[name]['reselected']=ids[name]['shared']
    return LayerFamily(model,representations,ids,config,device)


@torch.no_grad()
def reselect_layer4(family,cached_reference,seed):
    for name in family.cfg.get('reselect_layer4',[]):
        rep=family.representations[name]
        values=transform(cached_reference['grids'][rep['grid']][2].to(family.device),rep['transforms'][2]).cpu()
        labels=cached_reference['labels']
        count=rep['banks']['shared'][2].shape[1]
        selected=[]
        for c in range(rep['banks']['shared'][2].shape[0]):
            candidates=torch.where(labels.eq(c))[0]
            local=choose_medoids(values[candidates],count,seed+c)
            selected.append(candidates[local])
        selected=torch.stack(selected)
        bank=list(rep['banks']['shared'])
        bank[2]=values[selected].to(family.device)
        rep['banks']['reselected']=bank
        family.reference_ids[name]['reselected']=cached_reference['ids'][selected]
    family._grams={}
