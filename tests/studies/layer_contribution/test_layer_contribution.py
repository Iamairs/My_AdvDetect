from pathlib import Path
import sys
import unittest

import numpy as np
import torch
from torch import nn

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from research.r005_layer_contribution.family import LayerFamily
from research.r005_layer_contribution.pool import CandidatePool
from research.r005_layer_contribution.attack import attack_batch
from research.r002_shared_reference.detector import SharedReferenceDetector


class LayerContributionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def toy_family(self):
        torch.manual_seed(18)
        model=nn.Module()
        for s in range(1,5):setattr(model,f'layer{s}',nn.Identity())
        bank=[torch.randn(2,4,d,dtype=torch.double) for d in [3,4,5]]
        transforms=[{'mean':torch.zeros(d,dtype=torch.double),'std':torch.ones(d,dtype=torch.double),
                     'components':torch.eye(d,dtype=torch.double)} for d in [3,4,5]]
        family=LayerFamily(model,{'r':{'grid':1,'transforms':transforms,'banks':{'shared':bank}}},{},
                           {'representations':['r'],'layer_sets':[[4],[2,3,4]],'solver_steps':300,'eta':.1},torch.device('cpu'))
        return family,bank

    def test_batched_family_matches_existing_shared_independent_and_single(self):
        family,bank=self.toy_family()
        z=[torch.randn(2,d,dtype=torch.double) for d in [3,4,5]]
        classes=torch.tensor([0,1])
        scores,_=family.score_z({'r':z},classes)
        oracle=object.__new__(SharedReferenceDetector)
        oracle.cfg={'eta':.1,'solver_steps':300}
        oracle.banks={'shared':bank};oracle._grams={}
        torch.testing.assert_close(scores['r_L234_shared'],oracle.objective(z,classes)[1])
        independent=sum(oracle.objective(z,classes,layers=[j])[1]/3 for j in range(3))
        torch.testing.assert_close(scores['r_L234_independent'],independent)
        torch.testing.assert_close(scores['r_L4_shared'],oracle.objective(z,classes,layers=[2])[1])
        self.assertTrue((scores['r_L234_shared']>=independent-1e-9).all())

    def test_independent_and_shared_keep_input_gradient(self):
        family,_=self.toy_family()
        z=tuple(torch.randn(2,d,dtype=torch.double,requires_grad=True) for d in [3,4,5])
        classes=torch.tensor([0,1])
        for mode in ['r_L234_shared','r_L234_independent']:
            def f(*values):
                return family.score_z({'r':list(values)},classes,[mode])[0][mode]
            self.assertTrue(torch.autograd.gradcheck(f,z,fast_mode=True,atol=1e-5))

    def test_common_pool_preserves_union_indices_and_original_errors(self):
        images=torch.full((3,1,1,1),.2)
        labels=torch.tensor([0,1,0])
        pool=CandidatePool(images,labels,['a','b'],{'a':{'threshold':.5},'b':{'threshold':.5}},.2)
        pool.consume(images,[0,1,2],[0,0,0],{'a':[0,0,0],'b':[0,0,0]})
        attack=images[[0,2]]+.1
        pool.consume(attack,[0,2],[1,1],{'a':[.1,.9],'b':[.9,.1]})
        np.testing.assert_array_equal(pool.success,[[True,False],[True,True],[False,True]])
        np.testing.assert_array_equal(pool.unresolved('a',[True,False,True]),[2])
        old=pool.witness[0,0].clone()
        pool.consume(images[:1],[0],[1],{'a':[.9],'b':[.9]})
        torch.testing.assert_close(old,pool.witness[0,0])
        with self.assertRaises(ValueError):
            pool.consume(torch.ones_like(images[:1]),[0],[1],{'a':[0],'b':[0]})

    def test_early_stop_only_after_true_accepted_misclassification(self):
        class Model(nn.Module):
            def forward(self,x):
                v=x.flatten(1).mean(1)-.5
                return torch.stack([-v,v],1)
        class Family:
            model=Model()
            def evaluate(self,x,names,classes=None):
                score=(x.flatten(1).mean(1)-.55).square()
                return self.model(x),{names[0]:score},{}
        images=torch.full((2,1,1,1),.45)
        cfg={'epsilon':.2,'step_size':.02,'steps':20,'targets':1,'restarts':1,'lambdas':[1.],
             'kappa':.1,'seed':7}
        result=attack_batch(Family(),'a',images,torch.zeros(2,dtype=torch.long),np.arange(2),images,
                            {'threshold':.04,'score_scale':1.},cfg)
        self.assertTrue(result['own_success'].all())
        self.assertGreater(result['gradient_steps'],0)
        self.assertLess(result['gradient_steps'],20)
        self.assertTrue(Family.model(result['best']).argmax(1).eq(1).all())
        self.assertLessEqual((result['best']-images).abs().max().item(),.200001)


if __name__=='__main__':unittest.main()
