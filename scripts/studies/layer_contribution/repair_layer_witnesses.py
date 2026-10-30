"""仅修复重载审计失败的边界见证；不改模型、分数定义或防御阈值。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.studies.mechanisms.run_mechanisms import context
from advdetect.utils.config import load_config,resolve_path
from advdetect.utils.artifacts import create_run,finish_run,save_json
from research.r005_layer_contribution.family import LayerFamily
from research.r005_layer_contribution.attack import attack_batch


class InteriorAttackView:
    """攻击者优化更严格的内部余量，部署仍使用原始分数与阈值。"""
    def __init__(self,family,margin):
        self.family=family;self.model=family.model;self.margin=margin
    def evaluate(self,images,names,classes=None):
        logits,scores,gaps=self.family.evaluate(images,names,classes)
        return logits,{name:value+self.margin for name,value in scores.items()},gaps


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--margin-std',type=float,default=.05)
    args=parser.parse_args();source=resolve_path(args.run_dir)
    config=load_config(source/'config.yaml');directory=create_run(config,'layer-witness-repair')
    try:
        device,data,model,binding=context(config,directory)
        saved=torch.load(source/'family_state.pt',map_location='cpu',weights_only=True)
        if saved['binding']!=binding:raise ValueError('Binding mismatch')
        family=LayerFamily.restore(model,saved['state'],device)
        raw=torch.load(source/'common_pool.pt',map_location='cpu',weights_only=True)
        checked=torch.load(source/'witness_verified_mask.pt',map_location='cpu',weights_only=True)
        if raw['names']!=checked['names'] or not torch.equal(raw['ids'],checked['ids']):
            raise ValueError('Diagnostic mask identities differ')
        verified=checked['verified'].clone();witnesses=raw['witnesses'].clone()
        calibrations=json.loads((source/'development_thresholds.json').read_text(encoding='utf-8'))
        margins={name:max(1e-4,args.margin_std*calibrations[name]['score_scale']) for name in raw['names']}
        required=raw['success']&~verified
        required_indices=torch.where(required.any(1))[0].tolist()
        originals=raw['originals'];labels=raw['labels']
        cfg=config['layer_study'];budget=cfg['joint_attack']['epsilon']
        previous=resolve_path(cfg['source_p1'])
        cache=torch.load(previous/'common_pgd.pt',map_location='cpu',weights_only=True)
        positions=json.loads((source/'joint_sample_manifest.json').read_text(encoding='utf-8'))['positions_in_development']
        pgd=cache['images'][positions]
        old_candidates={index:[originals[index],pgd[index]] for index in required_indices}
        # Recover an already generated, clearly interior candidate before producing any new one.
        for path in sorted(source.glob('candidates_*.pt')):
            batch=torch.load(path,map_location='cpu',weights_only=True)
            for row,index in enumerate(batch['indices'].tolist()):
                if index in old_candidates:old_candidates[index].append(batch['images'][row])
        events=[];new_archives=[]

        @torch.no_grad()
        def consume(x,indices):
            for start in range(0,len(x),cfg['batch_size']):
                images=x[start:start+cfg['batch_size']].to(device)
                idx=np.asarray(indices)[start:start+len(images)]
                logits,values,_=family.evaluate(images)
                wrong=logits.argmax(1).cpu().ne(labels[idx])
                norm=(images.cpu()-originals[idx]).flatten(1).abs().amax(1)
                valid=norm.le(budget+1e-6)&images.cpu().flatten(1).amin(1).ge(0)&images.cpu().flatten(1).amax(1).le(1)
                for col,name in enumerate(raw['names']):
                    hit=wrong&valid&values[name].cpu().le(calibrations[name]['threshold']-margins[name])
                    for row,index in enumerate(idx):
                        if hit[row] and not verified[index,col]:
                            verified[index,col]=True;witnesses[index,col]=images[row].cpu()

        for index,values in old_candidates.items():
            x=torch.stack(values);consume(x,np.full(len(x),index))
        recovered=int((verified&required).sum())
        for col,name in enumerate(raw['names']):
            indices=torch.where(required[:,col]&~verified[:,col])[0].numpy()
            if not len(indices):continue
            print(f'Find interior witnesses {name}: {len(indices)}',flush=True)
            for start in range(0,len(indices),cfg['attack_batch_size']):
                chosen=indices[start:start+cfg['attack_batch_size']]
                attack_cfg={**cfg['joint_attack'],'seed':cfg['joint_attack']['seed']+900000+col*1009}
                view=InteriorAttackView(family,margins[name])
                result=attack_batch(view,name,originals[chosen].to(device),labels[chosen].to(device),chosen,
                                    pgd[chosen].to(device),calibrations[name],attack_cfg)
                path=directory/f'candidates_repair_{col}_{start}.pt'
                torch.save({'images':result['images'],'indices':torch.tensor(result['indices']),
                            'generator':name,'margin':margins[name],'events':result['events']},path)
                new_archives.append(str(path));consume(result['images'],result['indices'])
                events.append({'variant':name,'indices':chosen.tolist(),'margin':margins[name],
                               'gradient_steps':result['gradient_steps']})
        if (required&~verified).any():
            raise AssertionError('Unable to recover every invalid recorded witness; do not claim complete success')
        repaired={**raw,'success':verified,'witnesses':witnesses,
                  'repair_directory':str(directory),'original_pool_preserved':True}
        torch.save(repaired,source/'common_pool_repaired.pt')
        summary={'repair_run':str(directory),'source_run':str(source),'original_invalid_pairs':int(required.sum()),
                 'original_invalid_original_ids':raw['ids'][required.any(1)].tolist(),
                 'recovered_from_existing_candidates_with_margin':recovered,
                 'events':events,'new_candidate_archives':new_archives,
                 'defense_thresholds_unchanged':True,'score_definition_unchanged':True,
                 'optimization_margin_std_fraction':args.margin_std,
                 'verified_counts':{name:int(verified[:,i].sum()) for i,name in enumerate(raw['names'])}}
        save_json(directory/'repair.json',summary);save_json(source/'witness_repair.json',summary)
        finish_run(directory,'complete');print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        finish_run(directory,'failed',error=str(exc));raise


if __name__=='__main__':main()
