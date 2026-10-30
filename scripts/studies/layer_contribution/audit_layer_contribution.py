"""独立加载层消融状态，核验数据来源、公共池见证与真实输入梯度。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.studies.mechanisms.run_mechanisms import context,make_detectors,load_checkpoint,gradient_probe
from advdetect.datasets.cifar import make_loader
from advdetect.utils.config import load_config,resolve_path
from advdetect.utils.artifacts import create_run,finish_run,save_json,sha256_file,stable_hash
from research.r005_layer_contribution.family import LayerFamily


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--diagnostic-only',action='store_true')
    args=parser.parse_args();source=resolve_path(args.run_dir)
    config=load_config(source/'config.yaml')
    directory=create_run(config,'layer-contribution-audit')
    try:
        device,data,model,binding=context(config,directory)
        saved=torch.load(source/'family_state.pt',map_location='cpu',weights_only=True)
        if saved['binding']!=binding or saved['implementation_sha256']!=sha256_file(ROOT/'research/r005_layer_contribution/family.py'):
            raise ValueError('Layer study model/data/source binding mismatch')
        rep_dir=resolve_path(config['layer_study']['source_representations'])
        rep_classifier=json.loads((rep_dir/'classifier.json').read_text(encoding='utf-8'))
        rep_splits=json.loads((rep_dir/'splits.json').read_text(encoding='utf-8'))
        if rep_classifier['sha256']!=binding['classifier_sha256'] or stable_hash(rep_splits)!=binding['splits_sha256']:
            raise ValueError('Borrowed PCA/reference state belongs to different model/data')
        family=LayerFamily.restore(model,saved['state'],device)
        previous=make_detectors(model,config,device)['shared']
        p1=resolve_path(config['layer_study']['source_p1'])
        checkpoint=json.loads((p1/'result.json').read_text(encoding='utf-8'))['checkpoints']['shared']
        load_checkpoint('shared',previous,checkpoint,config,binding)
        for name,groups in family.reference_ids.items():
            torch.testing.assert_close(groups['shared'],previous.reference_ids['shared'])
            if 'reselected' in groups:
                references=groups['reselected']
                if not set(references.flatten().tolist())<=set(data['reference'].indices):
                    raise ValueError('Reselected references outside D_ref')
                labels=torch.tensor([data['reference'].source.targets[int(i)] for i in references.flatten()]).reshape_as(references)
                torch.testing.assert_close(labels,torch.arange(references.shape[0])[:,None].expand_as(references))
        thresholds=json.loads((source/'development_thresholds.json').read_text(encoding='utf-8'))
        pool_path=source/'common_pool.pt'
        if not args.diagnostic_only and (source/'common_pool_repaired.pt').is_file():
            pool_path=source/'common_pool_repaired.pt'
        pool=torch.load(pool_path,map_location='cpu',weights_only=True)
        manifest=json.loads((source/'joint_sample_manifest.json').read_text(encoding='utf-8'))
        if pool['ids'].tolist()!=manifest['ids'] or set(manifest['ids'])&set(manifest['excluded_previous_case_ids']):
            raise ValueError('Fresh joint sample identity mismatch')
        if pool['names']!=list(family.variants):
            raise ValueError('Candidate pool variant order mismatch')
        original=torch.stack([data['dev_evaluate'].source[int(i)][0] for i in pool['ids']])
        torch.testing.assert_close(original,pool['originals'])
        y=pool['labels'].to(device)
        audited={}
        verified_columns=[]
        with torch.no_grad():
            for i,name in enumerate(pool['names']):
                x=pool['witnesses'][:,i].to(device)
                recorded=pool['success'][:,i].to(device)
                logits,values,gaps=family.evaluate(x,[name])
                value=values[name]
                norm=(x-original.to(device)).flatten(1).abs().amax(1)
                valid=norm.le(config['layer_study']['joint_attack']['epsilon']+1e-6)
                valid&=x.flatten(1).amin(1).ge(0)&x.flatten(1).amax(1).le(1)
                real=logits.argmax(1).ne(y)&value.le(thresholds[name]['threshold'])&valid&torch.isfinite(value)
                if (recorded&~real).any() and not args.diagnostic_only:
                    raise AssertionError(f'Invalid retained common-pool witness: {name}')
                verified_columns.append((recorded&real).cpu())
                audited[name]={'recorded_success':int(recorded.sum()),'verified_success':int((recorded&real).sum()),
                               'max_linf':norm.max().item(),'max_relative_fw_gap':max(v.max().item() for v in gaps.values()),
                               'failed_indices':torch.where(recorded&~real)[0].cpu().tolist(),
                               'failed_ids':pool['ids'][(recorded&~real).cpu()].tolist(),
                               'max_failed_score_excess':(value[recorded&~real]-thresholds[name]['threshold']).max().item()
                                   if (recorded&~real).any() else None}
                print(f'Audit {name}: {audited[name]["verified_success"]}/{len(y)}',flush=True)
        # Numerical gradients cover the new single/shared/independent/spatial paths.
        probe_names=['gap1_pca32_L4_shared','gap1_pca32_L234_shared',
                     'gap1_pca32_L234_independent','grid2_pca32_L234_shared']
        x=original[:1].to(device)
        target=(model(x).argmax(1)+1)%config['model']['num_classes']
        gradients={}
        for name in ([] if args.diagnostic_only else probe_names):
            gradients[name]=gradient_probe(family,x,target,name)
        # Recheck the complete reused ordinary PGD cache's raw pixel/budget constraints.
        cache=torch.load(p1/'common_pgd.pt',map_location='cpu',weights_only=True)
        max_linf=0.;offset=0
        for clean,_,ids in make_loader(data['dev_evaluate'],config):
            adv=cache['images'][offset:offset+len(clean)]
            torch.testing.assert_close(ids,cache['ids'][offset:offset+len(clean)])
            max_linf=max(max_linf,(adv-clean).abs().max().item())
            if not torch.isfinite(adv).all() or adv.min()<0 or adv.max()>1:
                raise ValueError('Reused PGD cache has illegal pixels')
            offset+=len(clean)
        if max_linf>config['attack']['epsilon']+1e-6:
            raise ValueError('Reused PGD cache exceeds original budget')
        result={'audit_run':str(directory),'source_run':str(source),'source_state_model_data_verified':True,
                'source_pool':str(pool_path),
                'reference_identities_verified':True,'fresh64_excludes_old16':True,
                'ordinary_cache_max_linf':max_linf,'witnesses':audited,'gradient_probes':gradients}
        if args.diagnostic_only:
            save_json(directory/'diagnostic.json',result)
            save_json(source/'witness_diagnostic.json',result)
            torch.save({'verified':torch.stack(verified_columns,1),'names':pool['names'],'ids':pool['ids']},source/'witness_verified_mask.pt')
        else:
            save_json(directory/'audit.json',result);save_json(source/'independent_audit.json',result)
        finish_run(directory,'complete')
        print(json.dumps({'status':'complete','directory':str(directory)},ensure_ascii=False))
    except BaseException as exc:
        finish_run(directory,'failed',error=str(exc));raise


if __name__=='__main__':main()
