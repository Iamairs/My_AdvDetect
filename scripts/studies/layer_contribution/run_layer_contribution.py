"""独立执行层贡献、共享约束与空间信息的开发集消融。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.studies.mechanisms.run_mechanisms import context,load_checkpoint,make_detectors
from advdetect.datasets.cifar import IndexedSubset,make_loader,stratified_partition
from advdetect.evaluation.metrics import calibrate_threshold,detection_metrics
from advdetect.utils.artifacts import create_run,finish_run,save_json,sha256_file,stable_hash
from advdetect.utils.config import load_config,resolve_path
from research.r004_bypass_diagnosis.verification import extract_grids
from research.r005_layer_contribution.family import family_from_saved,reselect_layer4
from research.r005_layer_contribution.pool import CandidatePool
from research.r005_layer_contribution.attack import attack_batch


@torch.no_grad()
def score_cached(family,cached,config):
    scores={name:[] for name in family.variants};gaps={key:[] for key in family.problems}
    size=int(config['layer_study']['batch_size'])
    for start in range(0,len(cached['ids']),size):
        raw={grid:[x[start:start+size].to(family.device) for x in values] for grid,values in cached['grids'].items()}
        values,certificates=family.score_z(family.project(raw),cached['predictions'][start:start+size].to(family.device))
        for name,v in values.items():scores[name].append(v.cpu().numpy())
        for key,v in certificates.items():gaps[key].append(v.cpu().numpy())
        if start%(size*8)==0:print(f'Score cached features {start}/{len(cached["ids"])}',flush=True)
    return {k:np.concatenate(v) for k,v in scores.items()},{k:np.concatenate(v) for k,v in gaps.items()}


def summarize_gaps(gaps,tolerance):
    return {k:{'max_relative_fw_gap':float(v.max()),'not_converged_fraction':float((v>tolerance).mean())}
            for k,v in gaps.items()}


def paired_bootstrap(a,b,repetitions,seed,alpha=.05):
    if not len(a):return None
    delta=np.asarray(a,dtype=float)-np.asarray(b,dtype=float)
    rng=np.random.default_rng(seed)
    values=[]
    for start in range(0,repetitions,250):
        indices=rng.integers(len(delta),size=(min(250,repetitions-start),len(delta)))
        values.extend(delta[indices].mean(1).tolist())
    return {'baseline_minus_method':float(delta.mean()),'interval':np.quantile(values,[alpha/2,1-alpha/2]).tolist(),
            'confidence':1-alpha,'unit':'original_image','repetitions':repetitions}


def comparisons(family,scores,calibrations,cp,ap,labels,config,joint_union=None):
    cfg=config['layer_study'];correct=cp==labels
    hits={name:(ap!=labels)&(scores[name]<=calibrations[name]['threshold']) for name in family.variants}
    if joint_union is not None:
        hits={name:joint_union[:,i] for i,name in enumerate(family.variants)}
    primary=[]
    for baseline,method in cfg['primary_comparisons']:
        item=paired_bootstrap(hits[baseline][correct],hits[method][correct],cfg['bootstrap_repetitions'],
                              cfg['bootstrap_seed'],.05/len(cfg['primary_comparisons']))
        primary.append({'baseline':baseline,'method':method,'result':item,'familywise_alpha':.05})
    extra=[]
    for rep in cfg['representations']:
        pairs=[(f'{rep}_L4_shared',f'{rep}_L24_shared'),(f'{rep}_L4_shared',f'{rep}_L34_shared'),
               (f'{rep}_L34_shared',f'{rep}_L234_shared'),(f'{rep}_L24_shared',f'{rep}_L234_shared')]
        pairs += [(f'{rep}_L{layers}_independent',f'{rep}_L{layers}_shared') for layers in ['24','34','234']]
        for baseline,method in pairs:
            extra.append({'baseline':baseline,'method':method,'result':paired_bootstrap(
                hits[baseline][correct],hits[method][correct],cfg['bootstrap_repetitions'],cfg['bootstrap_seed'])})
    return {'primary_bonferroni':primary,'secondary_exploratory':extra}


@torch.no_grad()
def consume_candidates(family,pool,images,indices,seen,batch_size):
    keep=[]
    for row,index in enumerate(indices):
        digest=hashlib.sha256(images[row].contiguous().numpy().tobytes()).hexdigest()
        if digest not in seen[int(index)]:
            seen[int(index)].add(digest);keep.append(row)
    for start in range(0,len(keep),batch_size):
        rows=keep[start:start+batch_size]
        x=images[rows].to(family.device)
        logits,values,_=family.evaluate(x)
        pool.consume(x,np.asarray(indices)[rows],logits.argmax(1).cpu().numpy(),
                     {k:v.cpu().numpy() for k,v in values.items()})
    return len(keep)


def run(config,directory):
    device,data,model,binding=context(config,directory)
    cfg=config['layer_study']
    source=resolve_path(cfg['source_p1'])
    representation_dir=resolve_path(cfg['source_representations'])
    previous=make_detectors(model,config,device)['shared']
    checkpoint=json.loads((source/'result.json').read_text(encoding='utf-8'))['checkpoints']['shared']
    load_checkpoint('shared',previous,checkpoint,config,binding)
    states=torch.load(representation_dir/'verification_variants.pt',map_location='cpu',weights_only=True)
    family_cfg={**deepcopy(cfg),'eta':config['shared']['eta']}
    family=family_from_saved(model,states,family_cfg,device)
    cached={}
    for split in ['reference','dev_threshold','dev_evaluate']:
        print(f'Extract study features: {split}',flush=True)
        cached[split]=extract_grids(previous,data[split],config,device)
    reselect_layer4(family,cached['reference'],int(config['seed']))
    family_path=directory/'family_state.pt'
    torch.save({'format':'advdetect_layer_contribution_v1','state':family.state(),'binding':binding,
                'implementation_sha256':sha256_file(ROOT/'research/r005_layer_contribution/family.py')},family_path)
    save_json(directory/'variants.json',family.variants)
    save_json(directory/'reference_ids.json',{rep:{key:v.tolist() for key,v in groups.items()} for rep,groups in family.reference_ids.items()})
    cache=torch.load(source/'common_pgd.pt',map_location='cpu',weights_only=True)
    if cache['ids'].tolist()!=data['dev_evaluate'].indices:
        raise ValueError('Existing standard PGD cache identity mismatch')
    # Attacked features use the same frozen classifier and exact same source images for all variants.
    adv_chunks={g:[[] for _ in range(3)] for g in [1,2]};adv_predictions=[]
    with torch.no_grad():
        for start in range(0,len(cache['images']),config['batch_size']):
            images=cache['images'][start:start+config['batch_size']].to(device)
            logits,maps=family.view.forward(images);adv_predictions.append(logits.argmax(1).cpu())
            for grid,groups in adv_chunks.items():
                for group,s in zip(groups,[2,3,4]):
                    group.append(torch.nn.functional.adaptive_avg_pool2d(maps[s-1],grid).flatten(1).cpu())
    cached['adversarial']={'grids':{g:[torch.cat(v) for v in groups] for g,groups in adv_chunks.items()},
                          'predictions':torch.cat(adv_predictions),'ids':cache['ids']}
    all_scores={};convergence={}
    for split in ['dev_threshold','dev_evaluate','adversarial']:
        print(f'Evaluate all variants: {split}',flush=True)
        scores,gaps=score_cached(family,cached[split],config)
        all_scores[split]=scores;convergence[split]=summarize_gaps(gaps,cfg['relative_gap_tolerance'])
        save_json(directory/'convergence.json',convergence)
    calibration={name:calibrate_threshold(v,config['evaluation']['target_fpr']) for name,v in all_scores['dev_threshold'].items()}
    if any(v['threshold'] is None for v in calibration.values()):
        raise RuntimeError('Insufficient development threshold data')
    save_json(directory/'development_thresholds.json',calibration)
    cp=cached['dev_evaluate']['predictions'].numpy();ap=cached['adversarial']['predictions'].numpy()
    labels=cached['dev_evaluate']['labels'].numpy()
    ordinary={name:detection_metrics(all_scores['dev_evaluate'][name],all_scores['adversarial'][name],cp,ap,labels,calibration[name])
              for name in family.variants}
    save_json(directory/'ordinary_metrics.json',ordinary)
    contrasts=comparisons(family,all_scores['adversarial'],calibration,cp,ap,labels,config)
    save_json(directory/'ordinary_comparisons.json',contrasts)
    np.savez_compressed(directory/'ordinary_scores.npz',**{f'{split}__{name}':v for split,values in all_scores.items() for name,v in values.items()},
                        labels=labels,clean_predictions=cp,adv_predictions=ap,ids=cached['dev_evaluate']['ids'].numpy())
    # Fresh stratified originals: exclude all 16 cases used in the earlier diagnosis.
    old=torch.load(source/'adaptive_probe_witnesses.pt',map_location='cpu',weights_only=True)['ids'].tolist()
    eligible=np.flatnonzero(~np.isin(cached['dev_evaluate']['ids'].numpy(),old))
    chosen,_=stratified_partition(labels[eligible],int(cfg['joint_samples']),int(cfg['sample_seed']))
    positions=eligible[chosen]
    ids=cached['dev_evaluate']['ids'][positions]
    originals=torch.stack([data['dev_evaluate'].source[int(i)][0] for i in ids])
    y=torch.tensor(labels[positions]);pgd=cache['images'][positions]
    clean_correct=cp[positions]==labels[positions]
    save_json(directory/'joint_sample_manifest.json',{'ids':ids.tolist(),'positions_in_development':positions.tolist(),
        'excluded_previous_case_ids':old,'seed':cfg['sample_seed'],'num_samples':len(ids),
        'class_counts':np.bincount(y.numpy(),minlength=config['model']['num_classes']).tolist(),
        'num_clean_correct':int(clean_correct.sum()),'sample_selected_before_joint_outcomes':True})
    pool=CandidatePool(originals,y,list(family.variants),calibration,cfg['joint_attack']['epsilon'])
    seen=[set() for _ in range(len(ids))]
    consume_candidates(family,pool,originals,np.arange(len(ids)),seen,cfg['batch_size'])
    consume_candidates(family,pool,pgd,np.arange(len(ids)),seen,cfg['batch_size'])
    order=list(dict.fromkeys(cfg['attack_priority']+list(family.variants)))
    events=[]
    for generator_index,name in enumerate(order):
        unresolved=pool.unresolved(name,clean_correct)
        if not len(unresolved):
            events.append({'generator':name,'status':'skipped_already_all_clean_correct_broken_in_common_pool'})
            print(f'Joint {name}: all already broken in common pool',flush=True)
            continue
        print(f'Joint attack {name}: {len(unresolved)} unresolved originals',flush=True)
        paths=[];step_count=0
        for start in range(0,len(unresolved),cfg['attack_batch_size']):
            indices=unresolved[start:start+cfg['attack_batch_size']]
            attack_cfg={**cfg['joint_attack'],'seed':cfg['joint_attack']['seed']+generator_index*100003}
            result=attack_batch(family,name,originals[indices].to(device),y[indices].to(device),indices,
                                pgd[indices].to(device),calibration[name],attack_cfg)
            path=directory/f'candidates_{generator_index:02d}_{start:03d}.pt'
            torch.save({'generator':name,'configuration':attack_cfg,'images':result['images'],
                        'indices':torch.tensor(result['indices']),'events':result['events'],
                        'sample_ids':ids},path)
            paths.append(path.name);step_count+=result['gradient_steps']
            count=consume_candidates(family,pool,result['images'],result['indices'],seen,cfg['batch_size'])
            print(f'  Pool updated: generator={name}, batch={start}, new_candidates={count}',flush=True)
        events.append({'generator':name,'status':'executed','initial_unresolved':len(unresolved),
                       'gradient_steps':step_count,'archives':paths,
                       'own_common_pool_success_after':int(pool.success[:,pool.names.index(name)][clean_correct].sum())})
        save_json(directory/'attack_events.json',events)
        torch.save({'success':torch.tensor(pool.success),'witnesses':pool.witness,'names':pool.names,'ids':ids,
                    'labels':y,'originals':originals,'candidate_counts':torch.tensor(pool.candidate_counts)},directory/'common_pool.pt')
    # Every generated point was cross-scored; every variant sees exactly the same original-wise pool.
    torch.save({'success':torch.tensor(pool.success),'witnesses':pool.witness,'names':pool.names,'ids':ids,
                'labels':y,'originals':originals,'candidate_counts':torch.tensor(pool.candidate_counts)},directory/'common_pool.pt')
    save_json(directory/'attack_events.json',events)
    joint={name:{'num_samples':len(ids),'num_clean_correct':int(clean_correct.sum()),
                 'joint_success_count':int(pool.success[:,i][clean_correct].sum()),
                 'jasr0_common_pool':float(pool.success[:,i][clean_correct].mean()) if clean_correct.any() else None,
                 'wrong_acceptance_union_all_including_original':float(pool.success[:,i].mean()),
                 'normal_fpr_same_development_subset':float((all_scores['dev_evaluate'][name][positions]>calibration[name]['threshold']).mean())}
           for i,name in enumerate(pool.names)}
    save_json(directory/'joint_metrics.json',joint)
    joint_contrasts=comparisons(family,{k:v[positions] for k,v in all_scores['adversarial'].items()},calibration,
                               cp[positions],ap[positions],labels[positions],config,pool.success)
    save_json(directory/'joint_comparisons.json',joint_contrasts)
    return {'scope':'exploratory_development_layer_ablation','variant_count':len(family.variants),'unique_qp_count':len(family.problems),
            'ordinary_samples':len(labels),'joint_samples':len(ids),'old16_excluded':True,
            'classifier_training_independence_verified':False,'final_calibration_used':False,'official_test_used':False,
            'joint_success_counts':{k:v['joint_success_count'] for k,v in joint.items()}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/experiments/studies/layer_contribution/cifar10_layer_contribution.yaml')
    parser.add_argument('--set',dest='overrides',action='append',default=[])
    args=parser.parse_args();config=load_config(args.config,args.overrides)
    directory=create_run(config,'layer-contribution')
    print(f'RUN_DIRECTORY={directory}',flush=True)
    (directory/'study_entrypoint_snapshot.py').write_bytes(Path(__file__).read_bytes())
    try:
        result=run(config,directory);save_json(directory/'result.json',result);finish_run(directory,'complete')
        print(json.dumps({'directory':str(directory),'status':'complete','result':result},ensure_ascii=False),flush=True)
    except BaseException as exc:
        finish_run(directory,'failed',error=str(exc));raise


if __name__=='__main__':
    main()
