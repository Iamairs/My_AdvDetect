from __future__ import annotations

from copy import deepcopy
import json

import numpy as np
import torch
import torch.nn.functional as F

from advdetect.datasets.cifar import make_loader
from advdetect.evaluation.metrics import calibrate_threshold
from advdetect.utils.artifacts import save_json
from advdetect.utils.config import resolve_path
from research.mechanisms.adaptive import joint_pgd
from research.mechanisms.common import margin
from research.r002_shared_reference.detector import SharedReferenceDetector,fit_transform,transform
from .details import image_digest,shared_detail


@torch.no_grad()
def extract_grids(detector,dataset,config,device):
    chunks={g:[[] for _ in detector.stages] for g in [1,2]}
    labels=[];ids=[];predictions=[]
    for x,y,index in make_loader(dataset,config):
        logits,maps=detector.view.forward(x.to(device))
        for grid,groups in chunks.items():
            for group,stage in zip(groups,detector.stages):
                group.append(F.adaptive_avg_pool2d(maps[stage-1],grid).flatten(1).cpu())
        labels.append(y);ids.append(index);predictions.append(logits.argmax(1).cpu())
    return {'grids':{g:[torch.cat(v) for v in groups] for g,groups in chunks.items()},
            'labels':torch.cat(labels),'ids':torch.cat(ids),'predictions':torch.cat(predictions)}


def transform_subset(full,dimension):
    if dimension=='full':
        return full
    return {'mean':full['mean'],'components':full['components'][:,:int(dimension)],'std':full['std'][:int(dimension)]}


def fixed_identity_variant(baseline,features,full_transforms,spec,device):
    cfg=deepcopy(baseline.cfg)
    cfg.update(spatial_grid=spec['grid'],pca_dim=spec['pca'])
    variant=SharedReferenceDetector(baseline.model,cfg,device)
    variant.modes=('shared',)
    variant.transforms=[transform_subset(t,spec['pca']) for t in full_transforms[spec['grid']]]
    positions={int(v):i for i,v in enumerate(features['ids'])}
    selected=torch.tensor([[positions[int(v)] for v in group] for group in baseline.reference_ids['shared']])
    expected=torch.arange(selected.shape[0])[:,None].expand_as(selected)
    if not torch.equal(features['labels'][selected],expected):
        raise ValueError('Reference identity/class mismatch')
    variant.reference_ids={'shared':baseline.reference_ids['shared'].clone()}
    variant.banks={'shared':[transform(x,t)[selected].to(device)
                              for x,t in zip(features['grids'][spec['grid']],variant.transforms)]}
    variant._grams={}
    return variant


@torch.no_grad()
def cached_scores(detector,cached,grid,device,batch_size=64):
    scores=[];gaps=[]
    for start in range(0,len(cached['ids']),batch_size):
        z=[transform(v[start:start+batch_size].to(device),t)
           for v,t in zip(cached['grids'][grid],detector.transforms)]
        classes=cached['predictions'][start:start+batch_size].to(device)
        _,value,fw=detector.objective(z,classes)
        scores.append(value.cpu());gaps.append((fw/(1+value.abs())).cpu())
    return torch.cat(scores),torch.cat(gaps)


def replay(name,detector,mode,images,labels,calibration,config,directory):
    cfg={**config['adaptive_probe'],'score_scale':calibration['score_scale']}
    tau=calibration['threshold']
    frames=[];tags=[];records=[]
    original_logits=detector.model(images).detach()
    order=original_logits.masked_fill(F.one_hot(labels,original_logits.shape[1]).bool(),-torch.inf).argsort(1,descending=True)
    union=torch.zeros(len(images),dtype=torch.bool,device=images.device)
    @torch.no_grad()
    def observer(candidate,tag):
        logits=detector.model(candidate);pred=logits.argmax(1)
        value=detector.score(candidate,mode=mode)
        budget=(candidate-images).flatten(1).abs().amax(1)
        valid=budget.le(cfg['epsilon']+1e-6)&candidate.flatten(1).amin(1).ge(0)&candidate.flatten(1).amax(1).le(1)
        hit=pred.ne(labels)&value.le(tau)&valid&torch.isfinite(value)
        union.logical_or_(hit)
        frames.append(candidate.detach().cpu());tags.append(dict(tag))
        target=order[:,tag['target_rank']] if 'target_rank' in tag else labels
        records.append({**tag,'predictions':pred.cpu().tolist(),'margin_true':margin(logits,labels).cpu().tolist(),
                        'margin_target':margin(logits,target).cpu().tolist(),'score':value.cpu().tolist(),
                        'success':hit.cpu().tolist(),'union_success':union.cpu().tolist(),'linf':budget.cpu().tolist()})
    result=joint_pgd(detector.model,detector,images,labels,tau,cfg,mode,observer)
    with torch.no_grad():
        logits=detector.model(result['images']);score=detector.score(result['images'],mode=mode)
        real=logits.argmax(1).ne(labels)&score.le(tau)
        real&=(result['images']-images).flatten(1).abs().amax(1).le(cfg['epsilon']+1e-6)
        real&=result['images'].flatten(1).amin(1).ge(0)&result['images'].flatten(1).amax(1).le(1)
        if (result['success']&~real).any():
            raise AssertionError(f'Invalid replay witness: {name}')
    payload={'name':name,'configuration':cfg,'calibration':calibration,'frames':torch.stack(frames),
             'tags':tags,'records':records,'best_images':result['images'].cpu(),'success':result['success'].cpu()}
    torch.save(payload,directory/f'attack_{name}.pt')
    save_json(directory/f'trajectory_{name}.json',{'configuration':cfg,'threshold':tau,'records':records})
    print(f'Joint replay {name}: {int(result["success"].sum())}/{len(images)}',flush=True)
    return payload


def verify(detectors,data,config,directory,analysis_dir):
    study=torch.load(analysis_dir/'diagnosis.pt',map_location='cpu',weights_only=True)
    baseline=detectors['shared'];evidence=detectors['evidence']
    device=next(baseline.model.parameters()).device
    images=study['roles']['clean'].to(device);labels=study['labels'].to(device)
    source=resolve_path(config['diagnosis']['source_run'])
    old_cal_s=json.loads((source/'development_thresholds_shared.json').read_text(encoding='utf-8'))
    old_cal_e=json.loads((source/'development_thresholds_evidence.json').read_text(encoding='utf-8'))
    provenance=json.loads((analysis_dir/'witness_provenance.json').read_text(encoding='utf-8'))
    replayed={}
    for name,detector,mode,cal in [('shared',baseline,'shared',old_cal_s['shared']),
                                    ('evidence',evidence,'paired',old_cal_e['paired'])]:
        replayed[name]=replay(name,detector,mode,images,labels,cal,config,directory)
    # Exact-byte provenance matching. Never infer which attack generated a witness from its label alone.
    for role,entries in provenance.items():
        for i,entry in enumerate(entries):
            if entry['equals_ordinary_pgd']:
                entry['generator_origin']='ordinary_pgd';continue
            matches=[]
            for name,payload in replayed.items():
                for frame,tag in zip(payload['frames'],payload['tags']):
                    if image_digest(frame[i])==entry['sha256']:
                        matches.append({'generator':name,**tag})
            entry['replay_matches']=matches
            entry['generator_origin']='replay_exact_match' if matches else 'not_exactly_reproduced; origin remains unknown'
    save_json(directory/'resolved_witness_provenance.json',provenance)
    # Diagnose selected trajectory frames at fixed classes with high-precision QPs.
    trajectory_detail={}
    checkpoints=set(config['diagnosis']['trajectory_checkpoints'])
    for name,payload in replayed.items():
        selected=[i for i,t in enumerate(payload['tags']) if t.get('step') in checkpoints or t.get('kind')=='original']
        values=[]
        for index in selected:
            x=payload['frames'][index].to(device)
            d=shared_detail(baseline,x,steps=config['diagnosis']['detailed_qp_steps'])
            values.append({'tag':payload['tags'][index],'shared':d})
        trajectory_detail[name]=values
    torch.save(trajectory_detail,directory/'trajectory_qp_details.pt')
    # Fixed 2x2 representation experiment: same images, same reference identities and same eta.
    cached={}
    for split in ['reference','dev_threshold','dev_evaluate']:
        print(f'Extract fixed representation features: {split}',flush=True)
        cached[split]=extract_grids(baseline,data[split],config,device)
    full_transforms={}
    for grid in [1,2]:
        full_transforms[grid]=[]
        for j,x in enumerate(cached['reference']['grids'][grid]):
            print(f'Fit full PCA: grid={grid}, layer={baseline.stages[j]}, dim={x.shape[1]}',flush=True)
            full_transforms[grid].append(fit_transform(x,x.shape[1]))
    variants={};states={};results={};variant_cal={};score_arrays={}
    case_roles=['clean','pgd','joint_shared','joint_evidence']
    for spec in config['diagnosis']['variants']:
        name=spec['name']
        variant=baseline if name=='gap1_pca32' else fixed_identity_variant(baseline,cached['reference'],full_transforms,spec,device)
        variants[name]=variant
        with torch.no_grad():
            cal_scores,cal_gaps=cached_scores(variant,cached['dev_threshold'],spec['grid'],device)
            normal_scores,normal_gaps=cached_scores(variant,cached['dev_evaluate'],spec['grid'],device)
            cal=old_cal_s['shared'] if name=='gap1_pca32' else calibrate_threshold(cal_scores.numpy(),config['evaluation']['target_fpr'])
            variant_cal[name]=cal
            role_results={}
            for role in case_roles:
                x=study['roles'][role].to(device)
                predictions=baseline.model(x).argmax(1)
                values=variant.score(x)
                hit=predictions.ne(labels)&values.le(cal['threshold'])
                role_results[role]={'wrong_accepted':int(hit.sum()),'rejected':int(values.gt(cal['threshold']).sum()),
                                    'scores':values.cpu().tolist(),'predictions':predictions.cpu().tolist()}
            results[name]={'specification':spec,'same_reference_ids':True,'calibration':cal,
                           'normal_dev_fpr_1000':normal_scores.gt(cal['threshold']).float().mean().item(),
                           'max_relative_fw_gap_threshold':cal_gaps.max().item(),
                           'max_relative_fw_gap_normal':normal_gaps.max().item(),
                           'cases':role_results}
            score_arrays[f'{name}_threshold']=cal_scores.numpy();score_arrays[f'{name}_normal']=normal_scores.numpy()
            states[name]={'config':variant.cfg,'state':variant.state()}
        save_json(directory/'representation_transfer.json',results)
        print(f'Representation transfer {name}: {role_results}',flush=True)
    torch.save(states,directory/'verification_variants.pt')
    np.savez_compressed(directory/'representation_normal_scores.npz',**score_arrays)
    # Re-attack predeclared variants; transfer rejection is not treated as adaptive robustness.
    reattacks={}
    for name in config['diagnosis']['reattack']:
        if name=='evidence_no_control':
            detector,mode,cal=evidence,'no_control',old_cal_e['no_control']
        else:
            detector,mode,cal=variants[name],'shared',variant_cal[name]
        payload=replay(name,detector,mode,images,labels,cal,config,directory)
        reattacks[name]={'success_count':int(payload['success'].sum()),'num_samples':len(images),
                        'threshold':cal['threshold'],'same_attack_budget':True}
    # Cross-score all retained witnesses on every predeclared representation.
    with torch.no_grad():
        cross={}
        candidates={name:torch.load(directory/f'attack_{name}.pt',map_location='cpu',weights_only=True)['best_images']
                    for name in ['shared','evidence',*config['diagnosis']['reattack']]}
        for name,variant in variants.items():
            cross[name]={}
            for generator,x in candidates.items():
                x=x.to(device);prediction=baseline.model(x).argmax(1)
                cross[name][generator]=int((prediction.ne(labels)&variant.score(x).le(variant_cal[name]['threshold'])).sum())
    result={'analysis_directory':str(analysis_dir),'source_run':str(source),
            'scope':'fixed16_case_mechanism_verification_not_population_robustness',
            'replay':{name:int(v['success'].sum()) for name,v in replayed.items()},
            'representation_transfer':results,'reattacks':reattacks,'retained_witness_cross_acceptance':cross}
    save_json(directory/'verification_summary.json',result)
    return result
