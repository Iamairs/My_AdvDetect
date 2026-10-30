from __future__ import annotations

import csv
import hashlib

import numpy as np
import torch
import torch.nn.functional as F

from advdetect.datasets.cifar import IndexedSubset, make_loader
from advdetect.utils.artifacts import save_json
from advdetect.utils.config import resolve_path
from research.mechanisms.common import margin
from research.r002_shared_reference.qp import solve_simplex_qp


def image_digest(image):
    return hashlib.sha256(image.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def projection_energy(delta_map, components):
    """GAP投影到空间常数子空间的能量占比；PCA占GAP变化能量的比例。"""
    gap = delta_map.mean((2, 3))
    full = delta_map.flatten(1).square().sum(1)
    gap_energy = gap.square().sum(1)
    projected = gap @ components.to(gap)
    spatial = delta_map.shape[2]*delta_map.shape[3]
    return {'gap_retained_fraction': spatial*gap_energy/full.clamp_min(1e-20),
            'pca32_retained_of_gap_fraction': projected.square().sum(1)/gap_energy.clamp_min(1e-20),
            'map_delta_l2': full.sqrt(), 'gap_delta_l2': gap_energy.sqrt()}


def weight_metrics(weights):
    return {'effective_reference_count': 1/weights.square().sum(1).clamp_min(1e-12),
            'weight_entropy': -(weights*weights.clamp_min(1e-12).log()).sum(1),
            'support_above_1e-3': weights.gt(.001).sum(1)}


def percentile(value, normal):
    normal = np.asarray(normal)
    return float((np.sum(normal < value)+.5*np.sum(normal == value))/len(normal)) if len(normal) else None


def aligned_cohorts(source, data, config):
    payload = torch.load(source/'adaptive_probe_witnesses.pt', map_location='cpu', weights_only=True)
    cache = torch.load(source/'common_pgd.pt', map_location='cpu', weights_only=True)
    ids = payload['ids'].tolist()
    if not set(ids) <= set(data['dev_evaluate'].indices):
        raise ValueError('Witness IDs outside original development split')
    mapping = {v: i for i, v in enumerate(cache['ids'].tolist())}
    source_data = data['dev_evaluate'].source
    original = torch.stack([source_data[i][0] for i in ids])
    labels = torch.tensor([source_data.targets[i] for i in ids])
    torch.testing.assert_close(labels, payload['labels'])
    roles = {'clean': original, 'pgd': cache['images'][[mapping[i] for i in ids]],
             'joint_shared': payload['witnesses']['shared']['shared'],
             'joint_evidence': payload['witnesses']['evidence']['paired']}
    generator = torch.Generator().manual_seed(config['diagnosis']['random_seed'])
    budget = (roles['joint_shared']-original).flatten(1).abs().amax(1)
    for j in range(config['diagnosis']['random_controls']):
        noise = 2*torch.rand(original.shape, generator=generator)-1
        noise = noise/noise.flatten(1).abs().amax(1)[:,None,None,None]*budget[:,None,None,None]
        roles[f'random_{j}'] = (original+noise).clamp(0,1)
    for name,x in roles.items():
        if not torch.isfinite(x).all() or x.min()<0 or x.max()>1 or (x-original).abs().max()>config['attack']['epsilon']+1e-6:
            raise ValueError(f'Illegal images: {name}')
    provenance = {}
    for role in ['joint_shared', 'joint_evidence']:
        provenance[role] = [{'sample_id': sample_id, 'equals_ordinary_pgd': torch.equal(x, roles['pgd'][i]),
                             'equals_original': torch.equal(x, original[i]), 'sha256': image_digest(x),
                             'generator_origin': 'common-pool witness; unresolved until replay matching'}
                            for i,(sample_id,x) in enumerate(zip(ids,roles[role]))]
    return roles, labels, torch.tensor(ids), provenance


@torch.no_grad()
def shared_detail(detector, images, classes=None, steps=5000):
    logits,z = detector.features(images)
    classes = logits.argmax(1) if classes is None else classes
    terms = [detector.objective(z,classes,return_terms=True)]
    terms += [detector.objective(z,classes,layers=[j],return_terms=True) for j in range(len(z))]
    a,b,c = [torch.cat([t[i] for t in terms]) for i in range(3)]
    weights,values,fw = solve_simplex_qp(a,b,c,steps)
    weights,values,fw = [v.split(len(images)) for v in [weights,values,fw]]
    rho = detector.cfg.get('layer_weights',[1/len(z)]*len(z))
    free = sum(r*v for r,v in zip(rho,values[1:]))
    result = {'es': values[0], 'ef': free, 'gap': values[0]-free,
              'shared_weights': weights[0], 'independent_weights': torch.stack(weights[1:],1),
              'relative_fw_gaps': torch.stack([g/(1+v.abs()) for g,v in zip(fw,values)],1)}
    result.update(weight_metrics(weights[0]))
    result['shared_independent_weight_l1'] = torch.stack([(weights[0]-w).abs().sum(1) for w in weights[1:]],1)
    result['independent_layer_cosine'] = torch.stack([F.cosine_similarity(weights[i+1],weights[j+1],dim=1)
                                                    for i in range(len(z)) for j in range(i+1,len(z))],1)
    result['layer_shared_reconstruction_mse'] = torch.stack([
        (v-torch.bmm(detector.banks['shared'][j][classes].transpose(1,2),weights[0].unsqueeze(-1)).squeeze(-1)).square().mean(1)
        for j,v in enumerate(z)],1)
    result['class'] = classes
    result['reference_ids'] = detector.reference_ids['shared'].to(classes.device)[classes]
    result['deployed_es'] = detector.score(images,classes=classes)
    return {k:v.cpu() for k,v in result.items()}


@torch.no_grad()
def evidence_detail(detector, images, classes=None):
    logits,maps = detector.view.forward(images)
    classes = logits.argmax(1) if classes is None else classes
    branch = detector._branches(maps[detector.stages[0]-1],logits,classes,0)
    values = {'u':branch['u'], 'v':branch['v'], 'r':branch['r'],
              'random_drop_effect':branch['v']-branch['r'],
              'energy_error':branch['energy_error'], 'valid_control':branch['valid_control'],
              'keep_margin_class':margin(branch['keep_logits'],classes),
              'keep_prediction':branch['keep_logits'].argmax(1), 'drop_prediction':branch['drop_logits'].argmax(1)}
    for mode in ['paired','no_control','keep','drop']:
        values[f'score_{mode}'] = detector.score(images,classes,mode)
    for name,mode,dimension in [('u','keep',0),('v','drop',0),('r','paired',1)]:
        profile = detector.profiles[mode]
        covariance = torch.linalg.inv(profile['precision'].double()).float()
        mu = profile['mean'][classes.cpu(),dimension].to(images.device)
        std = covariance[classes.cpu(),dimension,dimension].clamp_min(1e-12).sqrt().to(images.device)
        values[f'z_{name}'] = (values[name]-mu)/std
    values['class'] = classes
    return {k:v.cpu() for k,v in values.items()}


@torch.no_grad()
def analyze(detectors, data, config, directory):
    source = resolve_path(config['diagnosis']['source_run'])
    shared,evidence = detectors['shared'],detectors['evidence']
    device = next(shared.model.parameters()).device
    roles,labels,ids,provenance = aligned_cohorts(source,data,config)
    tau_s = __import__('json').loads((source/'development_thresholds_shared.json').read_text(encoding='utf-8'))
    tau_e = __import__('json').loads((source/'development_thresholds_evidence.json').read_text(encoding='utf-8'))
    steps = config['diagnosis']['detailed_qp_steps']
    normal = np.load(source/'scores.npz')
    refined = np.load(source/'baseline_refinement_scores.npz')
    if not np.array_equal(normal['shared_ids'],refined['ids']):
        raise ValueError('Normal scores and refined baseline identities differ')
    mask = ~np.isin(normal['shared_ids'],ids.numpy())
    normal_reference = {'ids':normal['shared_ids'][mask], 'classes':normal['clean_predictions'][mask],
                        'es':normal['shared_clean_shared'][mask], 'ef':refined['clean_independent'][mask]}
    normal_reference['gap'] = normal_reference['es']-normal_reference['ef']
    details,rows = {},[]
    _,original_maps = shared.view.forward(roles['clean'].to(device))
    for name,images_cpu in roles.items():
        images = images_cpu.to(device)
        logits,maps = shared.view.forward(images)
        pred = logits.argmax(1)
        sd = shared_detail(shared,images,steps=steps)
        ed = evidence_detail(evidence,images)
        norm = (images_cpu-roles['clean']).flatten(1)
        energy = [projection_energy(h-h0,t['components'])
                  for h,h0,t in zip([maps[s-1] for s in shared.stages],
                                   [original_maps[s-1] for s in shared.stages],shared.transforms)]
        entry = {'shared':sd,'evidence':ed, 'predictions':pred.cpu(), 'logits':logits.cpu(),
                 'margin_pred':margin(logits,pred).cpu(), 'margin_true':margin(logits,labels.to(device)).cpu(),
                 'linf':norm.abs().amax(1),'l2':norm.norm(dim=1),
                 'compression':[{k:v.cpu() for k,v in e.items()} for e in energy]}
        details[name] = entry
        for i,sample_id in enumerate(ids.tolist()):
            c = int(pred[i]); matched = normal_reference['classes']==c
            row = {'role':name,'sample_id':sample_id,'label':int(labels[i]),'prediction':c,
                   'linf':float(entry['linf'][i]),'l2':float(entry['l2'][i]),
                   'margin_pred':float(entry['margin_pred'][i]),'margin_true':float(entry['margin_true'][i]),
                   'es':float(sd['es'][i]),'ef':float(sd['ef'][i]),'gap':float(sd['gap'][i]),
                   'shared_accepted':bool(sd['deployed_es'][i]<=tau_s['shared']['threshold']),
                   'effective_references':float(sd['effective_reference_count'][i]),
                   'es_normal_class_percentile':percentile(float(sd['es'][i]),normal_reference['es'][matched]),
                   'gap_normal_class_percentile':percentile(float(sd['gap'][i]),normal_reference['gap'][matched])}
            for key in ['u','v','r','random_drop_effect','z_u','z_v','z_r','score_paired','score_no_control']:
                row[key] = float(ed[key][i])
            row['evidence_accepted'] = bool(ed['score_paired'][i]<=tau_e['paired']['threshold'])
            row['no_control_accepted'] = bool(ed['score_no_control'][i]<=tau_e['no_control']['threshold'])
            for j,e in enumerate(energy):
                for key in ['gap_retained_fraction','pca32_retained_of_gap_fraction']:
                    row[f'layer{shared.stages[j]}_{key}'] = float(e[key][i])
            rows.append(row)
        print(f'Diagnosed {name}: {len(ids)} images',flush=True)
    # Keep class fixed while comparing clean to attack; never compare reference weights across different identities.
    target_s = details['joint_shared']['predictions'].to(device)
    target_e = details['joint_evidence']['predictions'].to(device)
    details['counterfactual_clean_wrong_class_shared'] = shared_detail(shared,roles['clean'].to(device),target_s,steps)
    details['counterfactual_joint_shared_true_class'] = shared_detail(shared,roles['joint_shared'].to(device),labels.to(device),steps)
    details['counterfactual_clean_wrong_class_evidence'] = evidence_detail(evidence,roles['clean'].to(device),target_e)
    normal_ids = normal_reference['ids'].tolist()
    normal_uvr=[];normal_labels=[];normal_pred=[]
    dataset = IndexedSubset(data['dev_evaluate'].source,normal_ids,'diagnostic_normal_excluding_16')
    for x,y,_ in make_loader(dataset,config):
        ed=evidence_detail(evidence,x.to(device))
        normal_uvr.append(torch.stack([ed['u'],ed['v'],ed['r']],1))
        normal_labels.append(y);normal_pred.append(ed['class'])
    np.savez_compressed(directory/'normal_reference.npz',**normal_reference,
                        uvr=torch.cat(normal_uvr).numpy(),labels=torch.cat(normal_labels).numpy(),
                        response_classes=torch.cat(normal_pred).numpy())
    csv_path=directory/'per_sample.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    torch.save({'roles':roles,'labels':labels,'ids':ids,'details':details},directory/'diagnosis.pt')
    save_json(directory/'witness_provenance.json',provenance)
    summary={}
    for name in roles:
        subset=[r for r in rows if r['role']==name]
        summary[name]={'num_samples':len(subset),'misclassified':sum(r['prediction']!=r['label'] for r in subset),
                       'shared_wrong_accepted':sum(r['prediction']!=r['label'] and r['shared_accepted'] for r in subset),
                       'evidence_wrong_accepted':sum(r['prediction']!=r['label'] and r['evidence_accepted'] for r in subset),
                       'no_control_wrong_accepted':sum(r['prediction']!=r['label'] and r['no_control_accepted'] for r in subset),
                       'mean_es':float(np.mean([r['es'] for r in subset])),
                       'mean_gap':float(np.mean([r['gap'] for r in subset])),
                       'mean_u':float(np.mean([r['u'] for r in subset])),
                       'mean_v':float(np.mean([r['v'] for r in subset])),
                       'mean_r':float(np.mean([r['r'] for r in subset])),
                       'mean_effective_references':float(np.mean([r['effective_references'] for r in subset])),
                       'max_relative_fw_gap':details[name]['shared']['relative_fw_gaps'].max().item()}
    save_json(directory/'analysis_summary.json',{'source_run':str(source),'scope':'16 fixed exploratory bypass cases',
              'normal_reference_excludes_case_ids':True,'normal_reference_size':len(normal_ids),
              'classifier_training_independence_verified':False,'roles':summary})
    return {'analysis_directory':str(directory),'summary':summary}
