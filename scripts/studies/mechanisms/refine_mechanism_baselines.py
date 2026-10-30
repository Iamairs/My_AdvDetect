"""复用同一PGD缓存，对较病态的独立/单层QP作高精度复核，不重新攻击。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.studies.mechanisms.run_mechanisms import context, load_checkpoint, make_detectors, paired_bootstrap
from research.r002_shared_reference.qp import solve_simplex_qp
from research.mechanisms.common import union_metrics
from advdetect.datasets.cifar import make_loader
from advdetect.evaluation.metrics import calibrate_threshold, detection_metrics
from advdetect.utils.artifacts import create_run, finish_run, save_json
from advdetect.utils.config import load_config, resolve_path


@torch.no_grad()
def score(detector, x, steps):
    logits, z = detector.features(x)
    classes = logits.argmax(1)
    terms = [detector.objective(z, classes, layers=[j], return_terms=True) for j in range(len(z))]
    a, b, c = [torch.cat([t[i] for t in terms]) for i in range(3)]
    _, values, fw = solve_simplex_qp(a, b, c, steps)
    values = values.split(len(x)); fw = fw.split(len(x))
    rho = detector.cfg.get('layer_weights', [1/len(z)]*len(z))
    independent = sum(r*v for r,v in zip(rho,values))
    return logits.argmax(1).cpu().numpy(), {'independent': independent.cpu().numpy(),
                                          'last_layer': values[-1].cpu().numpy()}, [
           (g/(1+v.abs())).cpu().numpy() for g,v in zip(fw,values)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--steps', type=int, default=5000)
    args = parser.parse_args()
    source = resolve_path(args.run_dir)
    config = load_config(source/'config.yaml')
    config['baseline_refinement_steps'] = args.steps
    directory = create_run(config, 'baseline-qp-refinement')
    try:
        device, data, model, binding = context(config, directory)
        detector = make_detectors(model, config, device)['shared']
        checkpoint = json.loads((source/'result.json').read_text(encoding='utf-8'))['checkpoints']['shared']
        load_checkpoint('shared', detector, checkpoint, config, binding)
        cache = torch.load(source/'common_pgd.pt', map_location='cpu', weights_only=True)
        if cache['ids'].tolist() != data['dev_evaluate'].indices:
            raise ValueError('Ordinary PGD cache identities differ')
        scores, convergence = {}, {}
        for split in ['dev_threshold','dev_evaluate','adversarial']:
            result = {'independent': [], 'last_layer': []}; predictions = []; labels=[]; ids=[]
            gaps = [[] for _ in detector.stages]
            offset=0
            for images,y,index in make_loader(data['dev_threshold' if split=='dev_threshold' else 'dev_evaluate'],config):
                x = cache['images'][offset:offset+len(y)] if split=='adversarial' else images
                cp, values, fw = score(detector,x.to(device),args.steps)
                for k,v in values.items():
                    result[k].append(v)
                for target,v in zip(gaps,fw):
                    target.append(v)
                predictions.append(cp); labels.append(y.numpy()); ids.append(index.numpy())
                offset+=len(y)
                if offset % 512 == 0:
                    print(f'High-precision baseline {split} {offset}',flush=True)
            scores[split] = {k: np.concatenate(v) for k,v in result.items()}
            scores[split].update(predictions=np.concatenate(predictions),labels=np.concatenate(labels),ids=np.concatenate(ids))
            convergence[split]={f'layer{s}': {'relative_fw_gap_max':float(np.concatenate(v).max()),
                  'not_converged_fraction':float((np.concatenate(v)>1e-5).mean())}
                  for s,v in zip(detector.stages,gaps)}
        clean,adv=scores['dev_evaluate'],scores['adversarial']
        thresholds={k:calibrate_threshold(scores['dev_threshold'][k],config['evaluation']['target_fpr'])
                    for k in ['independent','last_layer']}
        metrics={};bootstrap={};payload={}
        old=np.load(source/'scores.npz')
        old_tau=json.loads((source/'development_thresholds_shared.json').read_text(encoding='utf-8'))['shared']['threshold']
        correct=clean['predictions']==clean['labels']
        main_joint=(adv['predictions']!=clean['labels']) & (old['shared_adv_shared']<=old_tau)
        for mode in thresholds:
            tau=thresholds[mode]['threshold']
            if tau is None:
                raise ValueError('Insufficient normal development threshold data')
            hit=(adv['predictions']!=clean['labels']) & (adv[mode]<=tau)
            metrics[mode]=detection_metrics(clean[mode],adv[mode],clean['predictions'],adv['predictions'],clean['labels'],thresholds[mode])
            metrics[mode].update(union_metrics(clean[mode],clean['labels'],clean['predictions'],hit,tau))
            bootstrap[mode]=paired_bootstrap(hit[correct],main_joint[correct])
            payload[f'clean_{mode}']=clean[mode];payload[f'adv_{mode}']=adv[mode]
        result={'audit_run':str(directory),'source_run':str(source),'solver_steps':args.steps,
                'ordinary_attacks_reused_without_regeneration':True,'metrics':metrics,'thresholds':thresholds,
                'convergence':convergence,'bootstrap_vs_main':bootstrap}
        # Recheck the retained common-pool baseline witnesses at the refined threshold.
        witnesses=torch.load(source/'adaptive_probe_witnesses.pt',map_location='cpu',weights_only=True)
        witness_results={}
        for mode in thresholds:
            x=witnesses['witnesses']['shared'][mode].to(device)
            prediction,values,_=score(detector,x,args.steps)
            original_ids=witnesses['ids'].tolist()
            original=torch.stack([data['dev_evaluate'].source[i][0] for i in original_ids])
            budget=(x.cpu()-original).flatten(1).abs().amax(1).numpy()<=config['adaptive_probe']['epsilon']+1e-6
            pixel_valid=(x.cpu().flatten(1).amin(1).numpy()>=0)&(x.cpu().flatten(1).amax(1).numpy()<=1)
            hits=(prediction!=witnesses['labels'].numpy())&(values[mode]<=thresholds[mode]['threshold'])&budget&pixel_valid
            witness_results[mode]={'reverified_count':int(hits.sum()),'num_samples':len(hits)}
        result['refined_baseline_witness_verification']=witness_results
        save_json(directory/'baseline_refinement.json',result)
        save_json(source/'baseline_refinement.json',result)
        np.savez_compressed(source/'baseline_refinement_scores.npz',**payload,ids=clean['ids'])
        finish_run(directory,'complete')
        print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    except BaseException as exc:
        finish_run(directory,'failed',error=str(exc));raise


if __name__=='__main__':
    main()
