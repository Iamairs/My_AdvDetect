"""独立重建保存权重的残差、位移、目标类百分位及主版本真实特征。"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.studies.mechanisms.run_mechanisms import context
from advdetect.utils.config import load_config, resolve_path
from advdetect.utils.artifacts import create_run, finish_run, save_json, sha256_file
from research.r005_layer_contribution.family import LayerFamily


def close(a, b, key, tolerance=1e-5):
    error = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    if not np.isfinite(error) or error > tolerance:
        raise AssertionError(f'{key}: error={error}')
    return error


@torch.no_grad()
def run(config, source, directory):
    device, _, model, binding = context(config, directory)
    provenance = json.loads((source/'provenance.json').read_text(encoding='utf-8'))
    parent = Path(provenance['source_run'])
    if binding != provenance['binding']: raise ValueError('Model/data binding changed')
    for name, digest in provenance['hashes'].items():
        if sha256_file(parent/name) != digest: raise ValueError(f'Source file changed: {name}')
    frozen = torch.load(parent/'family_state.pt', map_location='cpu', weights_only=True)
    family = LayerFamily.restore(model, frozen['state'], device)
    archive = torch.load(source/'geometry_tensors.pt', map_location='cpu', weights_only=True)
    pool = torch.load(provenance['source_pool'], map_location='cpu', weights_only=True)
    thresholds = json.loads((parent/'development_thresholds.json').read_text(encoding='utf-8'))
    normal = np.load(source/'normal_details.npz')
    with (source/'per_layer.csv').open(encoding='utf-8-sig', newline='') as h: layer_rows = list(csv.DictReader(h))
    with (source/'per_sample.csv').open(encoding='utf-8-sig', newline='') as h: sample_rows = list(csv.DictReader(h))
    by_layer = {(r['variant'],int(r['local_index']),int(r['layer'])): r for r in layer_rows}
    by_sample = {(r['variant'],int(r['local_index'])): r for r in sample_rows}
    torch.testing.assert_close(archive['witness_ids'], pool['ids'])
    normal_correct = normal['predictions']==normal['labels']
    max_residual_error = 0.; max_score_error = 0.; max_percentile_error = 0.; max_triangle_excess = 0.
    max_gap = 0.; focus_replayed = {}; max_weight_simplex_error = 0.
    for method_index, name in enumerate(pool['names']):
        spec = family.variants[name]; rep = spec['representation']; item = archive['methods'][name]
        classes = item['classes']; targets = classes.tolist(); weights_by_layer = item['weights_by_layer']
        reconstructed = {'adv': torch.zeros(len(classes),dtype=torch.double), 'clean_wrong': torch.zeros(len(classes),dtype=torch.double)}
        if name in config['witness_geometry']['focus_variants']:
            clean_logits, clean_z = family.features(pool['originals'].to(device))
            logits, adv_z = family.features(pool['witnesses'][:,method_index].to(device))
            torch.testing.assert_close(logits.argmax(1).cpu(), classes)
            if not clean_logits.argmax(1).cpu().eq(pool['labels']).all(): raise AssertionError('Clean not correct')
            for j, l in enumerate(item['layers']):
                torch.testing.assert_close(clean_z[rep][l-2].cpu(), item['z_clean'][j])
                torch.testing.assert_close(adv_z[rep][l-2].cpu(), item['z_adv'][j])
            actual = family.score_z(adv_z, logits.argmax(1), [name])[0][name].cpu()
            if not actual.le(thresholds[name]['threshold']).all(): raise AssertionError('Replayed witness rejected')
            focus_replayed[name] = int(len(actual))
        # Algebraic reconstruction is independent of the analysis decompose helper.
        for j, layer in enumerate(item['layers']):
            problem_key = next(k for k, _ in spec['terms'] if layer in family.problems[k]['layers'])
            bank_name = family.problems[problem_key]['bank']
            bank = frozen['state']['representations'][rep]['banks'][bank_name][layer-2].double()
            z0 = item['z_clean'][j].double(); za = item['z_adv'][j].double()
            movement = torch.linalg.vector_norm(za-z0, dim=1)
            for tag, z in [('adv', za), ('clean_wrong', z0)]:
                w = weights_by_layer[str(layer)][tag].double()
                max_weight_simplex_error = max(max_weight_simplex_error,float((w.sum(1)-1).abs().max()))
                if w.min() < -1e-7: raise AssertionError('Negative weight')
                fitted = torch.einsum('nr,nrd->nd', w, bank[classes])
                residual = torch.linalg.vector_norm(z-fitted,dim=1)
                reconstructed[tag] += residual.square()/z.shape[1]/len(item['layers'])
                reconstructed[tag] += float(family.cfg['eta'])*w.square().sum(1)/len(item['layers'])
                for i, c in enumerate(targets):
                    row = by_layer[name,i,layer]
                    column = 'wrong_reconstruction_l2' if tag=='adv' else 'clean_wrong_refit_l2'
                    max_residual_error = max(max_residual_error,close(residual[i],float(row[column]),column))
                    if tag=='adv':
                        close(movement[i],float(row['shift_l2']),'movement')
                        normal_errors = normal[f'{name}__L{layer}__residual_l2'][normal_correct & (normal['labels']==c)]
                        error = float(residual[i])
                        p = 100*(np.mean(normal_errors<error)+.5*np.mean(normal_errors==error))
                        max_percentile_error = max(max_percentile_error,close(p,float(row['wrong_reconstruction_normal_percentile']),'percentile',1e-10))
                        fixed_clean = torch.linalg.vector_norm(z0[i]-fitted[i])
                        max_triangle_excess = max(max_triangle_excess,float((abs(fixed_clean-residual[i])-movement[i]).clamp_min(0)))
                        nn_wrong = torch.linalg.vector_norm(za[i][None]-bank[c],dim=1).min()
                        nn_true = torch.linalg.vector_norm(za[i][None]-bank[int(pool['labels'][i])],dim=1).min()
                        close(nn_wrong,float(row['nn_wrong_class']),'NN wrong')
                        close(nn_true,float(row['nn_true_class']),'NN true')
        # FW certificate computed directly from A,b,w for every shared/independent QP.
        for key,_ in spec['terms']:
            problem = family.problems[key]
            bank = frozen['state']['representations'][rep]['banks'][problem['bank']]
            layers = problem['layers']; indices = [item['layers'].index(l) for l in layers]
            d = [bank[l-2][classes].double() for l in layers]
            a = sum(torch.einsum('nrd,nsd->nrs',v,v)/v.shape[-1] for v in d)/len(layers)
            a += float(family.cfg['eta'])*torch.eye(a.shape[-1],dtype=torch.double)
            for tag in ['adv','clean_wrong']:
                vectors = item['z_adv'] if tag=='adv' else item['z_clean']
                b = sum(torch.einsum('nrd,nd->nr',v,vectors[idx].double())/v.shape[-1] for v,idx in zip(d,indices))/len(layers)
                w = weights_by_layer[str(layers[0])][tag].double()
                grad = 2*(torch.einsum('nrs,ns->nr',a,w)-b)
                gap = (grad*w).sum(1)-grad.min(1).values
                max_gap = max(max_gap,float(gap.max()))
        for tag,column in [('adv','adv_score'),('clean_wrong','clean_wrong_score')]:
            expected = np.array([float(by_sample[name,i][column]) for i in range(len(classes))])
            max_score_error = max(max_score_error,close(reconstructed[tag].numpy(),expected,'explicit score',1e-4))
        print(f'Geometry audit {name}: 64 algebraic reconstructions',flush=True)
    if max_weight_simplex_error>1e-5 or max_triangle_excess>1e-8: raise AssertionError('Simplex/triangle failed')
    result = {'source_run':str(source),'audit_run':str(directory),'source_hashes_verified':True,
        'algebraic_method_image_pairs':len(sample_rows),'layer_rows':len(layer_rows),
        'focus_model_feature_and_score_replayed':focus_replayed,
        'max_residual_l2_error':max_residual_error,'max_explicit_score_error':max_score_error,
        'max_target_class_percentile_error':max_percentile_error,'max_triangle_excess':max_triangle_excess,
        'max_simplex_sum_error':max_weight_simplex_error,'max_absolute_fw_gap_recomputed_float64':max_gap}
    save_json(directory/'audit.json',result);save_json(source/'independent_geometry_audit.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-dir',required=True)
    args=parser.parse_args();source=resolve_path(args.run_dir);config=load_config(source/'config.yaml')
    directory=create_run(config,'witness-geometry-audit')
    try:
        run(config,source,directory);finish_run(directory,'complete')
        print(json.dumps({'status':'complete','directory':str(directory)}))
    except BaseException as exc:
        finish_run(directory,'failed',error=str(exc));raise


if __name__=='__main__':main()
