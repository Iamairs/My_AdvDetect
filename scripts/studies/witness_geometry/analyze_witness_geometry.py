"""逐层对比已审计见证的特征位移、错误类别重构与正常距离尺度。"""
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
from advdetect.datasets.cifar import make_loader
from advdetect.utils.config import load_config, resolve_path
from advdetect.utils.artifacts import create_run, finish_run, save_json, sha256_file
from research.r005_layer_contribution.family import LayerFamily
from research.r006_witness_geometry.geometry import decompose, describe, percentile, within_class_pair_distances, nearest_reference


def array(t):
    return t.detach().cpu().numpy()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@torch.no_grad()
def normal_data(family, dataset, config):
    values = {rep: [[] for _ in range(3)] for rep in family.representations}
    labels, predictions, ids = [], [], []
    for x, y, index in make_loader(dataset, config):
        logits, z = family.features(x.to(family.device))
        for rep in values:
            for group, item in zip(values[rep], z[rep]): group.append(item.cpu())
        labels.append(y); predictions.append(logits.argmax(1).cpu()); ids.append(index)
    return {k: [torch.cat(v) for v in groups] for k, groups in values.items()}, torch.cat(labels), torch.cat(predictions), torch.cat(ids)


@torch.no_grad()
def run(config, directory):
    cfg = config['witness_geometry']
    source = resolve_path(cfg['source_run'])
    device, data, model, binding = context(config, directory)
    saved = torch.load(source/'family_state.pt', map_location='cpu', weights_only=True)
    if binding != saved['binding'] or saved['implementation_sha256'] != sha256_file(ROOT/'research/r005_layer_contribution/family.py'):
        raise ValueError('Frozen model/data/family implementation binding mismatch')
    family = LayerFamily.restore(model, saved['state'], device)
    audit = json.loads((source/'independent_audit.json').read_text(encoding='utf-8'))
    pool_path = source/cfg['source_pool']
    if resolve_path(audit['source_pool']) != pool_path.resolve():
        raise ValueError('Analyze only the pool used by the strict independent audit')
    pool = torch.load(pool_path, map_location='cpu', weights_only=True)
    thresholds = json.loads((source/'development_thresholds.json').read_text(encoding='utf-8'))
    manifest = json.loads((source/'joint_sample_manifest.json').read_text(encoding='utf-8'))
    if pool['names'] != list(family.variants) or pool['ids'].tolist() != manifest['ids']:
        raise ValueError('Witness identities or variants mismatch')
    if set(pool['ids'].tolist()) & set(manifest['excluded_previous_case_ids']):
        raise ValueError('Old 16 cases leaked into fresh set')
    originals = torch.stack([data['dev_evaluate'].source[int(i)][0] for i in pool['ids']])
    torch.testing.assert_close(originals, pool['originals'])
    clean_logits, clean_z = family.features(originals.to(device))
    y = pool['labels'].to(device)
    if not clean_logits.argmax(1).eq(y).all(): raise ValueError('An original is already misclassified')
    provenance = {'source_run': str(source), 'source_pool': str(pool_path), 'binding': binding,
        'hashes': {name: sha256_file(source/name) for name in ['family_state.pt', cfg['source_pool'],
                     'development_thresholds.json', 'independent_audit.json', 'joint_sample_manifest.json']},
        'analysis_code': {str(p.relative_to(ROOT)): sha256_file(p) for p in
                          [Path(__file__), ROOT/'research/r006_witness_geometry/geometry.py']},
        'normal_scale_split': cfg['normal_split'], 'read_only_source': True,
        'selection_warning': 'Each method retained its lowest-score successful candidate; not a minimal-movement witness.',
        'inference_unit': '64 originals per method; neither 2432 pairs nor per-layer rows are independent samples.'}
    save_json(directory/'provenance.json', provenance)
    print('Extract normal distance scales', flush=True)
    normal_z, normal_y, normal_pred, normal_ids = normal_data(family, data[cfg['normal_split']], config)
    if set(normal_ids.tolist()) & set(pool['ids'].tolist()): raise ValueError('Normal scale and witness originals overlap')
    for groups in family.reference_ids.values():
        if any(set(normal_ids.tolist()) & set(group.flatten().tolist()) for group in groups.values()):
            raise ValueError('Normal scale images overlap dictionary images')
    correct = normal_pred.eq(normal_y).numpy()
    n = len(normal_y); batch_size = cfg['batch_size']
    normal_arrays = {'labels': array(normal_y), 'predictions': array(normal_pred), 'ids': array(normal_ids)}
    normal_gaps = []; decomposition_errors = []
    chunks = {}
    for start in range(0, n, batch_size):
        z = {k: [v[start:start+batch_size].to(device) for v in groups] for k, groups in normal_z.items()}
        result = decompose(family, z, normal_pred[start:start+batch_size].to(device))
        for name, item in result.items():
            for field in ['score', 'regularizer', 'explicit_score']:
                chunks.setdefault(f'{name}__{field}', []).append(array(item[field]))
            for layer, layer_item in item['layers'].items():
                chunks.setdefault(f'{name}__L{layer}__residual_l2', []).append(array(layer_item['residual_l2']))
            normal_gaps.append(float(item['max_relative_fw_gap'].max()))
            decomposition_errors.append(float(item['decomposition_absolute_error'].max()))
        print(f'Normal reconstruction {min(start+batch_size,n)}/{n}', flush=True)
    normal_arrays.update({k: np.concatenate(v) for k, v in chunks.items()})
    np.savez_compressed(directory/'normal_details.npz', **normal_arrays)

    # Distance scales depend on the representation/layer/bank, not the sharing solver.
    distance_cache = {}; distance_stats = {}
    used = set()
    for spec in family.variants.values():
        for key, _ in spec['terms']:
            problem = family.problems[key]
            used.update((spec['representation'], problem['bank'], l) for l in problem['layers'])
    for rep, bank_name, layer in sorted(used):
        values = normal_z[rep][layer-2]
        bank = family.representations[rep]['banks'][bank_name][layer-2].cpu()
        nn = array(nearest_reference(values, normal_pred, bank))
        pair = within_class_pair_distances(values[correct], array(normal_y)[correct], cfg['pairs_per_class'], cfg['seed']+layer)
        key = f'{rep}|{bank_name}|L{layer}'
        distance_cache[key] = {'nn': nn, 'pair': pair}
        distance_stats[key] = {str(c): {'nearest_reference': describe(nn[correct & array(normal_y).astype(int).__eq__(c)]),
                                      'within_class_pair': describe(pair[c])} for c in pair}
    save_json(directory/'normal_distance_scales.json', distance_stats)
    class_score_stats = {name: {str(c): describe(normal_arrays[f'{name}__score'][correct & array(normal_y).astype(int).__eq__(c)])
                        for c in range(config['model']['num_classes'])} for name in family.variants}
    save_json(directory/'normal_score_scales.json', {name: {'all_normal': describe(normal_arrays[f'{name}__score']),
                  'correct_normal_by_class': class_score_stats[name], 'threshold': thresholds[name]['threshold']}
                  for name in family.variants})

    clean_true = decompose(family, clean_z, y)
    layer_rows = []; sample_rows = []; summaries = {}; tensor_archive = {}
    max_witness_gap = 0.; max_cf_gap = 0.; max_triangle_excess = 0.; max_cf_opt_excess = 0.
    for method_index, name in enumerate(pool['names']):
        spec = family.variants[name]; rep = spec['representation']; tau = thresholds[name]['threshold']
        x = pool['witnesses'][:, method_index].to(device)
        logits, z = family.features(x); wrong = logits.argmax(1)
        adv = decompose(family, z, wrong, [name])[name]
        cf = decompose(family, clean_z, wrong, [name])[name]
        real = wrong.ne(y) & adv['score'].le(tau) & (x-originals.to(device)).flatten(1).abs().amax(1).le(config['layer_study']['joint_attack']['epsilon']+1e-6)
        real &= x.flatten(1).amin(1).ge(0) & x.flatten(1).amax(1).le(1) & torch.isfinite(adv['score'])
        if not real.all() or not pool['success'][:, method_index].all() or audit['witnesses'][name]['verified_success'] != len(y):
            raise AssertionError(f'Reloaded witness invalid: {name}')
        max_witness_gap = max(max_witness_gap, float(adv['max_relative_fw_gap'].max()))
        max_cf_gap = max(max_cf_gap, float(cf['max_relative_fw_gap'].max()))
        decomposition_errors.extend([float(adv['decomposition_absolute_error'].max()), float(cf['decomposition_absolute_error'].max())])
        own = clean_true[name]
        fixed_clean_objective = adv['regularizer'].clone()
        rows_for_method = []
        weights = {}
        for layer, item in adv['layers'].items():
            delta = (z[rep][layer-2].double()-clean_z[rep][layer-2].double()).norm(dim=1)
            fixed_clean = (clean_z[rep][layer-2].double()-item['reconstruction']).norm(dim=1)
            triangle_excess = ((fixed_clean-item['residual_l2']).abs()-delta).clamp_min(0)
            max_triangle_excess = max(max_triangle_excess, float(triangle_excess.max()))
            fixed_clean_objective += item['objective_factor'] * fixed_clean.square()/z[rep][layer-2].shape[1]
            bank = family.representations[rep]['banks'][item['bank']][layer-2]
            nn_wrong = array(nearest_reference(z[rep][layer-2], wrong, bank))
            nn_true = array(nearest_reference(z[rep][layer-2], y, bank))
            key = f'{rep}|{item["bank"]}|L{layer}'
            scales = distance_cache[key]
            d, e, e0 = array(delta), array(item['residual_l2']), array(cf['layers'][layer]['residual_l2'])
            for i, (true_c, wrong_c) in enumerate(zip(y.tolist(), wrong.tolist())):
                true_mask = correct & (array(normal_y) == true_c)
                wrong_mask = correct & (array(normal_y) == wrong_c)
                normal_e = normal_arrays[f'{name}__L{layer}__residual_l2'][wrong_mask]
                median_nn = float(np.median(scales['nn'][true_mask]))
                median_pair = float(np.median(scales['pair'][true_c]))
                row = {'variant': name, 'representation': rep, 'layer': layer, 'local_index': i,
                    'dataset_id': int(pool['ids'][i]), 'true_class': true_c, 'wrong_class': wrong_c,
                    'feature_dimension': z[rep][layer-2].shape[1], 'shift_l2': float(d[i]),
                    'normal_true_nn_median': median_nn, 'shift_over_normal_nn_median': float(d[i]/median_nn),
                    'shift_normal_nn_percentile': float(percentile(scales['nn'][true_mask], d[i])),
                    'shift_over_normal_pair_median': float(d[i]/median_pair),
                    'shift_normal_pair_percentile': float(percentile(scales['pair'][true_c], d[i])),
                    'small_shift_vs_normal_nn_q05': bool(d[i] <= np.quantile(scales['nn'][true_mask], cfg['small_shift_quantile'])),
                    'wrong_reconstruction_l2': float(e[i]), 'wrong_reconstruction_rms': float(array(item['residual_rms'])[i]),
                    'wrong_reconstruction_normal_percentile': float(percentile(normal_e, e[i])),
                    'high_reconstruction_vs_normal_class_q95': bool(e[i] > np.quantile(normal_e, cfg['high_residual_quantile'])),
                    'clean_wrong_refit_l2': float(e0[i]), 'clean_true_refit_l2': float(array(own['layers'][layer]['residual_l2'])[i]),
                    'wrong_reconstruction_drop_l2': float(e0[i]-e[i]),
                    'clean_to_adv_fixed_wrong_combination_l2': float(array(fixed_clean)[i]),
                    'nn_wrong_class': float(nn_wrong[i]), 'nn_true_class': float(nn_true[i]),
                    'wrong_nn_closer_than_true_nn': bool(nn_wrong[i] < nn_true[i]),
                    'score_layer_contribution': float(array(item['mse'])[i]*item['objective_factor']),
                    'threshold': tau, 'adv_score': float(array(adv['score'])[i]),
                    'clean_wrong_score': float(array(cf['score'])[i])}
                layer_rows.append(row); rows_for_method.append(row)
            weights[str(layer)] = {'adv': item['weights'].cpu(), 'clean_wrong': cf['layers'][layer]['weights'].cpu()}
        # Counterfactual optimum cannot exceed evaluating the adversarial weights on clean z,
        # apart from a certified optimization gap and numerical expansion error.
        max_cf_opt_excess = max(max_cf_opt_excess, float((cf['explicit_score']-fixed_clean_objective).clamp_min(0).max()))
        records = []
        for i, c in enumerate(wrong.tolist()):
            local = [r for r in rows_for_method if r['local_index'] == i]
            class_scores = normal_arrays[f'{name}__score'][correct & (array(normal_y) == c)]
            adv_score = float(array(adv['score'])[i]); cf_score = float(array(cf['score'])[i])
            record = {'variant': name, 'local_index': i, 'dataset_id': int(pool['ids'][i]),
                'true_class': int(y[i]), 'wrong_class': c, 'verified_success': bool(real[i]),
                'adv_score': adv_score, 'threshold': tau, 'score_over_threshold': adv_score/tau,
                'score_normal_wrong_class_percentile': float(percentile(class_scores, adv_score)),
                'score_high_vs_normal_wrong_class_q95': bool(adv_score > np.quantile(class_scores, cfg['high_residual_quantile'])),
                'clean_wrong_score': cf_score, 'clean_wrong_already_accepted': cf_score <= tau,
                'clean_true_score': float(array(own['score'])[i]),
                'small_shift_all_used_layers': all(r['small_shift_vs_normal_nn_q05'] for r in local),
                'high_reconstruction_any_used_layer': any(r['high_reconstruction_vs_normal_class_q95'] for r in local),
                'wrong_reconstruction_decreased_all_layers': all(r['wrong_reconstruction_drop_l2'] > 0 for r in local),
                'nn_wrong_closer_all_layers': all(r['wrong_nn_closer_than_true_nn'] for r in local),
                'regularizer': float(array(adv['regularizer'])[i]),
                'explicit_residual_score': sum(r['score_layer_contribution'] for r in local),
                'clean_logit_true_minus_wrong': float(clean_logits[i, y[i]]-clean_logits[i, c]),
                'adv_logit_wrong_minus_true': float(logits[i, c]-logits[i, y[i]])}
            sample_rows.append(record); records.append(record)
        summaries[name] = {'n': len(records), 'counts': {key: sum(r[key] for r in records) for key in [
            'verified_success', 'clean_wrong_already_accepted', 'small_shift_all_used_layers',
            'high_reconstruction_any_used_layer', 'score_high_vs_normal_wrong_class_q95',
            'wrong_reconstruction_decreased_all_layers', 'nn_wrong_closer_all_layers']},
            'score_over_threshold': describe([r['score_over_threshold'] for r in records]),
            'layers': {str(layer): {key: describe([r[key] for r in rows_for_method if r['layer']==layer]) for key in [
                'shift_over_normal_nn_median', 'shift_over_normal_pair_median', 'wrong_reconstruction_normal_percentile',
                'wrong_reconstruction_drop_l2']} for layer in spec['layers']}}
        tensor_archive[name] = {'classes': wrong.cpu(), 'weights_by_layer': weights,
            'z_clean': [clean_z[rep][l-2].cpu() for l in spec['layers']],
            'z_adv': [z[rep][l-2].cpu() for l in spec['layers']], 'layers': spec['layers']}
        print(f'Geometry {name}: verified {int(real.sum())}/{len(real)}, clean-wrong accepted {summaries[name]["counts"]["clean_wrong_already_accepted"]}', flush=True)
        save_json(directory/'summary.json', summaries)
    if max(normal_gaps+[max_witness_gap, max_cf_gap]) > cfg['relative_gap_tolerance']:
        raise AssertionError('A geometry QP failed the original convergence tolerance')
    if max(decomposition_errors) > cfg['decomposition_tolerance'] or max_triangle_excess > 1e-8 or max_cf_opt_excess > cfg['decomposition_tolerance']:
        raise AssertionError('Explicit residual / triangle / counterfactual optimum identity failed')
    torch.save({'normal_z': normal_z, 'normal_ids': normal_ids, 'normal_labels': normal_y,
                'witness_ids': pool['ids'], 'methods': tensor_archive}, directory/'geometry_tensors.pt')
    write_csv(directory/'per_layer.csv', layer_rows); write_csv(directory/'per_sample.csv', sample_rows)
    save_json(directory/'numerical_checks.json', {'max_normal_relative_fw_gap': max(normal_gaps),
        'max_witness_relative_fw_gap': max_witness_gap, 'max_counterfactual_relative_fw_gap': max_cf_gap,
        'max_score_decomposition_absolute_error': max(decomposition_errors),
        'max_fixed_combination_triangle_excess': max_triangle_excess,
        'max_counterfactual_optimum_excess_over_fixed_combination': max_cf_opt_excess,
        'verified_witness_method_pairs': len(sample_rows), 'normal_scale_n': n,
        'correct_normal_for_class_scales_n': int(correct.sum()), 'normal_and_witness_and_dictionary_disjoint': True})
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/experiments/studies/witness_geometry/cifar10_witness_geometry.yaml')
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    args = parser.parse_args(); config = load_config(args.config, args.overrides)
    directory = create_run(config, 'witness-geometry')
    print(f'Run directory: {directory}', flush=True)
    try:
        run(config, directory)
        finish_run(directory, 'complete')
        print(json.dumps({'status': 'complete', 'directory': str(directory)}, ensure_ascii=False))
    except BaseException as exc:
        finish_run(directory, 'failed', error=str(exc)); raise


if __name__ == '__main__': main()
