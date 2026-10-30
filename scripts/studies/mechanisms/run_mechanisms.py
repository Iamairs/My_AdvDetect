"""两个独立研究方法的 P0/P1 入口；不调用或修改原 Experiment/factory。"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from advdetect.attacks import build_attack
from advdetect.datasets.cifar import IndexedSubset, make_loader
from advdetect.evaluation.metrics import calibrate_threshold, detection_metrics
from advdetect.models import load_classifier
from advdetect.utils.artifacts import choose_device, create_run, finish_run, save_json, seed_everything, sha256_file, stable_hash
from advdetect.utils.config import load_config, resolve_path
from research.mechanisms.adaptive import joint_pgd
from research.mechanisms.common import StageBackbone, development_data, margin, union_metrics
from research.r002_shared_reference import SharedReferenceDetector
from research.r003_evidence_intervention import EvidenceInterventionDetector


def context(config, directory):
    seed_everything(int(config['seed']))
    torch.set_num_threads(int(config.get('cpu_threads', 4)))
    device = choose_device(config['device'])
    data, manifest = development_data(config)
    source = data['gate'].source
    digest = hashlib.sha256(source.data.tobytes())
    digest.update(np.asarray(source.targets, dtype='<i8').tobytes())
    manifest['data_sha256'] = digest.hexdigest()
    model, meta = load_classifier(config, device)
    classifier_hash = sha256_file(resolve_path(config['model']['checkpoint']))
    save_json(directory/'splits.json', manifest)
    save_json(directory/'classifier.json', {**meta, 'sha256': classifier_hash,
                                          'training_split_verified': False})
    save_json(directory/'entrypoint.json', {'path': str(Path(__file__).relative_to(ROOT)),
                                          'sha256': sha256_file(Path(__file__))})
    (directory/'entrypoint_snapshot.py').write_bytes(Path(__file__).read_bytes())
    binding = {'classifier_sha256': classifier_hash, 'model': config['model'],
               'dataset': config['dataset'], 'splits_sha256': stable_hash(manifest),
               'backbone_code_sha256': sha256_file(ROOT/'src/advdetect/models/backbones.py')}
    return device, data, model, binding


def make_detectors(model, config, device):
    return {'shared': SharedReferenceDetector(model, config['shared'], device),
            'evidence': EvidenceInterventionDetector(model, config['evidence'], device, config['model']['num_classes'])}


def binding_for(name, config, binding):
    paths = [ROOT/'research/mechanisms/common.py']
    paths += sorted((ROOT/'research'/('r002_shared_reference' if name == 'shared' else 'r003_evidence_intervention')).glob('*.py'))
    return {**binding, 'method_config': config[name], 'seed': config['seed'],
            'implementation': {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}}


def save_checkpoint(name, detector, config, binding, directory):
    path = resolve_path('checkpoints/mechanisms')/directory.name/f'{name}.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'format': 'advdetect_mechanism_v1', 'method': name, 'binding': binding_for(name, config, binding),
                'state': detector.state()}, path)
    return path


def load_checkpoint(name, detector, path, config, binding):
    payload = torch.load(resolve_path(path), map_location='cpu', weights_only=True)
    if payload.get('format') != 'advdetect_mechanism_v1' or payload['method'] != name:
        raise ValueError('Wrong mechanism checkpoint format/method')
    if payload['binding'] != binding_for(name, config, binding):
        raise ValueError('Checkpoint model/data/split/config/source binding mismatch')
    detector.restore(payload['state'])


def fit(detectors, data, config, binding, directory, selected):
    paths = {}
    for name in selected:
        started = time.perf_counter()
        if name == 'shared':
            result = detectors[name].fit(data['reference'], config)
        else:
            result = detectors[name].fit_gates(data['gate'], data['dev_select'], config, directory)
            result['profile_diagnostics'] = detectors[name].fit_profiles(data['profile'], config)
        paths[name] = save_checkpoint(name, detectors[name], config, binding, directory)
        save_json(directory/f'fit_{name}.json', {**result, 'elapsed_seconds': time.perf_counter()-started,
                                                'checkpoint': str(paths[name])})
        print(f'{name} checkpoint: {paths[name]}', flush=True)
    return paths


@torch.no_grad()
def collect(detector, dataset, config, device):
    chunks = {k: [] for k in detector.modes}
    cp, labels, ids = [], [], []
    durations = []
    for x, y, index in make_loader(dataset, config):
        x = x.to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        started = time.perf_counter()
        values = detector.scores(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        durations.append(time.perf_counter()-started)
        for k in chunks:
            chunks[k].append(values[k].cpu())
        cp.append(detector.model(x).argmax(1).cpu()); labels.append(y); ids.append(index)
    return {'scores': {k: torch.cat(v).numpy() for k, v in chunks.items()},
            'predictions': torch.cat(cp).numpy(), 'labels': torch.cat(labels).numpy(),
            'ids': torch.cat(ids).numpy(), 'all_modes_seconds': sum(durations)}


def gradient_probe(detector, image, target, mode):
    # Finite differences in float32 suffer cancellation/TF32 quantization. Check the
    # exact same operations/constants on a separate float64 copy; deployment stays unchanged.
    detector = deepcopy(detector)
    detector.model.double()
    detector.view = StageBackbone(detector.model)
    if hasattr(detector, '_grams'):
        detector._grams = {}
    if hasattr(detector, 'theta'):
        detector.theta.double()
        detector.means = [v.double() for v in detector.means]
        detector.profiles = {k: {key: value.double() if isinstance(value, torch.Tensor) else value
                                for key, value in profile.items()} for k, profile in detector.profiles.items()}
    image = image.double()
    x = image.detach().requires_grad_(True)
    value = detector.score(x, classes=target, mode=mode)
    grad = torch.autograd.grad(value.sum(), x)[0]
    direction = torch.randn_like(x); direction /= direction.norm()
    reports = []
    for h in [1e-3, 1e-4, 1e-5, 1e-6]:
        with torch.no_grad():
            finite = (detector.score(x+h*direction, target, mode)-detector.score(x-h*direction, target, mode))/(2*h)
        analytic = (grad*direction).sum()
        reports.append({'step': h, 'finite_difference': finite.item(), 'autograd': analytic.item(),
                        'relative_error': abs(finite.item()-analytic.item())/max(abs(finite.item()), abs(analytic.item()), 1e-6)})
    return {'arithmetic': 'float64 copy; actual float32 deployment is unchanged',
            'input_gradient_finite': bool(torch.isfinite(grad).all()),
            'input_gradient_max': grad.abs().max().item(), 'directional_checks': reports}


@torch.no_grad()
def numerical_diagnostics(detector, dataset, config, device):
    totals = {str(n): {'objectives': [], 'relative_fw_gaps': []} for n in config['shared']['diagnostic_steps']}
    gaps = []
    for images, _, _ in make_loader(dataset, config):
        logits, z = detector.features(images.to(device))
        classes = logits.argmax(1)
        for steps in config['shared']['diagnostic_steps']:
            _, value, fw = detector.objective(z, classes, steps=steps)
            totals[str(steps)]['objectives'].append(value.cpu())
            totals[str(steps)]['relative_fw_gaps'].append((fw/(1+value.abs())).cpu())
        shared = detector.objective(z, classes)[1]
        free = sum(detector.objective(z, classes, layers=[j])[1]/len(z) for j in range(len(z)))
        gaps.append((shared-free).cpu())
    result = {}
    for steps, values in totals.items():
        objective = torch.cat(values['objectives']); fw = torch.cat(values['relative_fw_gaps'])
        result[steps] = {'objective_mean': objective.mean().item(), 'relative_fw_gap_max': fw.max().item(),
                         'relative_fw_gap_mean': fw.mean().item(),
                         'not_converged_fraction': (fw>config['shared']['relative_gap_tolerance']).float().mean().item()}
    gaps = torch.cat(gaps)
    result['shared_minus_independent'] = {'min': gaps.min().item(), 'mean': gaps.mean().item(),
                                          'negative_beyond_1e-5': int((gaps < -1e-5).sum())}
    return result


def paired_bootstrap(a, b, seed=42, repetitions=2000):
    if not len(a):
        return None
    rng = np.random.default_rng(seed)
    delta = np.asarray(a, float)-np.asarray(b, float)
    boot = [delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(repetitions)]
    return {'difference': float(delta.mean()), 'ci95': np.quantile(boot, [.025, .975]).tolist(),
            'unit': 'original_image', 'repetitions': repetitions}


@torch.no_grad()
def benchmark(detectors, images, device):
    result = {}
    operations = {'classifier': lambda: next(iter(detectors.values())).model(images)}
    operations.update({name: (lambda d=d, mode='shared' if name == 'shared' else 'paired': d.score(images, mode=mode))
                       for name, d in detectors.items()})
    for name, operation in operations.items():
        operation()
        if device.type == 'cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(5):
            start = time.perf_counter(); operation()
            if device.type == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter()-start)
        result[name] = {'batch_size': len(images), 'median_batch_seconds': float(np.median(times)),
                         'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None}
    for name in detectors:
        result[name]['relative_to_classifier'] = result[name]['median_batch_seconds']/result['classifier']['median_batch_seconds']
    return result


def development_experiment(detectors, data, config, device, directory):
    thresholds, clean, unions, ordinary, counts = {}, {}, {}, {}, {}
    selected = list(detectors)
    for name, detector in detectors.items():
        cal = collect(detector, data['dev_threshold'], config, device)
        clean[name] = collect(detector, data['dev_evaluate'], config, device)
        thresholds[name] = {mode: calibrate_threshold(values, config['evaluation']['target_fpr'])
                            for mode, values in cal['scores'].items()}
        for v in thresholds[name].values():
            v.update({'role': 'development_threshold_only', 'all_normal_samples': True,
                      'not_final_calibration': True})
        save_json(directory/f'development_thresholds_{name}.json', thresholds[name])
        np.savez_compressed(directory/f'dev_threshold_scores_{name}.npz', **cal['scores'], ids=cal['ids'])
        unions[name] = {mode: np.zeros(len(data['dev_evaluate']), dtype=bool) for mode in detector.modes}
        ordinary[name] = {mode: [] for mode in detector.modes}
    # Generate once, use exactly the same ordinary PGD images for every method/ablation.
    attack = build_attack(next(iter(detectors.values())).model, config['attack'])
    adv_predictions = []; adv_images = []; max_linf = 0.
    offset = 0
    for batch, (images, y, ids) in enumerate(make_loader(data['dev_evaluate'], config)):
        seed_everything(int(config['seed'])+200+batch)
        images = images.to(device)
        with torch.enable_grad():
            adv = attack(images, y.to(device))
        max_linf = max(max_linf, (adv-images).abs().max().item())
        if not torch.isfinite(adv).all() or max_linf > float(config['attack']['epsilon'])+1e-6 or adv.min() < 0 or adv.max() > 1:
            raise RuntimeError('Invalid ordinary attack')
        with torch.no_grad():
            ap = next(iter(detectors.values())).model(adv).argmax(1).cpu().numpy()
            adv_predictions.append(ap); adv_images.append(adv.cpu())
            for name, detector in detectors.items():
                values = detector.scores(adv)
                for mode, value in values.items():
                    score = value.cpu().numpy()
                    ordinary[name][mode].append(score)
                    tau = thresholds[name][mode]['threshold']
                    unions[name][mode][offset:offset+len(y)] |= (ap != y.numpy()) & (score <= (float('inf') if tau is None else tau))
        offset += len(y)
        print(f'P1 common PGD {offset}/{len(data["dev_evaluate"])}', flush=True)
    adv_predictions = np.concatenate(adv_predictions)
    adv_images = torch.cat(adv_images)
    torch.save({'format': 'mechanisms_development_attacks_v1', 'images': adv_images,
                'ids': torch.tensor(next(iter(clean.values()))['ids']), 'attack': config['attack']}, directory/'common_pgd.pt')
    reports, score_payload = {}, {}
    for name, detector in detectors.items():
        c = clean[name]; reports[name] = {}
        for mode in detector.modes:
            adv = np.concatenate(ordinary[name][mode])
            reports[name][mode] = detection_metrics(c['scores'][mode], adv, c['predictions'], adv_predictions,
                                                  c['labels'], thresholds[name][mode])
            tau = thresholds[name][mode]['threshold']
            reports[name][mode].update(union_metrics(c['scores'][mode], c['labels'], c['predictions'],
                unions[name][mode], float('inf') if tau is None else tau))
            reports[name][mode]['max_linf'] = max_linf
            score_payload[f'{name}_clean_{mode}'] = c['scores'][mode]
            score_payload[f'{name}_adv_{mode}'] = adv
        score_payload[f'{name}_ids'] = c['ids']
        reports[name]['all_modes_seconds_clean_1000'] = c['all_modes_seconds']
    sample = next(iter(clean.values()))
    score_payload.update(labels=sample['labels'], clean_predictions=sample['predictions'], adv_predictions=adv_predictions)
    # Confidence-bin diagnostics use bins fixed on development normal margins.
    edges = np.quantile(-sample['scores']['margin'], np.linspace(0, 1, 6))
    bins = np.digitize(-sample['scores']['margin'], edges[1:-1])
    binned = {}
    for name in detectors:
        binned[name] = {}
        for mode in detectors[name].modes:
            values = []
            for b in range(5):
                mask = bins == b
                success = mask & (sample['predictions'] == sample['labels']) & (adv_predictions != sample['labels'])
                a = score_payload[f'{name}_adv_{mode}'][success]
                c = score_payload[f'{name}_clean_{mode}'][mask]
                values.append({'bin': b, 'clean_count': int(mask.sum()), 'success_count': int(success.sum()),
                               'auroc': float(roc_auc_score(np.r_[np.zeros(len(c)), np.ones(len(a))], np.r_[c, a]))
                               if len(c) and len(a) else None})
            binned[name][mode] = values
    save_json(directory/'margin_bins.json', {'edges': edges.tolist(), 'results': binned})
    # Save ordinary results before the small attack implementation probe.
    save_json(directory/'ordinary_metrics.json', reports)
    np.savez_compressed(directory/'scores.npz', **score_payload)
    if 'evidence' in detectors:
        responses = {'clean': [], 'adv': [], 'clean_energy_error': [], 'adv_energy_error': []}
        offset = 0
        with torch.no_grad():
            for images, _, _ in make_loader(data['dev_evaluate'], config):
                for prefix, x in [('clean', images.to(device)), ('adv', adv_images[offset:offset+len(images)].to(device))]:
                    _, _, values, diag = detectors['evidence'].responses(x, include_random=False)
                    responses[prefix].append(values['paired'].cpu())
                    responses[f'{prefix}_energy_error'].append(diag['energy_error'].cpu())
                offset += len(images)
        np.savez_compressed(directory/'evidence_responses.npz', **{k: torch.cat(v).numpy() for k, v in responses.items()},
                            ids=sample['ids'], labels=sample['labels'])
    if 'shared' in detectors:
        diag = numerical_diagnostics(detectors['shared'], data['dev_evaluate'], config, device)
        save_json(directory/'p0_shared_convergence_1000.json', diag)
    probe = config.get('adaptive_probe', {})
    if probe.get('enabled', False):
        n = min(int(probe['samples']), len(data['dev_evaluate']))
        subset = IndexedSubset(data['dev_evaluate'].source, data['dev_evaluate'].indices[:n], 'adaptive_probe')
        images, y, ids = next(iter(make_loader(subset, {**config, 'batch_size': n})))
        images, yd = images.to(device), y.to(device)
        witness = {name: {mode: images.cpu().clone() for mode in detector.modes} for name, detector in detectors.items()}
        probe_union = {name: {mode: np.zeros(n, dtype=bool) for mode in detector.modes} for name, detector in detectors.items()}

        @torch.no_grad()
        def observer(candidate, tag):
            ap = next(iter(detectors.values())).model(candidate).argmax(1)
            valid = (candidate-images).flatten(1).abs().amax(1).le(float(probe['epsilon'])+1e-6)
            valid &= candidate.flatten(1).amin(1).ge(0) & candidate.flatten(1).amax(1).le(1)
            for name, detector in detectors.items():
                values = detector.scores(candidate)
                for mode, value in values.items():
                    tau = thresholds[name][mode]['threshold']
                    hit = ap.ne(yd) & valid & value.le(float('inf') if tau is None else tau)
                    new = hit.cpu().numpy() & (~probe_union[name][mode])
                    witness[name][mode][new] = candidate.detach().cpu()[new]
                    probe_union[name][mode] |= hit.cpu().numpy()
        observer(images, {'kind': 'original'})
        # Include the same ordinary PGD candidates in the common pool and witness ledger.
        observer(adv_images[:n].to(device), {'kind': 'ordinary_pgd'})
        traces = {}
        for name, detector in detectors.items():
            mode = 'shared' if name == 'shared' else 'paired'
            tau = thresholds[name][mode]['threshold']
            if tau is None:
                raise ValueError('Development threshold sample count insufficient for adaptive probe')
            cfg = {**probe, 'score_scale': thresholds[name][mode]['score_scale']}
            result = joint_pgd(detector.model, detector, images, yd, tau, cfg, mode, observer)
            traces[name] = result['trace']
            print(f'Adaptive implementation probe {name}: {int(result["success"].sum())}/{n}', flush=True)
        probe_result = {}
        correct = sample['predictions'][:n] == sample['labels'][:n]
        for name, detector in detectors.items():
            probe_result[name] = {}
            for mode in detector.modes:
                tau = thresholds[name][mode]['threshold']
                probe_result[name][mode] = union_metrics(clean[name]['scores'][mode][:n], sample['labels'][:n],
                    sample['predictions'][:n], probe_union[name][mode], float('inf') if tau is None else tau)
        torch.save({'witnesses': witness, 'unions': {name: {k: torch.from_numpy(v) for k, v in vals.items()}
                    for name, vals in probe_union.items()}, 'ids': ids, 'labels': y}, directory/'adaptive_probe_witnesses.pt')
        save_json(directory/'adaptive_probe.json', {'scope': 'small implementation probe, not adaptive robustness evidence',
                  'configuration': probe, 'common_candidate_pool': True, 'results': probe_result, 'traces': traces})
    comparisons = {}
    correct = sample['predictions'] == sample['labels']
    for name, main_mode, comparison in [('shared', 'shared', ['independent', 'shuffled', 'margin']),
                                         ('evidence', 'paired', ['random_pair', 'no_control', 'margin'])]:
        if name in detectors:
            comparisons[name] = {mode: paired_bootstrap(unions[name][mode][correct], unions[name][main_mode][correct])
                                 for mode in comparison}
    save_json(directory/'ordinary_paired_bootstrap.json', comparisons)
    # Numerical real-network gradient checks are run with gradients explicitly enabled.
    x, _, _ = next(iter(make_loader(data['dev_evaluate'], config)))
    x = x[:1].to(device)
    save_json(directory/'main_mode_latency.json', benchmark(detectors, x, device))
    target = (next(iter(detectors.values())).model(x).argmax(1)+1) % config['model']['num_classes']
    with torch.enable_grad():
        gradient_results = {name: gradient_probe(detector, x, target, 'shared' if name == 'shared' else 'paired')
                            for name, detector in detectors.items()}
    save_json(directory/'p0_real_network_gradients.json', gradient_results)
    return {'scope': 'exploratory_P0_P1_development_only', 'num_development_images': len(data['dev_evaluate']),
            'classifier_training_independence_verified': False, 'final_calibration_used': False,
            'official_test_used': False, 'ordinary_metrics': reports, 'bootstrap': comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['fit', 'p1', 'run'])
    parser.add_argument('--config', default='configs/experiments/studies/mechanisms/cifar10_mechanisms_p1.yaml')
    parser.add_argument('--method', choices=['shared', 'evidence', 'both'], default='both')
    parser.add_argument('--shared-checkpoint'); parser.add_argument('--evidence-checkpoint')
    parser.add_argument('--set', dest='overrides', action='append', default=[])
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)
    directory = create_run(config, f'mechanisms-{args.command}')
    print(f'RUN_DIRECTORY={directory}', flush=True)
    try:
        device, data, model, binding = context(config, directory)
        detectors = make_detectors(model, config, device)
        selected = ['shared', 'evidence'] if args.method == 'both' else [args.method]
        detectors = {k: detectors[k] for k in selected}
        paths = {}
        if args.command in ('fit', 'run'):
            paths = fit(detectors, data, config, binding, directory, selected)
        else:
            for name, detector in detectors.items():
                path = getattr(args, f'{name}_checkpoint')
                if not path:
                    parser.error(f'--{name}-checkpoint is required for p1')
                load_checkpoint(name, detector, path, config, binding)
                paths[name] = path
        result = {'checkpoints': {k: str(v) for k, v in paths.items()}}
        if args.command in ('p1', 'run'):
            result.update(development_experiment(detectors, data, config, device, directory))
        save_json(directory/'result.json', result)
        finish_run(directory, 'complete')
        print(json.dumps({'directory': str(directory), 'checkpoints': result['checkpoints'], 'status': 'complete'}, ensure_ascii=False), flush=True)
    except BaseException as exc:
        finish_run(directory, 'failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
