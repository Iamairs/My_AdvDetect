"""重新加载两个主方法，独立核验 P1 保存的联合绕过见证。"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts.studies.mechanisms.run_mechanisms import context, load_checkpoint, make_detectors
from advdetect.datasets.cifar import IndexedSubset, make_loader
from advdetect.utils.artifacts import create_run, finish_run, save_json
from advdetect.utils.config import load_config, resolve_path
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    args = parser.parse_args()
    source = resolve_path(args.run_dir)
    config = load_config(source/'config.yaml')
    directory = create_run(config, 'verify-mechanism-witnesses')
    try:
        device, data, model, binding = context(config, directory)
        detectors = make_detectors(model, config, device)
        checkpoints = json.loads((source/'result.json').read_text(encoding='utf-8'))['checkpoints']
        payload = torch.load(source/'adaptive_probe_witnesses.pt', map_location='cpu', weights_only=True)
        ids = payload['ids'].tolist()
        if not set(ids) <= set(data['dev_evaluate'].indices):
            raise ValueError('Witness indices are outside the bound development split')
        subset = IndexedSubset(data['dev_evaluate'].source, ids, 'witness_audit')
        original, labels, actual_ids = next(iter(make_loader(subset, {**config, 'batch_size': len(ids)})))
        torch.testing.assert_close(actual_ids, payload['ids'])
        torch.testing.assert_close(labels, payload['labels'])
        report = {}
        with torch.no_grad():
            for name in checkpoints:
                detector = detectors[name]
                load_checkpoint(name, detector, checkpoints[name], config, binding)
                mode = 'shared' if name == 'shared' else 'paired'
                cal = json.loads((source/f'development_thresholds_{name}.json').read_text(encoding='utf-8'))[mode]
                images = payload['witnesses'][name][mode].to(device)
                recorded = payload['unions'][name][mode]
                predictions = model(images).argmax(1).cpu()
                scores = detector.score(images, mode=mode).cpu()
                norms = (images.cpu()-original).flatten(1).abs().amax(1)
                valid = norms.le(config['adaptive_probe']['epsilon']+1e-6)
                valid &= images.cpu().flatten(1).amin(1).ge(0) & images.cpu().flatten(1).amax(1).le(1)
                real = predictions.ne(labels) & scores.le(cal['threshold']) & valid & torch.isfinite(scores)
                if (recorded & ~real).any():
                    raise AssertionError(f'{name}: saved success is not a real accepted misclassification')
                report[name] = {'recorded_success_count': int(recorded.sum()), 'reverified_success_count': int((recorded & real).sum()),
                                'max_linf_recorded': norms[recorded].max().item() if recorded.any() else None,
                                'threshold': cal['threshold'], 'scores': scores.tolist(),
                                'predictions': predictions.tolist(), 'labels': labels.tolist(), 'ids': ids}
        save_json(directory/'witness_verification.json', {'source_run': str(source), 'results': report})
        save_json(source/'witness_verification.json', {'audit_run': str(directory), 'results': report})
        finish_run(directory, 'complete')
        print(json.dumps(report, ensure_ascii=False, indent=2))
    except BaseException as exc:
        finish_run(directory, 'failed', error=str(exc))
        raise


if __name__ == '__main__':
    main()
