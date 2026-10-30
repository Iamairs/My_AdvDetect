"""重新加载固定表征变体，核验重新攻击结果及新表征的梯度。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import torch

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.studies.mechanisms.run_mechanisms import context,make_detectors,load_checkpoint,gradient_probe
from advdetect.utils.artifacts import create_run,finish_run,save_json,sha256_file
from advdetect.utils.config import load_config,resolve_path
from research.r002_shared_reference.detector import SharedReferenceDetector


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--analysis-dir',required=True)
    parser.add_argument('--verification-dir',required=True)
    args=parser.parse_args()
    analysis=resolve_path(args.analysis_dir);verification=resolve_path(args.verification_dir)
    config=load_config(verification/'config.yaml')
    directory=create_run(config,'bypass-independent-audit')
    try:
        device,data,model,binding=context(config,directory)
        detectors=make_detectors(model,config,device)
        source=resolve_path(config['diagnosis']['source_run'])
        checkpoints=json.loads((source/'result.json').read_text(encoding='utf-8'))['checkpoints']
        for name,d in detectors.items():
            load_checkpoint(name,d,checkpoints[name],config,binding)
        study=torch.load(analysis/'diagnosis.pt',map_location='cpu',weights_only=True)
        states=torch.load(verification/'verification_variants.pt',map_location='cpu',weights_only=True)
        original=study['roles']['clean'].to(device);labels=study['labels'].to(device)
        result={}
        for name in ['shared','evidence',*config['diagnosis']['reattack']]:
            payload=torch.load(verification/f'attack_{name}.pt',map_location='cpu',weights_only=True)
            if name in ('shared','evidence'):
                d=detectors[name];mode='shared' if name=='shared' else 'paired'
            elif name=='evidence_no_control':
                d=detectors['evidence'];mode='no_control'
            else:
                d=SharedReferenceDetector(model,states[name]['config'],device)
                d.restore(states[name]['state']);mode='shared'
            x=payload['best_images'].to(device);recorded=payload['success'].to(device)
            with torch.no_grad():
                score=d.score(x,mode=mode);pred=model(x).argmax(1)
                norm=(x-original).flatten(1).abs().amax(1)
                valid=norm.le(config['adaptive_probe']['epsilon']+1e-6)
                valid&=x.flatten(1).amin(1).ge(0)&x.flatten(1).amax(1).le(1)
                real=pred.ne(labels)&score.le(payload['calibration']['threshold'])&valid&torch.isfinite(score)
                if (recorded&~real).any():
                    raise AssertionError(f'{name}: reloaded witness is invalid')
                row={'recorded_success_count':int(recorded.sum()),'reverified_success_count':int((real&recorded).sum()),
                     'max_linf':norm.max().item(),'scores':score.cpu().tolist(),
                     'threshold':payload['calibration']['threshold']}
                if isinstance(d,SharedReferenceDetector):
                    _,z=d.features(x)
                    _,v,g=d.objective(z,pred,steps=config['diagnosis']['detailed_qp_steps'])
                    row['high_precision_relative_fw_gap_max']=(g/(1+v.abs())).max().item()
                    row['deployment_vs_high_precision_max_abs_error']=(v-score).abs().max().item()
            if name in config['diagnosis']['reattack']:
                target=(model(original[:1]).argmax(1)+1)%config['model']['num_classes']
                row['gradient_probe']=gradient_probe(d,original[:1],target,mode)
            result[name]=row
            print(f'Audited {name}: {row["reverified_success_count"]}/{len(labels)}',flush=True)
        report={'audit_run':str(directory),'results':result,'source_verification':str(verification)}
        save_json(directory/'independent_audit.json',report)
        save_json(verification/'independent_audit.json',report)
        save_json(directory/'actual_entrypoint.json',{'path':str(Path(__file__).relative_to(ROOT)),
                  'sha256':sha256_file(Path(__file__))})
        finish_run(directory,'complete')
        print(json.dumps({'directory':str(directory),'status':'complete'},ensure_ascii=False))
    except BaseException as exc:
        finish_run(directory,'failed',error=str(exc));raise


if __name__=='__main__':
    main()
