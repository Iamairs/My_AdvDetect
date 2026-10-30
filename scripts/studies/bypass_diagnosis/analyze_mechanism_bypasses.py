"""独立分析现有16组联合绕过，并在固定开发协议上做机制验证。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT=next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts.studies.mechanisms.run_mechanisms import context,load_checkpoint,make_detectors
from advdetect.utils.artifacts import create_run,finish_run,save_json,sha256_file
from advdetect.utils.config import load_config,resolve_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['analyze','verify'])
    parser.add_argument('--config',default='configs/experiments/studies/bypass_diagnosis/cifar10_bypass_diagnosis.yaml')
    parser.add_argument('--analysis-dir')
    parser.add_argument('--set',dest='overrides',action='append',default=[])
    args=parser.parse_args()
    if args.phase=='verify' and not args.analysis_dir:
        parser.error('verify requires --analysis-dir')
    config=load_config(args.config,args.overrides)
    directory=create_run(config,f'bypass-{args.phase}')
    print(f'RUN_DIRECTORY={directory}',flush=True)
    try:
        device,data,model,binding=context(config,directory)
        detectors=make_detectors(model,config,device)
        source=resolve_path(config['diagnosis']['source_run'])
        checkpoints=json.loads((source/'result.json').read_text(encoding='utf-8'))['checkpoints']
        for name,d in detectors.items():
            load_checkpoint(name,d,checkpoints[name],config,binding)
        save_json(directory/'diagnosis_entrypoint.json',{'path':str(Path(__file__).relative_to(ROOT)),
                  'sha256':sha256_file(Path(__file__))})
        (directory/'diagnosis_entrypoint_snapshot.py').write_bytes(Path(__file__).read_bytes())
        if args.phase=='analyze':
            from research.r004_bypass_diagnosis.details import analyze
            result=analyze(detectors,data,config,directory)
        else:
            from research.r004_bypass_diagnosis.verification import verify
            result=verify(detectors,data,config,directory,resolve_path(args.analysis_dir))
        save_json(directory/'result.json',result)
        finish_run(directory,'complete')
        print(json.dumps({'directory':str(directory),'status':'complete'},ensure_ascii=False),flush=True)
    except BaseException as exc:
        finish_run(directory,'failed',error=str(exc));raise


if __name__=='__main__':
    main()
