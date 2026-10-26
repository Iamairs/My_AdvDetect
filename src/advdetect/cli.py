from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback

from .utils.config import load_config, resolve_path


def summarize(output_root: str):
    files = sorted(resolve_path(output_root).glob("*/metrics.json"))
    rows = []
    for path in files:
        metrics = json.loads(path.read_text(encoding="utf-8"))
        metadata = json.loads((path.parent / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("status") != "complete":
            continue
        rows.append({"run_id": path.parent.name, "command": metadata["command"], **metrics})
    if not rows:
        print("没有已完成的评价结果。")
        return
    destination = resolve_path(output_root) / "summary.csv"
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"runs": len(rows), "summary": str(destination)}, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="对抗检测研究工程：资源检查、拟合、攻击与统一评价")
    sub = parser.add_subparsers(dest="command", required=True)
    commands = {
        "check-resources": "校验导入资源、模型前向与输入梯度",
        "evaluate-classifier": "检查已有分类器准确率",
        "fit-detector": "拟合检测器并保存权重",
        "generate-attacks": "生成或复用经过身份校验的攻击缓存",
        "evaluate": "加载检测器，独立校准阈值并评价",
        "run": "执行拟合、校准、攻击、评价的完整流程",
    }
    for name, help_text in commands.items():
        command = sub.add_parser(name, help=help_text)
        command.add_argument("--config", default="configs/experiments/cifar10_msp_smoke.yaml")
        command.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
        if name == "evaluate":
            command.add_argument("--detector-checkpoint")
        if name == "evaluate-classifier":
            command.add_argument("--split", choices=["calibration", "test", "full-test"], default="test")
    report = sub.add_parser("summarize", help="汇总已完成评价的原始指标，不自动混合不同实验")
    report.add_argument("--output-root", default="outputs")
    args = parser.parse_args(argv)
    if args.command == "summarize":
        summarize(args.output_root)
        return
    # Keep --help and config inspection independent from accelerator initialization.
    from .utils.artifacts import create_run, finish_run, save_json
    from .experiments.workflow import Experiment

    config = load_config(args.config, args.overrides)
    directory = create_run(config, args.command)
    print(f"运行目录：{directory}", flush=True)
    try:
        experiment = Experiment(config, directory)
        if args.command == "check-resources":
            result = experiment.inspect()
        elif args.command == "evaluate-classifier":
            result = experiment.evaluate_classifier(args.split)
        elif args.command == "fit-detector":
            _, checkpoint = experiment.fit()
            result = {"detector_checkpoint": str(checkpoint)}
        elif args.command == "run":
            detector, checkpoint = experiment.fit()
            result = experiment.evaluate(detector, checkpoint)
        elif args.command == "evaluate":
            if not args.detector_checkpoint:
                parser.error("evaluate requires --detector-checkpoint; use run for automatic fitting")
            detector, checkpoint = experiment.restore(args.detector_checkpoint)
            result = experiment.evaluate(detector, checkpoint)
        else:
            payload, cache = experiment.generate()
            result = {"attack_cache": str(cache), "num_samples": int(payload["shape"][0]),
                      "cache_format": payload.get("format"), "status": payload.get("status")}
        finish_run(directory, "complete")
        save_json(directory / "result.json", result)
        (directory / "run.log").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException as error:
        finish_run(directory, "failed", error=str(error))
        (directory / "run.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
