"""Sequential experiment sweeps with explicit plans and per-run status."""
from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="预览或顺序运行多配置实验")
    parser.add_argument("--sweep", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="实际运行；默认只预览组合")
    parser.add_argument("--max-runs", type=int, default=100)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = args.sweep if args.sweep.is_absolute() else root / args.sweep
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    grid = cfg.get("grid", {})
    if any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("Each grid axis must be a nonempty list")
    combinations = list(itertools.product(*grid.values()))
    if len(combinations) > args.max_runs:
        raise ValueError(f"{len(combinations)} runs exceed --max-runs")
    plan = []
    for values in combinations:
        overrides = [f"{key}={json.dumps(value, ensure_ascii=False)}" for key, value in zip(grid, values)]
        command = [sys.executable, str(root / "run.py"), "run", "--config", cfg["experiment"]]
        for value in overrides:
            command.extend(["--set", value])
        plan.append({"overrides": overrides, "command": command, "status": "planned"})
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if not args.execute:
        return
    output = root / "outputs" / f"sweep_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        output.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")

    save()
    for item in plan:
        item["status"] = "running"
        save()
        result = subprocess.run(item["command"], cwd=root)
        item.update(returncode=result.returncode, status="complete" if result.returncode == 0 else "failed")
        save()
        if result.returncode:
            raise SystemExit(result.returncode)
    print(f"批量运行记录：{output}")


if __name__ == "__main__":
    main()
