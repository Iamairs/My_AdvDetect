"""Summarize saved scores without running the model or attack again."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description="离线分析已保存的逐样本分数")
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    directory = args.run if args.run.is_absolute() else root / args.run
    with np.load(directory / "scores.npz") as values:
        successful = (values["clean_predictions"] == values["labels"]) & (values["adv_predictions"] != values["labels"])
        groups = {"clean": values["clean_scores"], "all_generated": values["adv_scores"],
                  "successful_attacks": values["adv_scores"][successful],
                  "clean_misclassified": values["clean_scores"][values["clean_predictions"] != values["labels"]]}
        result = {}
        for name, scores in groups.items():
            result[name] = {"count": len(scores)}
            if len(scores):
                result[name].update(mean=float(scores.mean()), std=float(scores.std()),
                                    quantiles={str(q): float(np.quantile(scores, q)) for q in (0, 0.25, 0.5, 0.75, 0.95, 1)})
    (directory / "score_analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
