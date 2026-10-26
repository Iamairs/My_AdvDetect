"""Run the independent LR innovation research workflow.

This entry point deliberately does not use the stable LR detector factory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from advdetect.attacks import build_attack
from advdetect.datasets import build_data, make_loader
from advdetect.evaluation import calibrate_threshold, detection_metrics
from advdetect.models import load_classifier
from advdetect.utils.artifacts import choose_device, create_run, finish_run, save_json, seed_everything, sha256_file, stable_hash
from advdetect.utils.config import load_config, resolve_path
from research.r001_lr_innovation import LRInnovationDetector


def loader_factory(config, dataset, shuffle=False, seed=None):
    return make_loader(dataset, config, shuffle=shuffle, seed=seed)


def build_detector(config, device):
    model, classifier_meta = load_classifier(config, device)
    detector = LRInnovationDetector(model, config["detector"], device)
    return model, detector, classifier_meta


def fit(config, directory):
    device = choose_device(config["device"])
    data, manifest = build_data(config)
    model, detector, classifier_meta = build_detector(config, device)
    binding = {
        "classifier_sha256": sha256_file(resolve_path(config["model"]["checkpoint"])),
        "data_sha256": manifest["data_sha256"],
        "fit_split_sha256": manifest["splits"]["fit"]["sha256"],
        "selection_split_sha256": manifest["splits"]["selection"]["sha256"],
        "calibration_split_sha256": manifest["splits"]["calibration"]["sha256"],
        "detector_config_sha256": stable_hash(config["detector"]),
    }
    selection_loader = loader_factory(config, data["selection"], shuffle=False, seed=config["seed"])
    history = detector.fit(
        data["fit"], selection_loader,
        lambda dataset, shuffle, seed: loader_factory(config, dataset, shuffle, seed),
        seed=int(config["seed"]),
    )
    checkpoint = resolve_path("checkpoints/research/lr_innovation") / directory.name / "detector.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=False)
    torch.save({
        "format": "advdetect_lr_innovation_v1",
        "config": config["detector"],
        "dataset": config["dataset"],
        "model": config["model"],
        "manifest": manifest,
        "classifier_meta": classifier_meta,
        "binding": binding,
        "state": detector.state(),
        "history": history,
    }, checkpoint)
    save_json(directory / "crossfit.json", history)
    result = {"detector_checkpoint": str(checkpoint), "protocol": history["protocol"],
              "folds": history["folds"], "selected_epoch": history["selected_epoch"]}
    save_json(directory / "result.json", result)
    return result


def restore(config, checkpoint, device, manifest):
    payload = torch.load(resolve_path(checkpoint), map_location="cpu", weights_only=True)
    if payload.get("format") != "advdetect_lr_innovation_v1":
        raise ValueError("Unsupported LR innovation checkpoint")
    binding = payload.get("binding", {})
    expected = {
        "classifier_sha256": sha256_file(resolve_path(config["model"]["checkpoint"])),
        "data_sha256": manifest["data_sha256"],
        "fit_split_sha256": manifest["splits"]["fit"]["sha256"],
        "selection_split_sha256": manifest["splits"]["selection"]["sha256"],
        "calibration_split_sha256": manifest["splits"]["calibration"]["sha256"],
        "detector_config_sha256": stable_hash(config["detector"]),
    }
    if binding != expected:
        raise ValueError("LR innovation checkpoint binding differs from classifier, data split, or config")
    model, detector, _ = build_detector(config, device)
    detector.restore(payload["state"])
    return model, detector, payload


def score_loader(model, detector, loader, device):
    scores = {}
    predictions, labels, ids = [], [], []
    modes = tuple(detector.config.get("score_modes", ["mse", "conditional_diag"]))
    for images, y, index in loader:
        images, y = images.to(device), y.to(device)
        batch_scores = detector.score_all(images)
        for mode in modes:
            if mode not in batch_scores:
                raise ValueError(f"Unknown or unavailable score mode: {mode}")
            scores.setdefault(mode, []).append(batch_scores[mode].detach().cpu())
        predictions.append(model(images).argmax(1).detach().cpu())
        labels.append(y.cpu())
        ids.append(index.cpu())
    return {"scores": {mode: torch.cat(values) for mode, values in scores.items()},
            "predictions": torch.cat(predictions), "labels": torch.cat(labels),
            "indices": torch.cat(ids)}


def evaluate(config, checkpoint, directory):
    device = choose_device(config["device"])
    data, manifest = build_data(config)
    model, detector, payload = restore(config, checkpoint, device, manifest)
    model.eval()
    with torch.no_grad():
        calibration = score_loader(model, detector, loader_factory(config, data["calibration"]), device)
    correct = calibration["predictions"].eq(calibration["labels"])
    target_fpr = float(config["evaluation"]["target_fpr"])
    thresholds = {}
    for mode, values in calibration["scores"].items():
        selected = values[correct] if config["evaluation"].get("calibrate_on_correct_only", True) else values
        thresholds[mode] = calibrate_threshold(selected.numpy(), target_fpr)
        thresholds[mode].update({"mode": mode, "calibration_correct": int(correct.sum()),
                                 "calibration_raw_size": len(values)})
        save_json(directory / f"calibration_{mode}.json", thresholds[mode])

    attack = build_attack(model, config["attack"])
    clean_scores = {mode: [] for mode in thresholds}
    adv_scores = {mode: [] for mode in thresholds}
    clean_predictions, adv_predictions, labels = [], [], []
    test_indices = []
    for images, y, indices in loader_factory(config, data["test"]):
        images, y_device = images.to(device), y.to(device)
        with torch.enable_grad():
            adv = attack(images, y_device)
        with torch.no_grad():
            clean_batch = detector.score_all(images)
            adv_batch = detector.score_all(adv)
            for mode in thresholds:
                clean_scores[mode].append(clean_batch[mode].cpu())
                adv_scores[mode].append(adv_batch[mode].cpu())
            clean_predictions.append(model(images).argmax(1).cpu())
            adv_predictions.append(model(adv).argmax(1).cpu())
            labels.append(y)
            test_indices.append(indices)
    cp, ap, labels = torch.cat(clean_predictions), torch.cat(adv_predictions), torch.cat(labels)
    test_indices = torch.cat(test_indices)
    result = {"checkpoint": str(resolve_path(checkpoint)), "attack": config["attack"]["name"],
              "num_samples": len(labels), "scores": {}}
    for mode in thresholds:
        metrics = detection_metrics(
            torch.cat(clean_scores[mode]).numpy(), torch.cat(adv_scores[mode]).numpy(),
            cp.numpy(), ap.numpy(), labels.numpy(), thresholds[mode],
        )
        result["scores"][mode] = metrics
    score_payload = {"indices": test_indices.numpy(), "labels": labels.numpy(),
                     "clean_predictions": cp.numpy(), "adv_predictions": ap.numpy()}
    for mode in thresholds:
        score_payload[f"clean_{mode}"] = torch.cat(clean_scores[mode]).numpy()
        score_payload[f"adv_{mode}"] = torch.cat(adv_scores[mode]).numpy()
    np.savez_compressed(directory / "scores_by_mode.npz", **score_payload)
    save_json(directory / "metrics_by_mode.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description="独立 LR 条件标准化创新量实验")
    parser.add_argument("command", choices=["fit", "evaluate"])
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args(argv)
    if args.command == "evaluate" and not args.checkpoint:
        parser.error("evaluate requires --checkpoint")
    config = load_config(args.config, args.overrides)
    directory = create_run(config, f"lr-innovation-{args.command}")
    try:
        result = fit(config, directory) if args.command == "fit" else evaluate(config, args.checkpoint, directory)
        finish_run(directory, "complete")
        (directory / "run.log").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException as error:
        finish_run(directory, "failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
