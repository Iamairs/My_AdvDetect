"""Train the CIFAR-100 ResNet50 Default/Ours models for the JAD transfer study.

The paper's original classifier is WideResNet on CIFAR-10.  This script keeps
the published optimizer, epoch count, LeakyReLU slope and spectral objective
while making the requested CIFAR-100/ResNet50 adaptation explicit.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").is_file() and (p / "src/advdetect").is_dir())
sys.path.insert(0, str(ROOT / "src"))
from advdetect.datasets.cifar import stratified_partition
from advdetect.models.backbones import build_classifier


def spectral_loss(model: nn.Module) -> torch.Tensor:
    values = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            matrix = module.weight.float().flatten(1)
            sigma_min = torch.linalg.svdvals(matrix)[-1]
            values.append(torch.log(sigma_min.clamp_min(1e-8)))
    if not values:
        return torch.zeros((), device=next(model.parameters()).device)
    return torch.stack(values).sum()


def train_one(args, seed: int, device: torch.device, train_aug, train_eval, test,
              fit_indices: list[int], selection_indices: list[int]) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_classifier(
        arch="timm_resnet50", in_channels=3, num_classes=100,
        normalize_input=True, input_mean=(.485, .456, .406),
        input_std=(.229, .224, .225), activation=args.variant_activation,
    ).to(device)
    fit = Subset(train_aug, fit_indices)
    selection = Subset(train_eval, selection_indices)
    fit_loader = DataLoader(fit, batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=device.type == "cuda")
    selection_loader = DataLoader(selection, batch_size=args.batch_size, shuffle=False,
                                  num_workers=args.num_workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(test, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=device.type == "cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    best_selection, best_epoch, best_state = -1.0, 0, None
    history = []
    for epoch in range(args.epochs):
        model.train()
        total, correct, running = 0, 0, 0.0
        for images, labels in fit_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                cls = criterion(logits, labels)
            reg = spectral_loss(model) if args.variant_activation == "leaky_relu" else torch.zeros((), device=device)
            loss = cls - args.spectral_lambda * reg
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += len(labels)
            correct += logits.detach().argmax(1).eq(labels).sum().item()
            running += cls.detach().float().item() * len(labels)
        model.eval()
        selection_correct = selection_total = test_correct = test_total = 0
        with torch.no_grad():
            for images, labels in selection_loader:
                prediction = model(images.to(device, non_blocking=True)).argmax(1).cpu()
                selection_correct += prediction.eq(labels).sum().item()
                selection_total += len(labels)
            for images, labels in test_loader:
                prediction = model(images.to(device, non_blocking=True)).argmax(1).cpu()
                test_correct += prediction.eq(labels).sum().item()
                test_total += len(labels)
        selection_acc = selection_correct / selection_total
        record = {"epoch": epoch + 1, "train_loss": running / total,
                  "train_accuracy": correct / total, "selection_accuracy": selection_acc,
                  "test_accuracy": test_correct / test_total}
        history.append(record)
        print(json.dumps({"seed": seed, **record}, ensure_ascii=False), flush=True)
        if selection_acc > best_selection:
            best_selection, best_epoch = selection_acc, epoch + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("no model checkpoint produced")
    out = ROOT / args.output_root / args.variant / f"seed_{seed:03d}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ccde_classifier_v1", "model": best_state,
        "config": {"classifier": {"arch": "timm_resnet50", "num_classes": 100,
          "normalize_input": True, "input_mean": [.485, .456, .406],
          "input_std": [.229, .224, .225], "activation": args.variant_activation}},
        "training": {"seed": seed, "epochs": args.epochs, "optimizer": "Adam",
          "lr": args.lr, "spectral_lambda": args.spectral_lambda,
          "fit_size": len(fit_indices), "selection_size": len(selection_indices),
          "best_epoch": best_epoch, "history": history},
    }
    torch.save(payload, out)
    return {"seed": seed, "checkpoint": str(out), "best_epoch": best_epoch,
            "best_selection_accuracy": best_selection, "final_test_accuracy": history[-1]["test_accuracy"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["default", "ours"], required=True)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--spectral-lambda", type=float, default=1e-2)
    parser.add_argument("--output-root", default="checkpoints/jad_resnet50_cifar100")
    parser.add_argument("--skip-complete", action="store_true",
                        help="跳过已有且训练 epochs 不少于本次请求的 seed")
    args = parser.parse_args()
    args.variant_activation = "relu" if args.variant == "default" else "leaky_relu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_aug = CIFAR100(str(ROOT / "data"), train=True, download=False,
                         transform=transforms.Compose([transforms.RandomCrop(32, padding=4),
                         transforms.RandomHorizontalFlip(), transforms.Resize((args.resize, args.resize)), transforms.ToTensor()]))
    train_eval = CIFAR100(str(ROOT / "data"), train=True, download=False,
                          transform=transforms.Compose([transforms.Resize((args.resize, args.resize)), transforms.ToTensor()]))
    test = CIFAR100(str(ROOT / "data"), train=False, download=False,
                    transform=transforms.Compose([transforms.Resize((args.resize, args.resize)), transforms.ToTensor()]))
    fit_indices, remainder = stratified_partition(train_aug.targets, 40000, seed=42)
    selection_local, _ = stratified_partition([train_aug.targets[i] for i in remainder], 8000, seed=43)
    selection_indices = [remainder[i] for i in selection_local]
    summaries = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        out = ROOT / args.output_root / args.variant / f"seed_{seed:03d}.pt"
        if args.skip_complete and out.exists():
            try:
                prior = torch.load(out, map_location="cpu", weights_only=True)
                prior_epochs = int(prior.get("training", {}).get("epochs", 0))
                if prior_epochs >= args.epochs:
                    record = {"seed": seed, "checkpoint": str(out),
                              "skipped": True, "epochs": prior_epochs}
                    print(json.dumps(record, ensure_ascii=False), flush=True)
                    summaries.append(record)
                    continue
            except Exception as exc:
                print(f"无法复用 {out}: {exc}; 重新训练", flush=True)
        summaries.append(train_one(args, seed, device, train_aug, train_eval, test,
                                   fit_indices, selection_indices))
        summary = ROOT / args.output_root / args.variant / "summary.json"
        summary.write_text(json.dumps({"variant": args.variant, "runs": summaries},
                                      ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
