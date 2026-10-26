"""Train the CIFAR-100 target classifier used by the LR reproduction."""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR100

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from advdetect.models.backbones import build_classifier
from advdetect.datasets.cifar import stratified_partition
DATA = ROOT / "data"
OUT = ROOT / "checkpoints" / "classifiers" / "cifar100_timm_resnet50.pt"


def main() -> None:
    torch.manual_seed(42)
    random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
    ])
    train_aug = CIFAR100(str(DATA), train=True, download=False, transform=train_transform)
    train_eval = CIFAR100(str(DATA), train=True, download=False, transform=eval_transform)
    test = CIFAR100(str(DATA), train=False, download=False, transform=eval_transform)
    fit_indices, remainder = stratified_partition(train_aug.targets, 40000, seed=42)
    selection_local, calibration_local = stratified_partition(
        [train_aug.targets[i] for i in remainder], 8000, seed=43)
    selection_indices = [remainder[i] for i in selection_local]
    calibration_indices = [remainder[i] for i in calibration_local]
    fit = Subset(train_aug, fit_indices)
    val = Subset(train_eval, selection_indices)
    train_loader = DataLoader(fit, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    model = build_classifier(
        arch="timm_resnet50", in_channels=3, num_classes=100,
        normalize_input=True,
        input_mean=(0.485, 0.456, 0.406), input_std=(0.229, 0.224, 0.225),
    ).to(device)
    target_state = model.backbone.state_dict()
    local_weights = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth"
    if local_weights.is_file():
        pretrained_state = torch.load(local_weights, map_location="cpu", weights_only=True)
        pretrained_source = str(local_weights)
    else:
        import timm
        pretrained_model = timm.create_model("resnet50", pretrained=True, num_classes=1000)
        pretrained_state = pretrained_model.state_dict()
        pretrained_source = "timm:resnet50:pretrained"
    compatible = {k: v for k, v in pretrained_state.items()
                  if k in target_state and v.shape == target_state[k].shape}
    model.backbone.load_state_dict(compatible, strict=False)
    del pretrained_state
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    total_epochs = 8
    start_epoch = 0
    best_acc = -1.0
    best_epoch = None
    best_state = None
    if OUT.is_file():
        prior = torch.load(OUT, map_location="cpu", weights_only=True)
        if prior.get("format") == "ccde_classifier_v1" and prior.get("config", {}).get("classifier", {}).get("arch") == "timm_resnet50":
            model.load_state_dict(prior["model"], strict=True)
            start_epoch = int(prior.get("epoch", 0))
            best_epoch = start_epoch
            best_acc = float(prior.get("metrics", {}).get("best_selection_accuracy", -1.0))
            best_state = prior["model"]
            print(f"Resume classifier from epoch {start_epoch}, selection accuracy={best_acc:.4f}", flush=True)
    remaining_epochs = max(1, total_epochs - start_epoch)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    for epoch in range(start_epoch, total_epochs):
        model.train()
        total_loss = total = correct = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(labels)
            total += len(labels)
            correct += logits.detach().argmax(1).eq(labels).sum().item()
        scheduler.step()
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                pred = model(images.to(device)).argmax(1).cpu()
                val_correct += pred.eq(labels).sum().item()
                val_total += len(labels)
        val_acc = val_correct / val_total
        print(json.dumps({"epoch": epoch + 1, "train_loss": total_loss / total,
                          "train_acc": correct / total, "val_acc": val_acc}, ensure_ascii=False), flush=True)
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save({
                "format": "ccde_classifier_v1", "epoch": best_epoch,
                "model": best_state,
                "config": {"classifier": {"arch": "timm_resnet50", "num_classes": 100,
                            "normalize_input": True, "input_mean": [0.485, 0.456, 0.406],
                            "input_std": [0.229, 0.224, 0.225]}},
                "metrics": {"best_selection_accuracy": best_acc,
                            "fit_indices": fit_indices, "selection_indices": selection_indices,
                            "calibration_indices": calibration_indices},
            }, OUT)
    model.load_state_dict(best_state)
    model.eval()
    test_correct = test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            pred = model(images.to(device)).argmax(1).cpu()
            test_correct += pred.eq(labels).sum().item()
            test_total += len(labels)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "ccde_classifier_v1",
        "epoch": best_epoch,
        "model": best_state,
        "config": {"classifier": {"arch": "timm_resnet50", "num_classes": 100,
                    "normalize_input": True, "input_mean": [0.485, 0.456, 0.406],
                    "input_std": [0.229, 0.224, 0.225]}},
        "metrics": {"best_selection_accuracy": best_acc, "test_accuracy": test_correct / test_total,
                    "test_correct": test_correct, "test_total": test_total},
        "pretrained_source": pretrained_source,
    }
    torch.save(payload, OUT)
    print(json.dumps({"checkpoint": str(OUT), **payload["metrics"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
