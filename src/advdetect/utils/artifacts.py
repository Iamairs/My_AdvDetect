from __future__ import annotations

import hashlib
import json
import platform
import random
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from datetime import timezone, timedelta

import numpy as np
import torch
import torchvision
import yaml

from .config import project_root, resolve_path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --set device=cpu")
    return device


def create_run(config: dict, command: str) -> Path:
    try:
        tz = ZoneInfo("Asia/Shanghai")
    except ZoneInfoNotFoundError:
        tz = timezone(timedelta(hours=8))
    run_id = f"{datetime.now(tz):%Y%m%d_%H%M%S}_{command}_{uuid.uuid4().hex[:6]}"
    directory = resolve_path(config.get("output_root", "outputs")) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    source_files = sorted((project_root() / "src").rglob("*.py"))
    source_files += sorted((project_root() / "research").rglob("*.py"))
    source_files += [project_root() / "run.py"]
    hashes = {str(p.relative_to(project_root())): sha256_file(p) for p in source_files if p.is_file()}
    git = {}
    for key, args in {"commit": ["rev-parse", "HEAD"], "status": ["status", "--porcelain"]}.items():
        try:
            p = subprocess.run(["git", *args], cwd=project_root(), capture_output=True, text=True, encoding="utf-8", errors="replace")
            git[key] = p.stdout.strip() if p.returncode == 0 else None
        except FileNotFoundError:
            git[key] = None
    save_json(directory / "metadata.json", {
        "run_id": run_id, "command": command, "status": "running",
        "started_at": datetime.now(tz).isoformat(), "python": sys.version,
        "executable": sys.executable, "platform": platform.platform(),
        "torch": torch.__version__, "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config_sha256": stable_hash(config), "source_files": hashes,
        "source_sha256": stable_hash(hashes), "git": git,
    })
    return directory


def finish_run(directory: Path, status: str, **extra) -> None:
    path = directory / "metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    metadata.update(status=status, **extra)
    save_json(path, metadata)
