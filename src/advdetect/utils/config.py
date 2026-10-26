from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    override = os.environ.get("ADVDETECT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # Prefer the source checkout; installations can set ADVDETECT_ROOT.
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError("Set ADVDETECT_ROOT to the experiment workspace.")


def resolve_path(value: str | Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (project_root() / path).resolve()


def merge(base: dict, update: dict) -> dict:
    result = copy.deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    def read(current: Path, stack: tuple[Path, ...]) -> dict:
        current = current.resolve()
        if current in stack:
            raise ValueError(f"Configuration inheritance cycle: {current}")
        payload = yaml.safe_load(current.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Configuration must be a mapping: {current}")
        parents = payload.pop("extends", [])
        if isinstance(parents, str):
            parents = [parents]
        result: dict = {}
        for parent in parents:
            result = merge(result, read(current.parent / parent, (*stack, current)))
        return merge(result, payload)

    result = read(resolve_path(path), ())
    for override in overrides or []:
        key, separator, value = override.partition("=")
        if not separator or not all(key.split(".")):
            raise ValueError("Overrides use section.key=value")
        target = result
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
            if not isinstance(target, dict):
                raise ValueError(f"Cannot override non-mapping: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    return result
