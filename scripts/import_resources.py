"""Copy existing resources into this workspace; never overwrite differing files."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="复制 CIFAR-10、ResNet18 实现和权重并校验哈希")
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = args.source.resolve()
    data_source = source / "data/cifar-10-batches-py"
    pairs = [(source / "classifiers.py", root / "src/advdetect/models/backbones.py")]
    pairs += [(source / "outputs/cifar10_smoke/classifier/checkpoints/best.pt",
               root / "checkpoints/classifiers/cifar10_resnet18.pt")]
    pairs += [(data_source / name, root / "data/cifar-10-batches-py" / name)
              for name in ["batches.meta", "test_batch", *(f"data_batch_{i}" for i in range(1, 6)), "readme.html"]]
    # Validate the entire operation before copying any files.
    hashes = {}
    for src, dst in pairs:
        if not src.is_file():
            raise FileNotFoundError(src)
        hashes[src] = digest(src)
        if dst.exists() and digest(dst) != hashes[src]:
            raise FileExistsError(f"Destination differs; refusing overwrite: {dst}")
    records = []
    for src, dst in pairs:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        if digest(dst) != hashes[src]:
            raise IOError(f"Checksum mismatch: {dst}")
        records.append({"source": str(src), "destination": dst.relative_to(root).as_posix(),
                        "sha256": hashes[src], "bytes": dst.stat().st_size})
    manifest = root / "docs/resource_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"imported_at_utc": datetime.now(timezone.utc).isoformat(),
                                   "files": records}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Verified {len(records)} copied resources. Manifest: {manifest}")


if __name__ == "__main__":
    main()
