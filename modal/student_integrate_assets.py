"""Preserve installed listener auxiliaries when replacing a primary acoustic model."""

import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_auxiliaries(base, out):
    base, out = Path(base), Path(out)
    manifest = json.loads((base / "manifest.json").read_text())
    if (base / "teacher.onnx").is_file() and not (base / "teacher.onnx.data").is_file():
        raise ValueError("The installed insertion teacher is missing its declared external weights")
    excluded = {Path(manifest["file"]), Path("model.onnx"), Path("manifest.json"), Path("parity.json")}
    copied = {}
    for source in sorted(base.rglob("*")):
        rel = source.relative_to(base)
        if rel in excluded or not source.is_file():
            continue
        if source.is_symlink():
            raise ValueError(f"Auxiliary asset is not a regular file: {rel}")
        destination = out / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        expected = digest(source)
        shutil.copyfile(source, destination)
        if digest(destination) != expected:
            raise ValueError(f"Auxiliary copy hash mismatch: {rel}")
        copied[str(rel)] = dict(bytes=destination.stat().st_size, sha256=expected)
    return copied
