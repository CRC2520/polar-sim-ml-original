"""Atomic, immutable result publication with explicit integrity checks.

These helpers never draw from an experiment's random generator. A complete
manifest is written only after all its referenced files have been rechecked.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import zipfile

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _atomic_bytes(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evidence: {path}")
    expected = hashlib.sha256(payload).hexdigest()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        # Link is atomic and refuses replacement if another writer wins the race.
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    actual = sha256(path)
    if actual != expected:
        raise IOError(f"Post-publication SHA256 mismatch: {path}")
    return {"path": str(path), "bytes": len(payload), "sha256": actual}


def atomic_text(path, value):
    return _atomic_bytes(path, str(value).encode("utf-8"))


def atomic_json(path, value):
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                         allow_nan=False, default=_json_default) + "\n"
    return atomic_text(path, payload)


def verify_npz(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise IOError(f"Invalid CRC: {path}:{bad}")
        if len(set(archive.namelist())) != len(archive.namelist()):
            raise IOError(f"Duplicate NPZ members: {path}")
    shapes = {}
    with np.load(path, allow_pickle=False) as arrays:
        for key in arrays.files:
            value = arrays[key]
            if value.dtype.hasobject:
                raise TypeError(f"Object array forbidden: {path}:{key}")
            shapes[key] = {"shape": list(value.shape), "dtype": str(value.dtype)}
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": sha256(path), "arrays": shapes}


def atomic_npz(path, arrays=None, **named_arrays):
    values = {} if arrays is None else dict(arrays)
    overlap = set(values).intersection(named_arrays)
    if overlap:
        raise ValueError(f"Duplicate array keys: {sorted(overlap)}")
    values.update(named_arrays)
    values = {key: np.asarray(value) for key, value in values.items()}
    for key, value in values.items():
        if value.dtype.hasobject:
            raise TypeError(f"Object array forbidden: {key}")
        if not isinstance(key, str) or not key or "/" in key or "\\" in key:
            raise ValueError(f"Unsafe NPZ array key: {key!r}")
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    payload = stream.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise IOError(f"Invalid in-memory CRC: {bad}")
    _atomic_bytes(path, payload)
    return verify_npz(path)


def write_manifest(path, files, metadata=None):
    path = Path(path)
    records = []
    for entry in files:
        item = Path(entry["path"] if isinstance(entry, dict) else entry)
        record = verify_npz(item) if item.suffix == ".npz" else {
            "path": str(item), "bytes": item.stat().st_size, "sha256": sha256(item)}
        if isinstance(entry, dict) and entry.get("sha256") not in (None, record["sha256"]):
            raise IOError(f"Changed after original publication: {item}")
        record["path"] = os.path.relpath(item, path.parent)
        records.append(record)
    return atomic_json(path, {"schema": "polar-r8-artifacts-v1", "complete": True,
                              "metadata": metadata or {}, "files": records})
