#!/usr/bin/env python3
"""Deterministic, bounded, independently extractable R8 evidence archives.

This packages existing records only. It does not inspect scientific outcomes,
run experiments, upload files, move sources, or bundle source code.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import zipfile


SCHEMA = "polar-r8-evidence-package-v1"
MAX_BYTES = 240_000_000  # Decimal MB; deliberately below 240 MiB.
MANIFEST_NAME = "R8_ARCHIVE_MANIFEST.json"
README_NAME = "R8_ARCHIVE_README.txt"
README = (
    "POLAR R8 evidence archive\n"
    "This ZIP is independently extractable; it is not a split ZIP volume.\n"
    "Data paths are relative to the POLAR_reconciled repository root.\n"
    "Extract the data paths into a clean evidence directory while preserving paths.\n"
    "R8_ARCHIVE_MANIFEST.json describes only this archive and can be retained separately.\n"
    "Other parts contain different records, not pieces required to open this ZIP.\n"
    "Versions, development records and adverse results are preserved as recorded.\n"
    "Python source, code snapshots and r8_completion/ are excluded intentionally;\n"
    "obtain the published source/protocols from the matching Git revision.\n"
    "The master index records every archive SHA-256. Inner NPZ/ZIP CRCs are checked\n"
    "before packaging; source SHA-256 is checked before and during the copy; all\n"
    "archive members are fully reread to verify CRC and content after writing.\n"
).encode("utf-8")


@dataclass(frozen=True)
class Record:
    path: str
    size: int
    signature: tuple[int, int, int, int]
    category: str
    seed: str | None


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _signature(path):
    s = path.lstat()
    if not stat.S_ISREG(s.st_mode):
        raise ValueError(f"Only regular evidence files are allowed: {path}")
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)


def _excluded(path):
    return ("__pycache__" in path.parts or path.suffix.lower() in {".py", ".pyc"}
            or path.name.startswith(".") or path.name.endswith((".tmp", ".partial")))


def _category(relative):
    parts = relative.parts
    low = [p.lower() for p in parts]
    seed = next((p for p in parts if re.fullmatch(r"seed_\d+", p)), None)
    if "integrated_final" in low:
        return ("integrated_final" if seed else "summary", seed)
    if any("audit" in p for p in low):
        return ("audits", seed)
    if "analysis_final" in low:
        return ("summary", seed)
    if any("pilot" in p or "development" in p for p in low):
        return ("development", seed)
    if any(re.fullmatch(r"(?:final_(?:q|a|thresholds?)|(?:q|a|thresholds?)_final)", p)
           for p in low):
        return ("population_final", seed)
    if len(parts) == 2 and parts[0] == "r8_results":
        return ("summary", seed)
    return ("other_r8", seed)


def scope_roots(source):
    """Explicit allowlist: never discover P1/P2 or similarly named old work."""
    roots = []
    if (source / "r8_results").exists():
        roots.append(source / "r8_results")
    results = source / "results"
    if results.exists():
        roots.extend(p for p in sorted(results.iterdir())
                     if p.name.lower().startswith("r8_") and p.is_dir())
    for root in roots:
        if root.is_symlink():
            raise ValueError(f"Symlink scope is not allowed: {root}")
    return roots


def inventory(source):
    """Names and stat metadata only; no scientific record contents are read."""
    source = Path(source).resolve(strict=True)
    records = []
    for root in scope_roots(source):
        for directory, dirs, files in os.walk(root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d != "__pycache__" and not d.startswith("."))
            for d in dirs:
                if (Path(directory) / d).is_symlink():
                    raise ValueError(f"Symlink directory is not allowed: {Path(directory) / d}")
            for name in sorted(files):
                full = Path(directory) / name
                rel = full.relative_to(source)
                if _excluded(rel):
                    continue
                signature = _signature(full)
                category, seed = _category(rel)
                records.append(Record(rel.as_posix(), signature[2], signature, category, seed))
    records.sort(key=lambda r: r.path)
    if not records:
        raise ValueError("No eligible R8 evidence files found")
    return records


def _manifest(records, category, filename, hashes=None, source_commit=None):
    return {"schema": SCHEMA, "archive": filename, "category": category,
            "path_base": "POLAR_reconciled repository root", "source_commit": source_commit,
            "compression": "ZIP_STORED", "independently_extractable": True,
            "code_included": False,
            "members": [{"path": r.path, "bytes": r.size,
                         "sha256": hashes[r.path] if hashes is not None else "0" * 64}
                        for r in records]}


def _zip_entry_bytes(name, size):
    # We set known sizes, use no descriptors, extras, comments or ZIP64.
    return size + 76 + 2 * len(name.encode("utf-8"))


def estimate_zip_bytes(records, category, source_commit=None):
    filename = f"POLAR_R8_{category.upper()}_0001_of_0001.zip"
    manifest = _json_bytes(_manifest(records, category, filename, source_commit=source_commit))
    return (22 + _zip_entry_bytes(MANIFEST_NAME, len(manifest))
            + _zip_entry_bytes(README_NAME, len(README))
            + sum(_zip_entry_bytes(r.path, r.size) for r in records))


def partition(records, max_bytes=MAX_BYTES, seeds_per_part=2, source_commit=None):
    if max_bytes < 4096 or max_bytes > MAX_BYTES:
        raise ValueError(f"max_bytes must be 4096..{MAX_BYTES}")
    if seeds_per_part < 1:
        raise ValueError("seeds_per_part must be positive")
    categories = defaultdict(list)
    for r in records:
        categories[r.category].append(r)
    parts = []
    for category in sorted(categories):
        items = categories[category]
        if category == "integrated_final":
            seeds = sorted({r.seed for r in items})
            batches = [seeds[i:i + seeds_per_part] for i in range(0, len(seeds), seeds_per_part)]
            groups = [[r for r in items if r.seed in batch] for batch in batches]
            # Avoid a tiny overflow part for every two-seed group. If any pair
            # does not fit, cut the full sorted category deterministically by
            # file instead; each archive still opens and verifies by itself.
            if any(estimate_zip_bytes(group, category, source_commit) > max_bytes for group in groups):
                groups = [items]
        else:
            groups = [items]
        chunks = []
        for group in groups:
            current = []
            for r in sorted(group, key=lambda item: item.path):
                proposed = current + [r]
                if len(proposed) + 2 >= 65535 or estimate_zip_bytes(proposed, category, source_commit) > max_bytes:
                    if not current:
                        raise ValueError(f"A single file cannot fit in an autonomous bounded ZIP: {r.path}")
                    chunks.append(current)
                    current = [r]
                    if estimate_zip_bytes(current, category, source_commit) > max_bytes:
                        raise ValueError(f"A single file cannot fit in an autonomous bounded ZIP: {r.path}")
                else:
                    current = proposed
            if current:
                chunks.append(current)
        if len(chunks) > 9999:
            raise ValueError("Too many parts for the deterministic four-digit archive scheme")
        for n, chunk in enumerate(chunks, 1):
            parts.append({"filename": f"POLAR_R8_{category.upper()}_{n:04d}_of_{len(chunks):04d}.zip",
                          "category": category, "records": chunk,
                          "estimated_bytes": estimate_zip_bytes(chunk, category, source_commit)})
    return parts


def _sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def _assert_unchanged(source, record):
    if _signature(source / record.path) != record.signature:
        raise RuntimeError(f"Source changed after inventory: {record.path}")


def preflight(source, records):
    hashes = {}
    nested_zip_count = 0
    for r in records:
        full = source / r.path
        _assert_unchanged(source, r)
        if full.suffix.lower() in {".npz", ".zip"}:
            with zipfile.ZipFile(full) as z:
                bad = z.testzip()
                if bad is not None:
                    raise zipfile.BadZipFile(f"Nested CRC failure in {r.path}: {bad}")
            nested_zip_count += 1
        hashes[r.path] = _sha256(full)
        _assert_unchanged(source, r)
    return hashes, nested_zip_count


def _info(name, size):
    info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.file_size = size
    return info


def _copy_member(z, source, r, expected):
    _assert_unchanged(source, r)
    h = hashlib.sha256()
    with (source / r.path).open("rb") as src, z.open(_info(r.path, r.size), "w") as dest:
        for data in iter(lambda: src.read(1024 * 1024), b""):
            h.update(data)
            dest.write(data)
    if h.hexdigest() != expected:
        raise RuntimeError(f"Source SHA-256 changed during packaging: {r.path}")
    _assert_unchanged(source, r)


def _verify_archive(path, records, hashes, manifest_bytes):
    expected = {r.path: hashes[r.path] for r in records}
    expected[MANIFEST_NAME] = hashlib.sha256(manifest_bytes).hexdigest()
    expected[README_NAME] = hashlib.sha256(README).hexdigest()
    with zipfile.ZipFile(path) as z:
        if sorted(z.namelist()) != sorted(expected):
            raise RuntimeError(f"Archive contents or duplicate members differ: {path.name}")
        for info in z.infolist():
            h = hashlib.sha256()
            if info.compress_type != zipfile.ZIP_STORED:
                raise RuntimeError("Unexpected ZIP compression")
            with z.open(info) as f:
                # Reading to EOF verifies this member's CRC in ZipExtFile.
                for data in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(data)
            if h.hexdigest() != expected[info.filename]:
                raise RuntimeError(f"Archive SHA mismatch: {info.filename}")


def _fsync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _publish_no_overwrite(temporary, target):
    # A hard link creates the destination atomically and fails if it exists.
    os.link(temporary, target)
    _fsync_directory(target.parent)
    temporary.unlink()
    _fsync_directory(target.parent)


def _atomic_bytes(target, data):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".r8-package-", dir=target.parent, delete=False) as f:
            temporary = Path(f.name)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _publish_no_overwrite(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def package(source, output, max_bytes=MAX_BYTES, seeds_per_part=2, source_commit=None):
    source = Path(source).resolve(strict=True)
    output = Path(output)
    if not output.is_absolute():
        raise ValueError("--output must be an absolute path")
    output = output.resolve()
    for root in scope_roots(source):
        if output == root or root in output.parents:
            raise ValueError("Output must be outside the evidence scopes to prevent recursive inclusion")
    records = inventory(source)
    parts = partition(records, max_bytes, seeds_per_part, source_commit)
    targets = [output / p["filename"] for p in parts]
    targets += [output / "R8_EVIDENCE_INDEX.json", output / "R8_EVIDENCE_ARCHIVES.sha256"]
    if any(p.exists() or p.is_symlink() for p in targets):
        raise FileExistsError("One or more package outputs already exist; choose a new output directory")
    # All input CRC/hash checks precede publication of any archive.
    hashes, nested_count = preflight(source, records)
    output.mkdir(parents=True, exist_ok=True)
    _fsync_directory(output.parent)
    archive_rows = []
    for part in parts:
        temporary = None
        data = _json_bytes(_manifest(part["records"], part["category"], part["filename"], hashes, source_commit))
        try:
            with tempfile.NamedTemporaryFile(prefix=".r8-package-", dir=output, delete=False) as f:
                temporary = Path(f.name)
                with zipfile.ZipFile(f, "w", compression=zipfile.ZIP_STORED, allowZip64=False) as z:
                    for r in part["records"]:
                        _copy_member(z, source, r, hashes[r.path])
                    z.writestr(_info(MANIFEST_NAME, len(data)), data)
                    z.writestr(_info(README_NAME, len(README)), README)
                f.flush()
                os.fsync(f.fileno())
            size = temporary.stat().st_size
            if size > max_bytes or size != part["estimated_bytes"]:
                raise RuntimeError(f"Archive size estimate/limit failure: {part['filename']} ({size})")
            _verify_archive(temporary, part["records"], hashes, data)
            digest = _sha256(temporary)
            target = output / part["filename"]
            _publish_no_overwrite(temporary, target)
            # Check published bytes too; the index is written only on complete success.
            if _sha256(target) != digest:
                raise RuntimeError(f"Published archive hash mismatch: {target.name}")
            archive_rows.append({"filename": target.name, "bytes": size, "sha256": digest,
                                 "category": part["category"], "file_count": len(part["records"]),
                                 "seeds": sorted({r.seed for r in part["records"] if r.seed}),
                                 "members": _manifest(part["records"], part["category"], part["filename"], hashes)["members"]})
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    if inventory(source) != records:
        raise RuntimeError("Evidence inventory changed during packaging; no complete index was published")
    index = {"schema": SCHEMA, "complete": True, "source_commit": source_commit,
             "path_base": "POLAR_reconciled repository root", "maximum_archive_bytes": max_bytes,
             "preferred_integrated_seeds_per_part": seeds_per_part,
             "scopes": [p.relative_to(source).as_posix() for p in scope_roots(source)],
             "exclusions": ["r8_completion/", "*.py", "*.pyc", "__pycache__/", "hidden files/directories", "*.tmp", "*.partial"],
             "file_count": len(records), "source_bytes": sum(r.size for r in records),
             "nested_npz_zip_crc_checked": nested_count,
             "validation": "source SHA before/during copy; nested NPZ/ZIP CRC; every archive member CRC and SHA; published ZIP SHA; unchanged final inventory",
             "archives": archive_rows}
    lines = "".join(f"{row['sha256']}  {row['filename']}\n" for row in archive_rows)
    _atomic_bytes(output / "R8_EVIDENCE_ARCHIVES.sha256", lines.encode("utf-8"))
    _atomic_bytes(output / "R8_EVIDENCE_INDEX.json", _json_bytes(index))
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Absolute output path, outside evidence scopes")
    parser.add_argument("--max-bytes", type=int, default=MAX_BYTES)
    parser.add_argument("--seeds-per-part", type=int, default=2)
    parser.add_argument("--source-commit", help="Optional exact published Git source revision")
    parser.add_argument("--inventory-only", action="store_true", help="Stat-only estimate; no content reads or writes")
    args = parser.parse_args()
    if args.inventory_only:
        records = inventory(args.source)
        parts = partition(records, args.max_bytes, args.seeds_per_part, args.source_commit)
        print(json.dumps({"inventory_only": True, "provisional_if_runs_active": True,
                          "files": len(records), "source_bytes": sum(r.size for r in records),
                          "archives": len(parts), "archive_categories": dict(Counter(p["category"] for p in parts)),
                          "estimated_archive_bytes": sum(p["estimated_bytes"] for p in parts),
                          "parts": [{k: v for k, v in p.items() if k != "records"} for p in parts]}, indent=2))
    else:
        if args.output is None:
            parser.error("--output is required unless --inventory-only is used")
        index = package(args.source, args.output, args.max_bytes, args.seeds_per_part, args.source_commit)
        print(json.dumps({"complete": True, "files": index["file_count"],
                          "archives": len(index["archives"]), "source_bytes": index["source_bytes"],
                          "output": str(args.output)}))


if __name__ == "__main__":
    main()
