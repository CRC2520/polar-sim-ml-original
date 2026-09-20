#!/usr/bin/env python3
"""Package closed R9 evidence into independently restorable, bounded ZIPs.

Read-only with respect to scientific evidence. Uses only the standard library.
Run after all final seeds and required reports/audits have finished:
    python r9_tools/package_evidence.py --output-dir /absolute/new/directory
No simulation, inference, upload, or source-code packaging is performed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "polar-r9-evidence-package-v1"
MAX_BYTES = 240 * 1024 * 1024
INDEX = "R9_EVIDENCE_INDEX.json"
CHECKSUMS = "R9_EVIDENCE_CHECKSUMS.sha256"
FREEZE = "r9_completion/FREEZE_R9.json"
AUTHORIZATION = "r9_results/FINAL_EXECUTION_AUTHORIZATION.json"
CHUNK = 1024 * 1024
README = (
    "POLAR R9 evidence; independently extractable ZIP, not a split volume.\n"
    "Data paths are relative to the companion simulation repository root.\n"
    "Extract every part into one clean directory while preserving these paths.\n"
    "Package metadata has unique paths under __r9_package__/ for each part.\n"
    "The index maps every file to its ZIP, size, SHA256 and CRC32.\n"
    "Source code, SOURCE_SNAPSHOT, caches and temporary files are excluded.\n"
    "Obtain source and protocols from the Git revision in the recorded receipts.\n"
    "The source freeze is stored in Git; the index records its path and SHA256.\n"
    "Packaging checks transport integrity and completion markers, not scientific\n"
    "validity or whether an experimental hypothesis was supported.\n"
).encode("utf-8")


@dataclass(frozen=True)
class Record:
    path: str
    size: int
    signature: tuple


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def signature(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Evidence must be a regular file: {path}")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def excluded(path):
    return (any(p.startswith(".") or p.lower() in {"source_snapshot", "__pycache__"}
                for p in path.parts)
            or path.suffix.lower() in {".py", ".pyc", ".pyo"}
            or path.name.lower().endswith((".tmp", ".partial", ".lock", "~")))


def inventory(root):
    root = Path(root).resolve(strict=True)
    scope = root / "r9_results"
    if not scope.is_dir() or scope.is_symlink():
        raise ValueError("A regular r9_results directory is required")
    records = []
    for directory, dirs, files in os.walk(scope, followlinks=False):
        dirs[:] = sorted(d for d in dirs
                         if not excluded((Path(directory) / d).relative_to(root)))
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ValueError(f"Symlink directory is not allowed: {name}")
        for name in sorted(files):
            full = Path(directory) / name
            relative = full.relative_to(root)
            if excluded(relative):
                continue
            stamp = signature(full)
            records.append(Record(relative.as_posix(), stamp[2], stamp))
    records.sort(key=lambda item: item.path)
    return records


def digest_stream(stream):
    digest, crc, size = hashlib.sha256(), 0, 0
    for block in iter(lambda: stream.read(CHUNK), b""):
        digest.update(block)
        crc = zlib.crc32(block, crc)
        size += len(block)
    return {"bytes": size, "sha256": digest.hexdigest(), "crc32": f"{crc & 0xffffffff:08x}"}


def digest_file(path):
    with Path(path).open("rb") as stream:
        return digest_stream(stream)


def unchanged(root, record):
    if signature(root / record.path) != record.signature:
        raise RuntimeError(f"Evidence changed after inventory: {record.path}")


def safe_relative(value):
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("Manifest paths must be relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in {".", ".."} for p in value.split("/")):
        raise ValueError(f"Unsafe or noncanonical manifest path: {value}")
    return path


def completion_contract(root, records, source_commit=None):
    """Cheap completion gate before reading gigabytes or publishing outputs."""
    freeze = json.loads((root / FREEZE).read_text())
    expected = {
        phase: set(freeze[f"{phase}_seeds"]) for phase in ("pilot", "final")
    }
    if any(not values for values in expected.values()):
        raise ValueError("Freeze must declare nonempty pilot and final seed sets")
    if any(len(values) != len(freeze[f"{phase}_seeds"]) for phase, values in expected.items()):
        raise ValueError("Duplicate seeds in freeze")
    frozen_digest = digest_file(root / FREEZE)["sha256"]
    auth = json.loads((root / AUTHORIZATION).read_text())
    if auth.get("authorized") is not True or auth.get("freeze_sha256") != frozen_digest:
        raise ValueError("Final authorization does not match the included freeze")
    recorded_commit = auth.get("source_commit", auth.get("scientific_source_commit"))
    if source_commit is not None and recorded_commit is not None and source_commit != recorded_commit:
        raise ValueError("Explicit source commit differs from execution authorization")
    scientific_commit = recorded_commit if source_commit is None else source_commit
    if not isinstance(scientific_commit, str) or not re.fullmatch(r"[0-9a-f]{40}", scientific_commit):
        raise ValueError("Authorization source_commit (or --source-commit) must name the exact Git revision")
    paths = {record.path for record in records}
    seed_dirs = set()
    for record in records:
        path = PurePosixPath(record.path)
        for parent in path.parents:
            if re.fullmatch(r"seed_\d+", parent.name):
                seed_dirs.add(parent)
    seen = {"pilot": set(), "final": set()}
    final_locations = {}
    manifests = []
    for directory in sorted(seed_dirs):
        marker = directory / "COMPLETE.json"
        if marker.as_posix() not in paths:
            raise ValueError(f"Incomplete seed directory: {directory}")
        data = json.loads((root / marker).read_text())
        meta = data.get("metadata", {})
        phase, seed = meta.get("phase"), meta.get("seed")
        if (data.get("complete") is not True or phase not in seen
                or seed not in expected[phase] or directory.name != f"seed_{seed}"):
            raise ValueError(f"Invalid completion marker: {marker}")
        # An explicitly named audit/replay copy is ancillary, not another seed.
        auxiliary = any("audit" in part.lower() or "replay" in part.lower()
                        for part in directory.parts[:-1])
        if not auxiliary:
            seen[phase].add(seed)
            if phase == "final":
                if seed in final_locations:
                    raise ValueError(f"Ambiguous duplicate final seed {seed}")
                final_locations[seed] = directory.as_posix()
        if phase == "final" and meta.get("source", {}).get("freeze_sha256") != frozen_digest:
            raise ValueError(f"Final marker references a different freeze: {marker}")
        listed = {}
        for entry in data.get("files", []):
            relative = safe_relative(entry["path"])
            full = (directory / relative).as_posix()
            if full in listed:
                raise ValueError(f"Duplicate manifest member: {full}")
            if excluded(relative):
                continue
            listed[full] = entry
        actual = {p for p in paths if p.startswith(directory.as_posix() + "/")
                  and p != marker.as_posix()}
        if not listed or actual != set(listed):
            raise ValueError(f"Completion manifest inventory differs: {directory}")
        manifests.append((marker.as_posix(), listed))
    if seen != expected:
        missing = {phase: sorted(expected[phase] - seen[phase]) for phase in seen}
        raise ValueError(f"Campaign is incomplete; missing seed sets: {missing}")
    return {"freeze_sha256": frozen_digest, "freeze_git_path": FREEZE,
            "source_commit": scientific_commit, "freeze_included_in_archives": False,
            "pilot_seeds": sorted(seen["pilot"]), "final_seeds": sorted(seen["final"]),
            "seed_manifests": len(manifests)}, manifests


def preflight(root, records, manifests):
    hashes, nested = {}, {"npz_or_zip": 0, "gzip": 0}
    for record in records:
        unchanged(root, record)
        path = root / record.path
        if path.suffix.lower() in {".npz", ".zip"}:
            with zipfile.ZipFile(path) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise zipfile.BadZipFile(f"Nested CRC failure: {record.path}: {bad}")
            nested["npz_or_zip"] += 1
        elif path.suffix.lower() == ".gz":
            with gzip.open(path, "rb") as stream:
                for _ in iter(lambda: stream.read(CHUNK), b""):
                    pass
            nested["gzip"] += 1
        hashes[record.path] = digest_file(path)
        unchanged(root, record)
    for marker, entries in manifests:
        for path, entry in entries.items():
            found = hashes[path]
            if found["sha256"] != entry["sha256"] or found["bytes"] != entry["bytes"]:
                raise ValueError(f"Recorded evidence hash/size mismatch: {marker}: {path}")
    return hashes, nested


def metadata_names(number):
    prefix = f"__r9_package__/part_{number:04d}"
    return prefix + "/MANIFEST.json", prefix + "/README.txt"


def archive_name(number):
    return f"POLAR_R9_EVIDENCE_{number:04d}.zip"


def manifest(records, number, freeze_hash, hashes=None):
    return {"schema": SCHEMA, "archive": archive_name(number),
            "path_base": "companion simulation repository root",
            "freeze_sha256": freeze_hash, "independently_extractable": True,
            "compression": "ZIP_STORED", "source_code_included": False,
            "files": [dict(path=r.path, **(hashes[r.path] if hashes else
                       {"bytes": r.size, "sha256": "0" * 64, "crc32": "0" * 8}))
                      for r in records]}


def zip_entry_bytes(name, size):
    return 76 + 2 * len(name.encode("utf-8")) + size


def estimate(records, number, freeze_hash):
    name, readme = metadata_names(number)
    payload = json_bytes(manifest(records, number, freeze_hash))
    return (22 + zip_entry_bytes(name, len(payload)) + zip_entry_bytes(readme, len(README))
            + sum(zip_entry_bytes(record.path, record.size) for record in records))


def partition(records, freeze_hash, maximum):
    if maximum < 4096 or maximum > MAX_BYTES:
        raise ValueError(f"Maximum ZIP size must be 4096..{MAX_BYTES} bytes")
    parts, current = [], []
    base = estimate([], 1, freeze_hash)
    running = base
    for record in records:
        number = len(parts) + 1
        # A singleton's metadata overhead is a conservative per-file bound.
        # This avoids repeatedly serializing a growing multi-thousand-row list.
        increment = estimate([record], number, freeze_hash) - base
        if len(current) >= 60000 or running + increment > maximum:
            if not current:
                raise ValueError(f"One whole file exceeds the ZIP limit: {record.path}")
            parts.append(current)
            current = [record]
            if estimate(current, len(parts) + 1, freeze_hash) > maximum:
                raise ValueError(f"One whole file exceeds the ZIP limit: {record.path}")
            running = base + increment
        else:
            current.append(record)
            running += increment
    if current:
        parts.append(current)
    if len(parts) > 9999:
        raise ValueError("More than 9999 independent ZIPs are unsupported")
    return parts


def zip_info(name, size):
    entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    entry.compress_type = zipfile.ZIP_STORED
    entry.create_system = 3
    entry.external_attr = 0o100644 << 16
    entry.file_size = size
    return entry


def fsync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(temporary, target):
    os.link(temporary, target)  # atomic, same filesystem, fails on existing target
    fsync_directory(target.parent)
    temporary.unlink()
    fsync_directory(target.parent)


def atomic_bytes(target, data):
    with tempfile.NamedTemporaryFile(prefix=".r9-package-", dir=target.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        publish(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def verify_archive(path, expected):
    with zipfile.ZipFile(path) as archive:
        if sorted(archive.namelist()) != sorted(expected):
            raise RuntimeError("Archive member inventory differs or contains duplicates")
        for entry in archive.infolist():
            if entry.compress_type != zipfile.ZIP_STORED:
                raise RuntimeError("Unexpected archive compression")
            with archive.open(entry) as stream:
                found = digest_stream(stream)  # ZipExtFile verifies CRC while reading.
            if found != expected[entry.filename] or f"{entry.CRC:08x}" != found["crc32"]:
                raise RuntimeError(f"Archive member CRC/SHA mismatch: {entry.filename}")


def package_evidence(root, output_dir, maximum=MAX_BYTES, source_commit=None):
    """Public function accepts a synthetic root for isolated fixture tests."""
    root = Path(root).resolve(strict=True)
    requested = Path(output_dir)
    if requested.is_symlink():
        raise ValueError("Output directory must not be a symlink")
    output = requested.resolve()
    for scope in (root / "r9_results", root / "r9_completion"):
        if output == scope or scope in output.parents:
            raise ValueError("Output must be outside evidence and frozen-source directories")
    records = inventory(root)
    closure, manifests = completion_contract(root, records, source_commit)
    parts = partition(records, closure["freeze_sha256"], maximum)
    names = [archive_name(i) for i in range(1, len(parts) + 1)] + [INDEX, CHECKSUMS]
    if any((output / name).exists() or (output / name).is_symlink() for name in names):
        raise FileExistsError("Package outputs already exist; choose a new output directory")
    hashes, nested = preflight(root, records, manifests)
    output.mkdir(parents=True, exist_ok=True)
    fsync_directory(output.parent)
    archives = []
    for number, items in enumerate(parts, 1):
        temporary = None
        metadata = {metadata_names(number)[0]: json_bytes(manifest(
            items, number, closure["freeze_sha256"], hashes)),
            metadata_names(number)[1]: README}
        expected = {item.path: hashes[item.path] for item in items}
        expected.update({name: digest_stream(io.BytesIO(data)) for name, data in metadata.items()})
        try:
            with tempfile.NamedTemporaryFile(prefix=".r9-package-", dir=output, delete=False) as stream:
                temporary = Path(stream.name)
                with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED,
                                     allowZip64=False) as archive:
                    for item in items:
                        unchanged(root, item)
                        sha, crc, size = hashlib.sha256(), 0, 0
                        with (root / item.path).open("rb") as src:
                            with archive.open(zip_info(item.path, item.size), "w") as dest:
                                for block in iter(lambda: src.read(CHUNK), b""):
                                    sha.update(block); crc = zlib.crc32(block, crc)
                                    size += len(block); dest.write(block)
                        found = {"bytes": size, "sha256": sha.hexdigest(),
                                 "crc32": f"{crc & 0xffffffff:08x}"}
                        if found != hashes[item.path]:
                            raise RuntimeError(f"Source changed while copying: {item.path}")
                        unchanged(root, item)
                    for name, data in metadata.items():
                        archive.writestr(zip_info(name, len(data)), data)
                stream.flush()
                os.fsync(stream.fileno())
            if temporary.stat().st_size != estimate(items, number, closure["freeze_sha256"]):
                raise RuntimeError("ZIP size differs from deterministic estimate")
            if temporary.stat().st_size > maximum:
                raise RuntimeError("ZIP exceeds configured size limit")
            verify_archive(temporary, expected)
            archive_digest = digest_file(temporary)
            target = output / archive_name(number)
            publish(temporary, target)
            if digest_file(target) != archive_digest:
                raise RuntimeError(f"Published ZIP changed: {target.name}")
            archives.append(dict(filename=target.name, **archive_digest,
                                 files=[dict(path=item.path, **hashes[item.path]) for item in items],
                                 metadata_files=[dict(path=name, **expected[name]) for name in metadata]))
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    if inventory(root) != records:
        raise RuntimeError("Evidence inventory changed; no complete index published")
    if digest_file(root / FREEZE)["sha256"] != closure["freeze_sha256"]:
        raise RuntimeError("External Git freeze changed; no complete index published")
    index = dict(schema=SCHEMA, complete=True, path_base="companion simulation repository root",
                 maximum_archive_bytes=maximum, compression="ZIP_STORED", closure=closure,
                 file_count=len(records), source_bytes=sum(r.size for r in records),
                 inner_container_crc_checks=nested,
                 exclusions=["*.py", "*.pyc", "*.pyo", "SOURCE_SNAPSHOT/", "__pycache__/",
                             "hidden paths", "*.tmp", "*.partial", "*.lock", "*~"],
                 scopes=["r9_results/"], external_git_records=[dict(
                     path=FREEZE, sha256=closure["freeze_sha256"], commit=closure["source_commit"])],
                 archives=archives,
                 validation="completion manifests; source size/SHA; source/ZIP CRC32; inner NPZ/ZIP/gzip CRC; "
                            "member SHA; published ZIP SHA/CRC; unchanged final inventory")
    index_bytes = json_bytes(index)
    checksums = "".join(f"{row['sha256']}  {row['filename']}\n" for row in archives)
    checksums += f"{hashlib.sha256(index_bytes).hexdigest()}  {INDEX}\n"
    atomic_bytes(output / CHECKSUMS, checksums.encode("utf-8"))
    atomic_bytes(output / INDEX, index_bytes)  # complete index is the final commit marker
    if (output / INDEX).read_bytes() != index_bytes:
        raise RuntimeError("Published index changed")
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="New output directory, outside r9_results and r9_completion")
    parser.add_argument("--source-commit", help="Exact Git revision if not present in execution authorization")
    args = parser.parse_args()
    result = package_evidence(ROOT, args.output_dir, source_commit=args.source_commit)
    print(json.dumps({"complete": True, "files": result["file_count"],
                      "archives": len(result["archives"]), "source_bytes": result["source_bytes"],
                      "maximum_archive_bytes": MAX_BYTES, "output_dir": str(args.output_dir.resolve())}))


if __name__ == "__main__":
    main()
