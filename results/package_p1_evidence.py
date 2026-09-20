"""Package immutable P1 observations into standalone bounded-size ZIP archives.

Infrastructure only: no scientific results are calculated or altered here.
Run after all front-specific final verification has completed.
"""
from pathlib import Path
import hashlib
import json
import os
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "delivery/P1_20260920"
LIMIT = 230_000_000
FRONTS = {
    "A": ROOT / "p1_results/final_A",
    "C": ROOT / "p1_results/final_C",
    "E": ROOT / "results/p1_ecology_final_20260920",
    "T": ROOT / "results/p1_tension_final_20260920",
    "RECUPERACIONES": [ROOT / "p1_results/RECOVERY_A",
                       ROOT / "p1_results/RECOVERY_C",
                       ROOT / "results/p1_tension_recovered_20260920"],
    "PILOTOS_A_C": [ROOT / "p1_results/pilot_A", ROOT / "p1_results/pilot_C"],
    "PILOTOS_E_T": [ROOT / "results/p1_ecology_pilot_20260920",
                    ROOT / "results/p1_tension_pilot_20260920",
                    ROOT / "results/p1_tension_pilot_v2_20260920"],
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def write_archive(path, files):
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".zip.partial")
    with zipfile.ZipFile(temporary, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for item in files:
            archive.write(item, str(item.relative_to(ROOT)),
                          compress_type=zipfile.ZIP_STORED if item.suffix == ".npz" else zipfile.ZIP_DEFLATED)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Archive checksum failed: {bad}")
    return {"filename": path.name, "bytes": path.stat().st_size,
            "sha256": sha(path), "file_count": len(files),
            "members": [str(p.relative_to(ROOT)) for p in files]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    catalog = {"source_commit": "2c63f3df6365751af1331eb75cfd16882f88cc44",
               "format": "Independent ZIP files, not split-volume ZIP. Extract all in the same folder; paths do not collide.",
               "artifact_incidents": "Original damaged containers are deliberately retained. Use recovered copies as documented in recovery reports; T four files and C seven files match original SHA256, A diagnostic trace is reconstructed against saved transition hashes/outcomes.",
               "all_observation_files": {}, "archives": []}
    summaries = []
    for name, roots in FRONTS.items():
        roots = roots if isinstance(roots, list) else [roots]
        files = sorted(p for root in roots for p in root.rglob("*")
                       if p.is_file() and p.suffix != ".pyc")
        if not files:
            raise FileNotFoundError(name)
        for p in files:
            catalog["all_observation_files"][str(p.relative_to(ROOT))] = {
                "bytes": p.stat().st_size, "sha256": sha(p)}
            if p.suffix in (".json", ".md", ".csv") and "PILOTOS" not in name:
                summaries.append(p)
        parts, group, size = [], [], 0
        for p in files:
            if group and size + p.stat().st_size > LIMIT:
                parts.append(group)
                group, size = [], 0
            group.append(p)
            size += p.stat().st_size
        if group:
            parts.append(group)
        for index, group in enumerate(parts, 1):
            archive = OUT / f"POLAR_P1_{name}_Datos_{index:02d}_{len(parts):02d}_20260920.zip"
            metadata = write_archive(archive, group)
            metadata["front"] = name
            catalog["archives"].append(metadata)
            print(json.dumps({k: v for k, v in metadata.items() if k != "members"}), flush=True)
    metadata_files = [ROOT / "p1_results/SOURCE_COMMIT.json", ROOT / "p1_completion/FREEZE_P1.json"]
    metadata_files += sorted(p for p in (ROOT / "p1_results").glob("*")
                             if p.is_file() and p.suffix in (".json", ".md", ".txt"))
    summaries = sorted(set(summaries + metadata_files))
    metadata = write_archive(OUT / "POLAR_P1_Resumen_Verificaciones_20260920.zip", summaries)
    metadata["front"] = "SUMMARY"
    catalog["archives"].append(metadata)
    index_path = OUT / "POLAR_P1_Indice_Evidencia_20260920.json"
    index_path.write_text(json.dumps(catalog, indent=2, ensure_ascii=False) + "\n")
    checksums = "".join(f"{r['sha256']}  {r['filename']}\n" for r in catalog["archives"])
    checksums += f"{sha(index_path)}  {index_path.name}\n"
    (OUT / "POLAR_P1_Evidencia_20260920.sha256").write_text(checksums)


if __name__ == "__main__":
    main()
