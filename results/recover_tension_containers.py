"""Forensic deterministic recovery, without overwriting final evidence.

Four NPZ containers failed their original post-write SHA256 and have no ZIP
central directory. Regenerate frozen seed arrays, serialize in memory, require
the ORIGINAL manifest's byte hash, and check whether damaged bytes are an exact
prefix. Only hash-identical candidates are written with an atomic rename.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from p1_completion import tension as t

ORIGINAL = ROOT/'results/p1_tension_final_20260920'
RECOVERED = ROOT/'results/p1_tension_recovered_20260920'
MIRROR = ROOT.parent/'P1_tension_verification_mirror'
SEEDS = (934002, 934003, 934004, 934010)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def raw_seed(seed):
    raw, pulses, tasks = {}, [], []
    for source in range(8):
        for pole in range(2):
            for sigma in (0., .02):
                for variant in t.VARIANTS+('tau_clamp',):
                    data = t.pulse_pair(seed, source, pole, sigma, variant)
                    prefix = f'p{source}_{pole}_n{sigma}_{variant}'
                    raw.update({prefix+'__'+k: v for k, v in data.items()})
                    pulses.append(dict(seed=seed, source=source, pole=pole, sigma=sigma,
                                       variant=variant, **t.pulse_metrics(data, source)))
    for task in ('switching_memory', 'gain_resource_shift'):
        for variant in t.TASK_VARIANTS:
            data = t.task_trial(seed, task, variant)
            raw.update({f'task_{task}_{variant}__'+k: v for k, v in data.items()})
            tasks.append(dict(seed=seed, task=task, variant=variant, **t.task_metrics(data)))
    return raw, pulses, tasks


def main():
    manifest_bytes = (ORIGINAL/'MANIFEST.json').read_bytes()
    manifest = json.loads(manifest_bytes)
    assert t.source_hashes() == manifest['source_sha256']
    if RECOVERED.exists() or MIRROR.exists():
        raise FileExistsError('Recovery and mirror output must be new')
    (RECOVERED/'raw').mkdir(parents=True)
    snapshot = {'original_manifest_sha256': digest(manifest_bytes),
                'original_manifest': manifest, 'source_unchanged': True,
                'cause': 'undetermined; observed ZIP truncation, no causal attribution to platform',
                'affected_seeds': SEEDS, 'records': []}
    for seed in SEEDS:
        path = ORIGINAL/'raw'/f'seed_{seed}.npz'
        damaged = path.read_bytes()
        stat = path.stat()
        try:
            with zipfile.ZipFile(path) as z:
                crc = z.testzip()
            zip_status = {'valid': True, 'crc_bad_member': crc}
        except zipfile.BadZipFile as exc:
            zip_status = {'valid': False, 'error': str(exc)}
        record = {'seed': seed, 'original_path': str(path),
                  'original_container_sha256': digest(damaged),
                  'expected_manifest_sha256': manifest['files'][f'raw/seed_{seed}.npz'],
                  'damaged_bytes': len(damaged), 'mtime_ns': stat.st_mtime_ns,
                  'ctime_ns': stat.st_ctime_ns, 'inode': stat.st_ino,
                  'zip_status': zip_status, 'eocd_offset': damaged.rfind(b'PK\x05\x06')}
        snapshot['records'].append(record)
    (RECOVERED/'FORENSICS_INITIAL.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    old_pulses = json.loads((ORIGINAL/'pulse_rows.json').read_text())
    old_tasks = json.loads((ORIGINAL/'task_rows.json').read_text())
    for record in snapshot['records']:
        seed = record['seed']
        raw, pulses, tasks = raw_seed(seed)
        assert t.canonical(pulses) == t.canonical([x for x in old_pulses if x['seed'] == seed])
        assert t.canonical(tasks) == t.canonical([x for x in old_tasks if x['seed'] == seed])
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **raw)
        candidate = buffer.getvalue()
        candidate_hash = digest(candidate)
        record['reconstructed_bytes'] = len(candidate)
        record['reconstructed_sha256'] = candidate_hash
        record['matches_original_manifest_byte_exactly'] = candidate_hash == record['expected_manifest_sha256']
        damaged = Path(record['original_path']).read_bytes()
        record['damaged_is_exact_prefix'] = candidate[:len(damaged)] == damaged
        record['scientific_metric_rows_exact'] = True
        if not record['matches_original_manifest_byte_exactly']:
            (RECOVERED/'FORENSICS_FAILED.json').write_text(json.dumps(snapshot, indent=2)+'\n')
            raise AssertionError('Frozen replay does not match original container hash')
        with np.load(io.BytesIO(candidate), allow_pickle=False) as loaded:
            assert set(loaded.files) == set(raw)
            for key, value in raw.items():
                assert np.array_equal(value, loaded[key]), key
        record['reconstructed_arrays_verified'] = len(raw)
        target = RECOVERED/'raw'/f'seed_{seed}.npz'
        temp = target.with_suffix('.npz.partial')
        with temp.open('xb') as f:
            f.write(candidate)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, target)
        assert digest(target.read_bytes()) == record['expected_manifest_sha256']
        record['recovered_path'] = str(target)
        assert digest(Path(record['original_path']).read_bytes()) == record['original_container_sha256']
        print(f'recovered seed {seed}: original SHA match; prefix={record["damaged_is_exact_prefix"]}', flush=True)
    snapshot['source_unchanged_after_replay'] = t.source_hashes() == manifest['source_sha256']
    snapshot['original_manifest_unchanged'] = digest((ORIGINAL/'MANIFEST.json').read_bytes()) == digest(manifest_bytes)
    (MIRROR/'raw').mkdir(parents=True)
    resolved = {}
    for seed in manifest['seeds']:
        source = (RECOVERED if seed in SEEDS else ORIGINAL)/'raw'/f'seed_{seed}.npz'
        dest = MIRROR/'raw'/source.name
        dest.symlink_to(source)
        resolved[dest.name] = str(source)
    for source in ORIGINAL.iterdir():
        if source.is_file():
            shutil.copy2(source, MIRROR/source.name)
    snapshot['logical_dataset_paths'] = resolved
    snapshot['logical_verification_mirror'] = str(MIRROR)
    snapshot['no_original_files_overwritten'] = True
    snapshot['additional_independent_replicates'] = 0
    (RECOVERED/'FORENSIC_RECOVERY.json').write_text(json.dumps(snapshot, indent=2)+'\n')
    print(json.dumps({'recovered_seeds': SEEDS, 'all_original_hashes_match': True,
                      'source_unchanged': snapshot['source_unchanged_after_replay'],
                      'mirror': str(MIRROR)}, indent=2))


if __name__ == '__main__':
    main()
