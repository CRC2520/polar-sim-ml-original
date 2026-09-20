"""Create once and verify the complete prospective R8 source/design freeze."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from threadpoolctl import threadpool_info

from r8_completion.io import atomic_json, sha256
from p1_completion.provenance import verify as verify_p1
from r8_completion.confirmation import CLAIMS, conservative_power

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / 'r8_completion/FREEZE_R8.json'
BASE_COMMIT = 'd1ff3157a23e94215613afc9088b2ae57c61dfdc'
ENGINE_SHA256 = '48330c1a88a2312182bb488dae0096e2ccf674c6fe2b0ca7ddf6fa9b8a29dccc'


def source_paths(additional_files=()):
    paths = [p for p in (ROOT / 'r8_completion').rglob('*')
             if p.is_file() and p.suffix in ('.py', '.md', '.json')
             and '__pycache__' not in p.parts and p != FREEZE]
    # Include the previous immutable contract and every source in its closure.
    prior = ROOT / 'p1_completion/FREEZE_P1.json'
    paths += [prior]
    paths += [ROOT / name for name in json.loads(prior.read_text())['source_sha256']]
    paths += [p for p in (ROOT / 'collective/bridge_v2').glob('*.py')]
    for name in additional_files:
        path = Path(name)
        paths.append(path if path.is_absolute() else ROOT / path)
    return sorted(set(p.resolve() for p in paths))


def verify(path=FREEZE):
    frozen = json.loads(Path(path).read_text())
    bad = [name for name, digest in frozen['source_sha256'].items()
           if not (ROOT / name).is_file() or sha256(ROOT / name) != digest]
    if bad:
        raise ValueError('Frozen source or calibration changed: '+', '.join(bad))
    previous = verify_p1()
    return {'all_hashes_match': True, 'source_files': len(frozen['source_sha256']),
            'freeze_sha256': sha256(path), 'historical_p1_verified': previous['source_files'],
            'final_seeds': frozen['final_seeds']}


def create(additional_files=()):
    verify_p1()
    if sha256(ROOT / 'collective/bridge_v2/engine.py') != ENGINE_SHA256:
        raise ValueError('Historical engine modified')
    required = ('POPULATION_DESIGN.md', 'INTEGRATED_DESIGN.md', 'NETWORK_DESIGN.md',
                'CLAIMS_AND_CONFIRMATION.md', 'population.py', 'integrated.py',
                'learned_network.py', 'tests_population.py', 'tests_integrated.py')
    for name in required:
        if not (ROOT / 'r8_completion' / name).is_file():
            raise FileNotFoundError('Required prospective design/source missing: '+name)
    record = {
        'schema': 'polar-r8-prospective-freeze-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'base_simulation_commit': BASE_COMMIT,
        'status': 'private versioned prospective source/design freeze; not external public registration',
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in source_paths(additional_files)},
        'runtime': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                    'platform': platform.platform(),
                    'threadpools': [{k: pool.get(k) for k in
                                    ('user_api', 'internal_api', 'num_threads', 'version', 'architecture')}
                                   for pool in threadpool_info()]},
        'final_seeds': {'population': list(range(941001, 941031)),
                        'integrated': list(range(942001, 942031))},
        'development_seed_reservations': {'population': [940001,940099],
                                          'integrated': [940101,940199],
                                          'network': [940201,940299]},
        'primary_claims': list(CLAIMS),
        'inference': 'seed-level exact one-sided binomial robustness tests and Holm family six',
        'power_bound': conservative_power(),
        'historical_endpoints': 'preserved, not pooled with R8 and not counted as independent replications',
        'audit_plan': {
            'source': 'verify all frozen source/calibration hashes before and after final execution',
            'integrity': 'recheck every recorded artifact SHA256; CRC and safe load all NPZ members',
            'metrics': 'independently regenerate all six seed success vectors from saved endpoints',
            'population': 'replay first final seed representative Q and factorial cells; verify fixed-donor score pathway null',
            'integrated': 'replay first final seed ID and both delay-shift OODs from saved model and tapes',
            'controls': 'coordinate equivalence, temporal alignment and no-future-information contracts',
            'interpretation': 'retain all domains, null findings and adverse effects; no phenomenal-consciousness endpoint',
        },
    }
    atomic_json(FREEZE, record)
    return verify()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', action='store_true')
    parser.add_argument('--artifact', action='append', default=[])
    args = parser.parse_args()
    print(json.dumps(create(args.artifact) if args.create else verify(), indent=2))


if __name__ == '__main__':
    main()
