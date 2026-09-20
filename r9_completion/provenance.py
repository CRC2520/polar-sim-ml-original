"""Immutable prospective R9 source closure and final execution guard."""
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
from r8_completion.provenance import verify as verify_r8
from r9_completion.config import FINAL_SEEDS, PILOT_SEEDS, PROTOCOL

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT/'r9_completion/FREEZE_R9.json'
BASE_COMMIT = '6743addb639fe77e0e4439ffc89cd4a58e8e07be'


def source_paths():
    files = [p for p in (ROOT/'r9_completion').rglob('*')
             if p.is_file() and p.suffix in ('.py', '.md', '.json')
             and '__pycache__' not in p.parts and p != FREEZE]
    old = ROOT/'r8_completion/FREEZE_R8.json'
    files += [old, ROOT/'r8_completion/provenance.py']
    files += [ROOT/p for p in json.loads(old.read_text())['source_sha256']]
    return sorted(set(files))


def verify(path=FREEZE):
    record = json.loads(Path(path).read_text())
    bad = [p for p, h in record['source_sha256'].items()
           if not (ROOT/p).is_file() or sha256(ROOT/p) != h]
    if bad:
        raise ValueError('Frozen R9 source drift: '+', '.join(bad))
    old = verify_r8()
    return dict(all_hashes_match=True, source_files=len(record['source_sha256']),
                freeze_sha256=sha256(path), historical_r8_verified=old['source_files'])


def create():
    old = verify_r8()
    required = ('agent.py', 'runner.py', 'network.py', 'workspace.py',
                'environments.py', 'statistics.py', 'PREREG_R9.md', 'DESIGN_R9.md')
    for name in required:
        if not (ROOT/'r9_completion'/name).is_file():
            raise FileNotFoundError(name)
    record = dict(schema='polar-r9-freeze-v1',
                  created_at_utc=datetime.now(timezone.utc).isoformat(),
                  base_simulation_commit=BASE_COMMIT,
                  status='private prospective source freeze, not external registration',
                  source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in source_paths()},
                  final_seeds=list(FINAL_SEEDS), pilot_seeds=list(PILOT_SEEDS),
                  protocol=PROTOCOL, historical_r8=old,
                  runtime=dict(python=platform.python_version(), numpy=np.__version__,
                               scipy=scipy.__version__, platform=platform.platform(),
                               threadpools=threadpool_info()),
                  scope='new bounded realization; not a numerical continuation or replication of R8',
                  transfer='separate internal-team generators, no external laboratory replication')
    atomic_json(FREEZE, record)
    return verify()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', action='store_true')
    args = parser.parse_args()
    print(json.dumps(create() if args.create else verify(), indent=2))


if __name__ == '__main__':
    main()
