"""Post-freeze scheduling utility for the immutable R9 seed auditor.

Scientific checks remain in r9_completion.audit. This module only parallelizes
complete independent seed audits and repeats campaign/source/authorization
guards around them. It never executes a simulation or changes evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
import multiprocessing
from pathlib import Path

from threadpoolctl import threadpool_limits

from r8_completion.io import atomic_json
from r8_completion.provenance import verify as verify_r8
from r9_completion.audit import (audit_campaign, audit_pilot_snapshot, audit_seed,
    close, independent_inference, require, sha256)
from r9_completion.config import FINAL_SEEDS, PILOT_SEEDS
from r9_completion.provenance import FREEZE, ROOT, verify
from r9_completion.statistics import DEFAULT_PLAN, compile_from_metrics


def _worker(arguments):
    directory, seed, phase, frozen, replay_seed = arguments
    # The limit also applies when a caller imports this module after NumPy has
    # already initialized BLAS with a larger process-wide default.
    with threadpool_limits(limits=1):
        return audit_seed(Path(directory) / f'seed_{seed}', phase=phase,
                          frozen=frozen, replay=seed == replay_seed)


def _seed_inventory(directory):
    found = []
    for path in Path(directory).glob('seed_*'):
        if path.is_dir():
            suffix = path.name.removeprefix('seed_')
            require(suffix.isdecimal() and path.name == f'seed_{int(suffix)}',
                    'Noncanonical seed directory')
            found.append(int(suffix))
    return sorted(found)


def _authorization(frozen):
    path = ROOT / 'r9_results/FINAL_EXECUTION_AUTHORIZATION.json'
    record = json.loads(path.read_text())
    require(record.get('authorized') is True, 'Final execution authorization is not affirmative')
    require(record['freeze_sha256'] == frozen['freeze_sha256'], 'Final authorization/source hash mismatch')
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path),
                freeze_sha256=record['freeze_sha256'])


def audit_parallel(directory, *, phase, replay_seed=None, workers=6, compare_sequential=False):
    require(phase in ('pilot', 'final'), 'Unknown campaign phase')
    require(isinstance(workers, int) and not isinstance(workers, bool) and workers >= 1,
            'Workers must be a positive integer')
    require(not compare_sequential or phase == 'pilot',
            'Sequential equivalence mode is restricted to pilots, never reruns 80 final audits')
    directory = Path(directory).resolve()
    found = _seed_inventory(directory)
    historical = verify_r8()
    pilot_snapshot = audit_pilot_snapshot(directory) if phase == 'pilot' else None
    prospective_guard = verify() if FREEZE.is_file() else None
    if phase == 'final':
        require(found == list(FINAL_SEEDS), 'Final audit requires all 80 exact seeds; no omissions/extras')
        require(prospective_guard is not None, 'Final source freeze missing')
        frozen = prospective_guard
        authorization = _authorization(frozen)
    else:
        require(bool(found) and set(found).issubset(PILOT_SEEDS), 'Unknown/empty pilot seed set')
        frozen, authorization = None, None
    if replay_seed is not None:
        require(replay_seed in found, 'Replay seed absent from campaign')
    effective_workers = min(workers, len(found))
    arguments = [(str(directory), seed, phase, frozen, replay_seed) for seed in found]
    # executor.map preserves seed order while worker exceptions propagate to
    # the caller; a failed/missing seed never becomes an excluded observation.
    with ProcessPoolExecutor(max_workers=effective_workers,
                             mp_context=multiprocessing.get_context('spawn')) as executor:
        records = list(executor.map(_worker, arguments, chunksize=1))
    require([row['seed'] for row in records] == found, 'Worker results reordered/missing/duplicated')
    metrics = {row['seed']: row['raw_metrics'] for row in records}
    inference = independent_inference(metrics, found) if phase == 'final' else None
    if phase == 'final':
        expected = compile_from_metrics(metrics, plan=replace(DEFAULT_PLAN, n=len(found), seeds=tuple(found)))
        require(len(inference['rows']) == len(expected['rows']) == 6, 'Incomplete hypothesis family')
        for a, b in zip(inference['rows'], expected['rows']):
            for key in ('claim', 'n', 'wins', 'seed_success', 'supported'):
                require(a[key] == b[key], 'Independent inference differs: ' + key)
            for key in ('p_one_sided_exact', 'p_holm'):
                close(a[key], b[key], 'Independent inference differs: ' + key, atol=1e-15, rtol=1e-12)
        require(_authorization(frozen) == authorization, 'Final authorization changed during audit')
    require(_seed_inventory(directory) == found, 'Campaign seed inventory changed during audit')
    require(verify_r8() == historical, 'Historical R8 source drift during audit')
    if prospective_guard is not None:
        require(verify() == prospective_guard, 'R9 source drift during audit')
    if phase == 'pilot':
        require(audit_pilot_snapshot(directory) == pilot_snapshot, 'Pilot snapshot changed during audit')
    result = dict(schema='polar-r9-independent-audit-v1', all_checks_passed=True,
        phase=phase, seed_order=found, seeds_audited=len(found), source=frozen,
        historical_r8_source=historical, pilot_snapshot=pilot_snapshot,
        rollouts_audited=sum(row['rollouts'] for row in records),
        raw_reconstruction=inference,
        pilot_inference_scope='no confirmatory inference; transfer domains deliberately unexecuted' if phase == 'pilot' else None,
        seeds=records, actor_replay_seed=replay_seed,
        scope='CRC/SHA, raw factual targets, scores/actions, delayed memory, calibration, capacity and inference; no environment replay, refit or external replication',
        limitations=['Full actor replay is optional and limited to the explicitly named seed and variants.',
                     'Source inspection establishes actor information boundaries; hashes alone do not prove causal identification.',
                     'noCross sets the shared gate to zero in preliminary and final predictions; it is an acute computational lesion.',
                     'Equal allocated coefficients and budgets do not imply equal effective statistical capacity.'])
    equivalence = None
    if compare_sequential:
        with threadpool_limits(limits=1):
            sequential = audit_campaign(directory, phase=phase, replay_seed=replay_seed)
        require(result == sequential, 'Parallel output differs from immutable sequential auditor')
        equivalence = dict(exact_full_report_match=True, seeds=list(found),
                           replay_seed=replay_seed, scope='Three pilot seeds; no final sequential rerun')
        # Repeat outer guards after the optional equivalence audit as well.
        require(_seed_inventory(directory) == found, 'Inventory changed during equivalence check')
        require(verify_r8() == historical, 'R8 source drift during equivalence check')
        if prospective_guard is not None:
            require(verify() == prospective_guard, 'R9 source drift during equivalence check')
    result['parallel'] = dict(requested_workers=workers, effective_workers=effective_workers,
        blas_threads_per_worker=1, scheduling='ProcessPoolExecutor spawn; complete seed unit; deterministic ordered collection',
        utility_sha256=sha256(Path(__file__)), outside_scientific_freeze=True,
        sequential_equivalence=equivalence, r9_source_guard=prospective_guard,
        final_execution_authorization=authorization)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--phase', choices=('pilot', 'final'), required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--replay-seed', type=int)
    parser.add_argument('--compare-sequential-pilots', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit_parallel(args.input, phase=args.phase, replay_seed=args.replay_seed,
                            workers=args.workers, compare_sequential=args.compare_sequential_pilots)
    atomic_json(args.output, result)
    print(json.dumps({key: result[key] for key in
                      ('all_checks_passed', 'phase', 'seeds_audited', 'rollouts_audited', 'parallel')}, indent=2))


if __name__ == '__main__':
    main()
