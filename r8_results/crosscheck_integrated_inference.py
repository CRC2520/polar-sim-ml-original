"""Crosscheck compiled inference against the independent integrated raw audit.

Reads small JSON reports only. Does not rerun or replace scientific evidence.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r8_completion.io import atomic_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def crosscheck(audit_path, compiled_dir):
    audit = json.loads(audit_path.read_text())
    confirmation_path = compiled_dir/'R8_CONFIRMATION.json'
    descriptive_path = compiled_dir/'R8_DESCRIPTIVES.json'
    confirmation = json.loads(confirmation_path.read_text())
    descriptive = json.loads(descriptive_path.read_text())
    rows = audit['rows']
    if audit['seeds'] != confirmation['seed_lists']['integrated'] or audit['seeds'] != descriptive['integrated_seeds']:
        raise AssertionError('Different integrated seed ordering')
    checks, maximum_error = 0, 0.

    def numeric(actual, expected):
        nonlocal checks, maximum_error
        if isinstance(actual, list):
            if not isinstance(expected, list) or len(actual) != len(expected):
                raise AssertionError('Shape mismatch')
            for a, b in zip(actual, expected):
                numeric(a, b)
            return
        error = abs(actual-expected)
        if not math.isfinite(error) or error > 1e-12:
            raise AssertionError(f'Numeric mismatch: {actual}, {expected}')
        checks += 1
        maximum_error = max(maximum_error, error)

    claims = {r['claim']: r for r in confirmation['rows']}
    for claim, vector in audit['success_vectors'].items():
        if claims[claim]['seed_success'] != vector:
            raise AssertionError('Different independent success vector: '+claim)
    for domain in audit['expected_domains']:
        for variant in rows[0]['native'][domain]:
            for metric in rows[0]['native'][domain][variant]:
                numeric([r['native'][domain][variant][metric] for r in rows],
                        descriptive['native'][domain][variant][metric]['by_seed'])
        for mode in ('aligned', 'miscredit'):
            numeric([r['prediction'][domain][mode]['mse'] for r in rows],
                    descriptive['prediction'][domain][mode]['by_seed'])
            numeric([r['prediction'][domain][mode]['mse_by_horizon_variable'] for r in rows],
                    descriptive['prediction'][domain][mode]['mse_by_horizon_variable_by_seed'])
        for claim, fields in {
            'E_utility': ('eco_delta_alive', 'eco_delta_resource'),
            'T_utility': ('network_delta_alive', 'network_delta_resource'),
            'I_generic': ('generic_delta_alive', 'generic_delta_resource'),
        }.items():
            for field, target in zip(fields, ('delta_alive', 'delta_resource')):
                numeric([r['criteria']['effects'][domain][field] for r in rows],
                        confirmation['integrated_components'][domain][claim][target])
    for variant, fields in {
        'full': ('ecology_weight', 'network_gate'),
        'generic_joint': ('generic_ecology_weight', 'generic_gate'),
    }.items():
        numeric([[r['checkpoint_gates'][key] for key in fields] for r in rows],
                descriptive['diagnostics']['selections_by_seed'][variant])

    # Independent finite enumeration of binomial tails and Holm recurrence.
    pvalues = []
    for row in confirmation['rows']:
        n, wins = row['n'], sum(row['seed_success'])
        if wins != row['wins']:
            raise AssertionError('Invalid number of successes')
        p = sum(math.comb(n, k) for k in range(wins, n+1))/(2**n)
        numeric(p, row['p_one_sided_exact'])
        pvalues.append(p)
    previous = 0.
    for rank, i in enumerate(sorted(range(len(pvalues)), key=lambda i: (pvalues[i], i))):
        adjusted = min(1., max(previous, (len(pvalues)-rank)*pvalues[i]))
        numeric(adjusted, confirmation['rows'][i]['p_holm'])
        if (adjusted <= .05) != confirmation['rows'][i]['supported_at_family_alpha_0_05']:
            raise AssertionError('Wrong adjusted decision')
        previous = adjusted
    return {'crosscheck': 'PASS', 'integrated_seeds': audit['seeds'],
            'four_independent_vectors_match': True, 'native_metrics_mse_and_gates_match': True,
            'all_six_binomial_tails_and_holm_recomputed': True,
            'numeric_values_compared': checks, 'maximum_absolute_error': maximum_error,
            'source_commit': confirmation['source_commit'],
            'freeze_sha256': confirmation['freeze_sha256'],
            'evidence': {str(p): sha(p) for p in (audit_path, confirmation_path, descriptive_path)},
            'script_sha256': sha(__file__),
            'scope': 'Internal crosscheck in shared environment; Q endpoints audited separately.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--compiled-dir', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = crosscheck(args.audit, args.compiled_dir)
    atomic_json(args.report, result)
    print(json.dumps(result, indent=2))
