"""CP-NB-1.0.0 design and arithmetic; no learner or simulator imports."""
from __future__ import annotations
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent
BASE = '98d286f90a23b8a2b5ade19dfe0eb305d03c5468'
HISTORICAL = '50c4e08cf3d69c998ce95b8da1e1647a09b31339'
BRANCH = 'research/competence-normalization-budget-20260914'
REPO = 'CRC2520/polar-sim-ml-original'
NAMESPACE = 'CP-NB-1.0.0-20260914'
ROLES = ['G0', 'GD', 'PPO', 'SAC']
MODES = ['raw', 'standardized']
BUDGETS = [524288, 1048576]
REPLICATES = 3
EVAL_EPISODES = 32
CALIBRATION_STEPS = 32768
DELTA = 130 / 30
BOUNDARY = dict(B1E_executed=False, final_seeds_generated=False,
                ready_for_b1e_protocol_design=False, ready_for_b1e_freeze=False,
                ready_for_b1e_confirmatory_run=False, H_CAT='NOT_EVALUABLE',
                H_TRANSFER='NOT_EVALUATED', B1E_disposition='ON_HOLD_FOR_CURRENT_STRUCTURAL_CONFIRMATION')
GRAPH_HP = dict(lr=.0003, gamma=.99, rollout_steps=512, minibatch=128,
                epochs=4, gae_lambda=.95, clip=.2, value_coef=.5,
                max_grad_norm=.5, adam_eps=1e-5)
GENERIC_HP = dict(GRAPH_HP, rollout_steps=2048, minibatch=64, epochs=10)
SAC_HP = dict(lr=.0003, gamma=.99, batch_size=256, learning_starts=10000,
              train_freq=1, gradient_steps=1, tau=.005, buffer_size=BUDGETS[-1])


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def encode(x):
    return (json.dumps(x, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_new(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    b = encode(obj)
    if path.exists():
        require(path.read_bytes() == b, 'Content conflict: ' + str(path))
    else:
        with path.open('xb') as f:
            f.write(b)


def read(path):
    return json.loads(Path(path).read_bytes())


def seed(split, *identity, bits=63):
    require(split in {'normalization', 'training', 'evaluation', 'qa'}, 'No final/B1-E namespace allowed')
    require(bits in (32, 63), 'Unsupported seed width')
    b = json.dumps([NAMESPACE, split, *identity], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.sha256(b).digest()[:8], 'big') % (2**bits)


def registry():
    return [dict(id=f'{role}-{mode}-r{rep}', role=role, mode=mode, rep=rep,
                 mask={'G0': '000000', 'GD': '111111'}.get(role),
                 hp=(SAC_HP if role == 'SAC' else GENERIC_HP if role == 'PPO' else GRAPH_HP).copy(),
                 budgets=BUDGETS.copy(),
                 initial_seed=seed('training', 'weights', 'graph' if role in ('G0', 'GD') else role, rep, bits=32))
            for role in ROLES for mode in MODES for rep in range(REPLICATES)]


def plan():
    return dict(schema=NAMESPACE, roles=ROLES, input_modes=MODES, independent_blocks=REPLICATES,
                budgets=BUDGETS, budget_relation='nested checkpoints in one uninterrupted fit',
                fits=24, train_native_steps=24 * BUDGETS[-1],
                normalization_native_steps=REPLICATES * CALIBRATION_STEPS,
                evaluation_episodes_per_checkpoint=EVAL_EPISODES,
                evaluation_native_step_cap=(24 * 2 + REPLICATES) * EVAL_EPISODES * 500,
                max_parallel_fits=4, fit_timeout_minutes=180,
                normalization=dict(type='fixed_per_block_shared_standardizer', samples_per_block=CALIBRATION_STEPS,
                                   zero_action_steps=CALIBRATION_STEPS // 2, uniform_action_steps=CALIBRATION_STEPS // 2,
                                   min_std=.01, clip=False, reward_normalization=False,
                                   uses_evaluation_data=False, online_updates=False),
                primary_loss='negative_undiscounted_native_return',
                competence=dict(loss_improvement_strictly_greater_than=DELTA, additional_displacement_at_least=1),
                structural_search=False, interventions=False, new_hyperparameter_search=False,
                automatic_progression=False, **BOUNDARY)


def git(*args):
    return subprocess.check_output(['git', *args], text=True).strip()


def preserve():
    def tree(ref):
        raw = subprocess.check_output(['git', 'ls-tree', '-r', '-z', ref])
        return {x.split(b'\t', 1)[1]: x.split(b'\t', 1)[0] for x in raw.split(b'\0') if x}
    a, b = tree(BASE), tree('HEAD')
    require(all(b.get(p) == v for p, v in a.items()), 'Historical files or modes changed')
    for p in set(b) - set(a):
        require(p.startswith(b'experiments/competence_pilot/') or p == b'.github/workflows/competence-pilot.yml',
                'Unexpected added path: ' + p.decode())
    return dict(status='PASS', base_commit=BASE, preserved_objects_and_modes=len(a))


def execution_guard():
    require(os.environ.get('GITHUB_REPOSITORY') == REPO, 'Wrong repository')
    require(os.environ.get('GITHUB_REF') == 'refs/heads/' + BRANCH, 'Wrong branch')
    require(os.environ.get('GITHUB_RUN_ATTEMPT') == '1', 'Scientific retries are not authorized')
    request = read(ROOT / 'START_REQUEST.json')
    require(request['namespace'] == NAMESPACE and request['authorize_competence_pilot'] is True, 'No scoped start request')
    require(request['B1E_executed'] is False and request['final_seeds_generated'] is False, 'Invalid boundary')
    require(git('rev-parse', 'HEAD') == os.environ['GITHUB_SHA'], 'Unexpected checkout')
    changed = git('diff', '--name-only', request['code_commit'], 'HEAD').splitlines()
    require(changed == ['experiments/competence_pilot/START_REQUEST.json'], 'Start commit contains unapproved scientific edits')
    require(sha(ROOT / 'PLAN.json') == request['plan_sha256'], 'Plan changed after start request')
    require(sha(ROOT / 'PROTOCOL_ES.md') == request['protocol_sha256'], 'Protocol changed')
    return request


def summarize(rows):
    require(bool(rows), 'Cannot summarize absent episodes')
    require(all(math.isfinite(float(x[k])) for x in rows for k in ('loss', 'displacement', 'cycles')), 'Nonfinite episode')
    return dict(loss=statistics.mean(x['loss'] for x in rows),
                displacement=statistics.mean(x['displacement'] for x in rows),
                fall_fraction=statistics.mean(x['fallen_count'] > 0 for x in rows),
                drop_fraction=statistics.mean(bool(x['package_dropped']) for x in rows),
                cycles=statistics.mean(x['cycles'] for x in rows), episodes=len(rows))


def competence(summary, witness):
    improvement = witness['loss'] - summary['loss']
    displacement = summary['displacement'] - witness['displacement']
    return dict(loss_improvement=improvement, additional_displacement=displacement,
                loss_component_pass=improvement > DELTA, displacement_component_pass=displacement >= 1,
                descriptive_competence_pass=improvement > DELTA and displacement >= 1)


def factorial_contrasts(cells):
    out = {}
    for role in ROLES:
        values = {}
        for metric in ('loss', 'displacement', 'fall_fraction', 'cycles'):
            def v(mode, budget):
                return [x['summary'][metric] for x in cells[f'{role}-{mode}-{budget}']['by_rep']]
            raw_lo, raw_hi = v('raw', BUDGETS[0]), v('raw', BUDGETS[1])
            norm_lo, norm_hi = v('standardized', BUDGETS[0]), v('standardized', BUDGETS[1])
            arrays = dict(normalization_at_low=[a-b for a,b in zip(norm_lo, raw_lo)],
                          normalization_at_high=[a-b for a,b in zip(norm_hi, raw_hi)],
                          budget_raw=[a-b for a,b in zip(raw_hi, raw_lo)],
                          budget_standardized=[a-b for a,b in zip(norm_hi, norm_lo)],
                          interaction=[(a-b)-(c-d) for a,b,c,d in zip(norm_hi, norm_lo, raw_hi, raw_lo)])
            values[metric] = {k: dict(by_rep=a, mean=statistics.mean(a), sd_between_blocks=statistics.stdev(a))
                              for k,a in arrays.items()}
        out[role] = values
    return out
