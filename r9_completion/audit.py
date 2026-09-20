"""Independent, read-only checks of saved R9 observations and checkpoints.

This module never instantiates an environment, refits an actor or selects a
controller. Numerical endpoint/inference checks deliberately do not call the
runner's statistics functions. Raw artifacts can establish consistency with a
frozen implementation; they cannot establish construct validity or external
replication.
"""
from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path
import zipfile

import numpy as np
from scipy.stats import binomtest


class AuditError(ValueError):
    """A saved artifact violates a prospectively specified contract."""


def require(condition, message):
    if not bool(condition):
        raise AuditError(message)


def close(actual, expected, name, *, atol=1e-11, rtol=1e-10):
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape, name + ': shape differs')
    require(np.isfinite(a).all() and np.isfinite(b).all(), name + ': nonfinite')
    require(np.allclose(a, b, atol=atol, rtol=rtol), name + ': value differs')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_npz(path):
    """Check every member's CRC and load finite non-object arrays only."""
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, str(path) + ': NPZ CRC mismatch')
        names = archive.namelist()
        require(len(names) == len(set(names)), str(path) + ': duplicate NPZ member')
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    for key, value in arrays.items():
        require(value.dtype.kind != 'O', key + ': object arrays forbidden')
        if value.dtype.kind in 'fciub':
            require(np.isfinite(value).all(), key + ': nonfinite raw data')
    return arrays


def independent_endpoints(reward, alive, *, horizon=320):
    reward, alive = np.asarray(reward), np.asarray(alive)
    require(reward.shape == alive.shape == (horizon,), 'Fixed horizon incomplete')
    require(np.isfinite(reward).all(), 'Nonfinite reward')
    require(((reward >= 0.) & (reward <= 1.)).all(), 'Reward outside analytical scale')
    require(np.isin(alive, (0, 1)).all(), 'Alive status must be binary')
    require((np.diff(alive.astype(int)) <= 0).all(), 'Resurrection within episode')
    require((reward[alive == 0] == 0.).all(), 'Reward at or after death must be zero')
    return dict(reward_mean=float(np.sum(reward) / horizon),
                alive_fraction=float(np.count_nonzero(alive) / horizon),
                horizon=int(horizon), terminal_alive=bool(alive[-1]))


def independent_inference(metrics, seeds):
    """Recalculate six conjunctions, binomial tests and Holm independently.

    A shorter seed tuple is allowed only for mathematical fixtures/pilot
    descriptive consistency. The campaign entry point enforces all 80 finals.
    """
    seeds = tuple(int(seed) for seed in seeds)
    require(bool(seeds) and len(seeds) == len(set(seeds)), 'Seed identities invalid')
    normalized = {int(key): value for key, value in metrics.items()}
    require(len(normalized) == len(metrics), 'Duplicate normalized seed')
    require(set(normalized) == set(seeds), 'Missing or extra seed metrics')
    ecology = ('ecology_train', 'ecology_delay9')
    claims = (
        ('REL', ('noCross',), ecology, .02),
        ('GATE', ('constant_gate', 'permuted_gate'), ecology, .01),
        ('TENSION', ('scalar',), ecology, .01),
        ('CONTENT', ('permuted_content',), ecology, .01),
        ('GENERIC', ('dense',), ecology, .01),
        ('TRANSFER', ('dense',), ('inventory_transfer', 'thermal_transfer'), .01),
    )
    rows = []
    for claim, comparators, domains, margin in claims:
        success = []
        for seed in seeds:
            passed = []
            for domain in domains:
                for comparator in comparators:
                    try:
                        full = normalized[seed][domain]['full']
                        other = normalized[seed][domain][comparator]
                        fr, cr = float(full['reward_mean']), float(other['reward_mean'])
                        fa, ca = float(full['alive_fraction']), float(other['alive_fraction'])
                    except (KeyError, TypeError, ValueError) as exc:
                        raise AuditError(f'Missing metric {seed}/{domain}/{comparator}') from exc
                    require(all(np.isfinite(x) and 0 <= x <= 1 for x in (fr, cr, fa, ca)),
                            'Endpoint outside finite analytical scale')
                    passed.append(fr - cr >= margin and fa >= .80 and fa - ca >= -.005)
            success.append(all(passed))
        wins = sum(success)
        rows.append(dict(claim=claim, n=len(seeds), wins=wins, seed_success=success,
                         p_one_sided_exact=float(binomtest(wins, len(seeds), .5,
                                                          alternative='greater').pvalue)))
    order = sorted(range(6), key=lambda i: rows[i]['p_one_sided_exact'])
    for rank, index in enumerate(order):
        adjusted = min(1., max((6 - k) * rows[order[k]]['p_one_sided_exact']
                              for k in range(rank + 1)))
        rows[index].update(p_holm=float(adjusted), supported=adjusted <= .05)
    return dict(seed_order=list(seeds), rows=rows)


def training_digest(x, action, target):
    h = hashlib.sha256()
    for value in (x, action, target):
        arr = np.ascontiguousarray(value)
        h.update(str(arr.shape).encode() + arr.dtype.str.encode() + arr.tobytes())
    return h.hexdigest()


def audit_model(state, x, action, target):
    """Check training-only normalization, support, budgets and direct losses.

    Recomputes residuals from saved coefficients and factual training rows;
    does not run optimization or trust self-reported capacity counts.
    """
    x, action, target = np.asarray(x, dtype=float), np.asarray(action), np.asarray(target, dtype=float)
    cfg = state['config']
    d, k, iterations = cfg['n_features'], cfg['n_outputs'], cfg['iterations']
    require(state['fitted'] is True, 'Model must be fitted')
    require(x.ndim == 2 and x.shape[1] == d and len(x) > 0, 'Training X shape')
    require(target.shape == (len(x), k), 'Training target shape')
    require(action.shape == (len(x),) and action.dtype.kind in 'iu', 'Training action shape/type')
    require(np.isin(action, np.arange(4)).all(), 'Training action range')
    require(np.isfinite(x).all() and np.isfinite(target).all(), 'Training nonfinite')
    counts = np.bincount(action, minlength=4)
    require((counts >= 2).all(), 'Missing factual own-action coverage')
    mean = np.asarray(state['feature_mean_'])
    scale = np.asarray(state['feature_scale_'])
    close(mean, x.mean(0), 'Training-only mean')
    close(scale, np.maximum(x.std(0), cfg['normalization_floor']), 'Training-only scale')
    z = (x - mean) / scale
    coef, intercept = np.asarray(state['coef_']), np.asarray(state['intercept_'])
    support = np.asarray(state['allowed_support'])
    require(coef.shape == support.shape == (4, d, k), 'Coefficient/support shape')
    require(intercept.shape == (4, k), 'Intercept shape')
    require(np.isfinite(coef).all() and np.isfinite(intercept).all(), 'Nonfinite coefficients')
    require(np.isin(support, (False, True)).all(), 'Support must be binary')
    support = support.astype(bool)
    expected = np.ones_like(support)
    if cfg['mode'] == 'fixed':
        expected[:] = False
        rng = np.random.Generator(np.random.PCG64(1729))
        for ai in range(4):
            for output in range(k):
                expected[ai, rng.permutation(d)[:max(1, d // 2)], output] = True
    else:
        require(cfg['mode'] in ('sparse', 'dense'), 'Unknown model mode')
    require(np.array_equal(support, expected), 'Data-independent allowed support mismatch')
    require((coef[~support] == 0.).all(), 'Nonzero forbidden coefficient')
    history = np.asarray(state['objective_history_'])
    require(history.shape == (4, iterations + 1), 'Objective step budget differs')
    require(np.isfinite(history).all(), 'Nonfinite objective')
    require((np.diff(history, axis=1) <= 1e-10).all(), 'Objective increase beyond rounding')
    penalty = float(cfg['l1']) if cfg['mode'] == 'sparse' else 0.
    losses, rmses = [], []
    for ai in range(4):
        residual = z[action == ai] @ coef[ai] + intercept[ai] - target[action == ai]
        losses.append(.5 * np.mean(np.sum(residual ** 2, axis=1))
                      + .5 * cfg['ridge'] * np.sum(coef[ai] ** 2)
                      + penalty * np.sum(np.abs(coef[ai])))
        rmses.append(np.sqrt(np.mean(residual ** 2, axis=0)))
    close(history[:, -1], losses, 'Final direct per-action objectives')
    close(state['residual_rmse_'], rmses, 'In-sample residual diagnostic')
    diagnostics = state['fit_diagnostics']
    require(diagnostics['split'] == 'train', 'Model fit split differs')
    require(diagnostics['training_arrays_sha256'] == training_digest(x, action, target), 'Training array hash differs')
    require(diagnostics['samples'] == len(x), 'Training row count differs')
    require(diagnostics['action_counts'] == counts.tolist(), 'Factual action counts differ')
    require(diagnostics['feature_rank_centered'] == int(np.linalg.matrix_rank(z)), 'Effective training feature rank differs')
    capacity = dict(allocated_coefficients_including_intercepts=4 * (d + 1) * k,
                    allowed_slope_coefficients=int(support.sum()),
                    active_slope_coefficients=int(np.count_nonzero(coef)),
                    gradient_steps_per_action=int(iterations), gradient_steps_total=4 * int(iterations),
                    objective_evaluations_total=4 * (int(iterations) + 1))
    for key, value in capacity.items():
        require(diagnostics[key] == value, 'Model diagnostic differs: ' + key)
    close(diagnostics['final_objective_direct'], np.mean(losses), 'Final direct objective')
    require(diagnostics['causal_graph_identified'] is False, 'Predictive graph mislabeled causal')
    return dict(samples=len(x), action_counts=counts.tolist(), mode=cfg['mode'], **capacity)


def parameter_digest(state):
    keys = ('model', 'q', 'visits', 'q_scale', 'margin', 'gate_coef', 'gate_mode', 'constant_mode')
    record = {key: state[key] for key in keys}
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _local_mask():
    mask = np.zeros((48, 4), bool)
    for output, pairs in enumerate(((2,), (0,), (1,), (1, 3))):
        for pair in pairs:
            mask[2 * pair:2 * pair + 2, output] = True
            mask[16 + 4 * pair:20 + 4 * pair, output] = True
    return mask


def _model_predictions(model, features, observation):
    if model is None:
        return np.tile(np.column_stack((observation, np.full(len(features), .5)))[:, None, :],
                       (1, 4, 1)), np.zeros((len(features), 4, 4))
    z = (features - np.asarray(model['feature_mean_'])) / np.asarray(model['feature_scale_'])
    b, w = np.asarray(model['intercept_']), np.asarray(model['coef_'])
    mask = _local_mask()
    return (np.einsum('td,adk->tak', z, w * mask) + b,
            np.einsum('td,adk->tak', z, w * ~mask))


def audit_events(path, trace, variant):
    """Verify factual delayed targets and identified persistent memory changes."""
    memory_count = np.zeros(len(trace['action']), dtype=int)
    goal_count = np.zeros_like(memory_count)
    before_by_key, count_by_key = {}, {}
    seen_targets, seen_steps, episode_ids = set(), set(), set()
    held_goal = None
    goal_map = {'survival': [.75, .15, .10], 'reserve': [.25, .65, .10],
                'task': [.25, .15, .60], 'balanced': [.5, .3, .2]}
    reconsolidations = 0
    compressed = Path(str(path) + '.gz')
    require(not (path.exists() and compressed.exists()), 'Duplicate compressed and plain event streams')
    if compressed.exists():
        path = compressed
    if path.exists():
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                t = item['step']
                require(isinstance(t, int) and 0 <= t < len(memory_count), 'Event step outside episode')
                require(t not in seen_steps, 'Duplicate event step')
                require(t == len(seen_steps), 'Event stream is not complete chronological observation order')
                seen_steps.add(t)
                memory_count[t] += len(item['memory'])
                goal_count[t] += len(item['goals'])
                for event in item['memory']:
                    origin, destination, horizon = event['origin_time'], event['destination_time'], event['horizon']
                    require(horizon in (1, 4, 12) and destination == t and 0 <= origin < t,
                            'Delayed event temporal identifiers invalid')
                    require(destination - origin == horizon, 'Delayed event bridges wrong horizon')
                    pair = (origin, horizon)
                    require(pair not in seen_targets, 'Duplicate delayed target')
                    seen_targets.add(pair)
                    episode_ids.add(event['episode_id'])
                    require(len(episode_ids) == 1, 'Delayed memory events bridge episodes')
                    require(event['destination_absolute_time'] - event['origin_absolute_time'] == horizon,
                            'Delayed absolute time differs')
                    require(event['action'] == int(trace['action'][origin]), 'Memory target attributed to unexecuted action')
                    cue = [int(np.searchsorted(cuts, value)) for cuts, value in zip(
                        ((.3, .65), (.25, .6), (.33, .67)), trace['policy_observation'][origin])]
                    require(event['cue'] == cue, 'Memory cue differs from factual origin')
                    canonical_key = ':'.join(map(str, (*cue, event['action'], horizon)))
                    require(event['key'] == canonical_key, 'Memory key differs from factual origin/action/horizon')
                    close(event['baseline'], trace['policy_observation'][origin], 'Delayed factual baseline')
                    close(event['observed'], trace['policy_observation'][destination], 'Delayed factual observation')
                    delta = np.asarray(event['observed']) - event['baseline']
                    close(event['delta'], delta, 'Delayed factual delta')
                    before, after = np.asarray(event['before']), np.asarray(event['after'])
                    error = float(np.mean(np.abs(delta - before)))
                    close(event['prediction_error'], error, 'Memory prediction error')
                    key = event['key']
                    if key in before_by_key:
                        close(before, before_by_key[key], 'Persistent prior memory value')
                        require(event['count_before'] == count_by_key[key], 'Persistent memory count differs')
                    applied = event['update_applied']
                    require(applied == (variant != 'frozenMemory'), 'Memory lesion update contract')
                    recon = bool(event['recalled'] and error >= .08 and applied)
                    require(event['reconsolidated'] == recon, 'Reconsolidation label differs')
                    if event['recalled']:
                        require(event['recalled_at'] >= event['origin_absolute_time']
                                and event['recalled_at'] < event['destination_absolute_time'],
                                'Recall must precede observed correction')
                        require(event['count_before'] > 0, 'Recalled record must exist already')
                    rate = (0. if not applied else .6 if recon else 1. / (event['count_before'] + 1))
                    close(event['learning_rate'], rate, 'Memory update rate')
                    close(after, before + rate * (delta - before), 'Memory update equation')
                    require(event['count_after'] == event['count_before'] + int(applied), 'Memory update count')
                    before_by_key[key], count_by_key[key] = after, event['count_after']
                    reconsolidations += int(recon)
                for event in item['goals']:
                    require(event['time'] == t, 'Goal event time differs')
                    require(event['type'] in ('goal_formed', 'goal_revised', 'goal_abandoned', 'goal_persisted'), 'Unknown goal event')
                    if variant == 'noGoalRevision':
                        require(event['type'] not in ('goal_revised', 'goal_abandoned'), 'Frozen goal revised')
                    if event['type'] == 'goal_revised':
                        require(held_goal is not None, 'Revision before goal formation')
                        close(event['before'], held_goal, 'Goal revision starts from persistent weights')
                        held_goal = np.asarray(event['after'])
                    elif event['type'] == 'goal_formed':
                        require(event['goal'] in goal_map, 'Unknown operational goal')
                        close(event['weights'], goal_map[event['goal']], 'Defined goal weights')
                        held_goal = np.asarray(event['weights'])
                    elif event['type'] == 'goal_persisted':
                        require(held_goal is not None, 'Persistence before goal formation')
                        close(held_goal, goal_map[event['goal']], 'Persistent goal label/weights')
                require(held_goal is not None, 'No goal formed before native readout')
                expected_goal = held_goal
                if variant == 'block_resources':
                    expected_goal = np.asarray(goal_map['balanced'])
                elif variant == 'permuted_content':
                    expected_goal = held_goal[[2, 0, 1]]
                close(trace['goals'][t], expected_goal, 'Native goal weights follow persistent typed events')
                if variant == 'noGoalRevision':
                    close(trace['goals'][t], goal_map['balanced'], 'Frozen goal readout remains initial')
    close(memory_count, trace['workspace_update_count'], 'All memory events accounted for', atol=0, rtol=0)
    close(goal_count, trace['goal_event_count'], 'All goal events accounted for', atol=0, rtol=0)
    require(seen_steps == set(range(len(memory_count))), 'Missing observation goal-event stream')
    expected_targets = set() if variant == 'noMemory' else {
        (origin, horizon) for horizon in (1, 4, 12) for origin in range(len(memory_count) - horizon)}
    require(seen_targets == expected_targets, 'Incomplete factual delayed target set')
    if variant == 'noMemory':
        require(int(memory_count.sum()) == 0, 'No-memory lesion emitted memory update')
    return dict(memory_events=int(memory_count.sum()), goal_events=int(goal_count.sum()),
                reconsolidations=reconsolidations)


def _forecast_poles(obs, history, novelty):
    """Independent public encoder for the four preliminary forecasts."""
    r, e, d = obs
    last, actions = history[-1, :3], history[:, 3]
    result = np.array([[e, max(0., last[1] - e) * 8.],
        [d, max(0., d - actions[-1] / 3.)],
        [r, max(0., last[0] - r) * 8.],
        [actions[-1] / 3., np.mean(actions == 0)],
        [max(0., .65 - e) / .65, max(0., .65 - r) / .65],
        [novelty, np.mean(np.diff(actions) == 0)],
        [history[:, 1].min(), abs(actions[-1] - actions.mean()) / 3.],
        [max(0., d - last[2]) * 4., max(0., last[2] - d) * 4.]])
    return np.clip(result, 0., 1.)


def audit_rollout(directory, *, model=None, checkpoint=None, expected=None, margin=None):
    directory = Path(directory)
    trace = read_npz(directory / 'trace.npz')
    record = json.loads((directory / 'metrics.json').read_text())
    meta, reported = record['metadata'], record['metrics']
    if expected:
        for key, value in expected.items():
            require(meta[key] == value, 'Rollout metadata differs: ' + key)
    require(meta['measured_steps'] == 320, 'Rollout horizon differs')
    shapes = dict(observation=(320, 3), policy_observation=(320, 3), next_observation=(320, 3),
                  features=(320, 48), target=(320, 4), action=(320,), reward=(320,), alive=(320,),
                  predictions=(320, 4, 4), raw_predictions=(320, 4, 4),
                  local_predictions=(320, 4, 4), cross_predictions=(320, 4, 4),
                  scores=(320, 4), q_scores=(320, 4), gate=(320,), gate_context=(320, 6),
                  gate_context_used=(320, 6), goals=(320, 3), feasible=(320, 4),
                  lower_energy=(320, 4), memory_energy=(320, 4), memory_resource=(320, 4),
                  poles=(320, 8, 2), tension=(320, 8, 4), previous_prediction=(320, 3),
                  issued_pole_predictions=(320, 4, 8, 2), lag_predictions=(320, 4, 3))
    for key, shape in shapes.items():
        require(key in trace and trace[key].shape == shape, 'Raw shape differs: ' + key)
    require(trace['action'].dtype.kind in 'iu' and np.isin(trace['action'], range(4)).all(), 'Invalid action')
    close(trace['observation'][1:], trace['next_observation'][:-1], 'Factual observation continuity', atol=0, rtol=0)
    close(trace['target'], np.column_stack((trace['next_observation'], trace['reward'])), 'Factual target', atol=0, rtol=0)
    policy_obs = trace['observation'].copy()
    if meta['variant'] in ('observation_shock', 'dense_shock'):
        policy_obs[80:96] = np.clip(policy_obs[80:96] + .15 * np.array([1., -1., 1.]), 0., 1.)
    close(trace['policy_observation'], policy_obs, 'Public policy sensor perturbation', atol=0, rtol=0)
    close(trace['features'][:, :16], trace['poles'].reshape(320, 16), 'Observable pole features')
    tension_features = trace['tension']
    if meta['kind'] == 'scalar':
        tension_features = np.zeros_like(tension_features)
        tension_features[:, :, 0] = trace['tension'].sum(-1)
    close(trace['features'][:, 16:], tension_features.reshape(320, 32), 'Tension feature mapping')
    local, cross = _model_predictions(model, trace['features'], policy_obs)
    close(trace['local_predictions'], local, 'Saved local prediction from saved model')
    close(trace['cross_predictions'], cross, 'Saved cross prediction from saved model')
    raw = local + trace['gate'][:, None, None] * cross
    close(trace['raw_predictions'], raw, 'Actual gate contribution')
    close(trace['predictions'], np.clip(raw, 0., 1.), 'Physical prediction clipping')
    require(np.isin(trace['gate'], (0., .25, .5, 1.)).all(), 'Invalid gate branch')
    context = trace['gate_context'].copy()
    close(context[:, 0], np.ones(320), 'Gate intercept context')
    close(context[:, 1:4], policy_obs, 'Gate public context')
    close(context[:, 4], trace['tension'][:, :, 0].mean(1), 'Gate mismatch context')
    if meta['variant'] == 'permuted_gate':
        context[:, 1:4] = context[:, [2, 3, 1]]
    close(trace['gate_context_used'], context, 'Gate permutation exactly applied')
    close(trace['previous_prediction'][1:],
          trace['predictions'][np.arange(319), trace['action'][:-1], :3],
          'Preissued prediction uses own prior action')
    close(trace['previous_prediction'][0], policy_obs[0], 'Initial persistence prediction')
    previous_poles = np.concatenate((trace['poles'][:1],
        trace['issued_pole_predictions'][np.arange(319), trace['action'][:-1]]))
    close(trace['tension'][:, :, 0], np.abs(trace['poles'] - previous_poles).mean(-1), 'Mismatch uses preissued pole forecast')
    close(trace['tension'][:, :, 1], .2 * trace['poles'][:, :, 0] * trace['poles'][:, :, 1], 'Weighted coactivation')
    lag_tension = np.concatenate((np.zeros((1, 8, 4)), trace['tension'][:-1]))
    if meta['kind'] == 'scalar':
        scalar = lag_tension.sum(-1)
        lag_tension = np.zeros_like(lag_tension)
        lag_tension[:, :, 0] = scalar
    lag_x = np.column_stack((trace['poles'].reshape(320, 16), lag_tension.reshape(320, 32)))
    lag_local, lag_cross = _model_predictions(model, lag_x, policy_obs)
    close(trace['lag_predictions'], np.clip((lag_local + trace['gate'][:, None, None] * lag_cross)[:, :, :3], 0., 1.),
          'Same gate blocks preliminary prediction route')
    history = np.tile(np.r_[trace['observation'][0], 0.], (16, 1))
    opposition = []
    for step in range(320):
        current_history = np.tile(np.r_[policy_obs[step], 0.], (16, 1)) if meta['variant'] == 'noMemory' else history
        candidates = np.asarray([_forecast_poles(prediction, current_history, trace['poles'][step, 5, 0])
                                 for prediction in trace['lag_predictions'][step]])
        change = candidates[1:] - candidates[0]
        opposition.append(np.maximum(-change[:, :, 0] * change[:, :, 1], 0.).mean(0))
        history = np.concatenate((history[1:], np.r_[policy_obs[step], float(trace['action'][step])][None]))
    close(trace['tension'][:, :, 2], opposition, 'Operational predicted opposition from current forecasts')
    if margin is None:
        margin = np.asarray(checkpoint['margin']) if checkpoint is not None else np.full(4, .1)
    error_history = np.zeros((16, 3))
    uncertainty = []
    for step in range(320):
        error = np.zeros(3) if step == 0 else policy_obs[step] - trace['previous_prediction'][step]
        if meta['variant'] == 'noMemory':
            error_history[:] = error
        else:
            error_history = np.concatenate((error_history[1:], error[None]))
        uncertainty.append(float(np.mean(margin[:3]) + np.sqrt(np.mean(error_history ** 2))))
    close(trace['gate_context'][:, -1], uncertainty, 'Past-observation-only uncertainty')
    close(trace['tension'][:, :, 3], np.repeat(np.asarray(uncertainty)[:, None], 8, axis=1), 'Calibrated variable uncertainty')
    endpoints = independent_endpoints(trace['reward'], trace['alive'])
    recomputed = dict(reward_mean=endpoints['reward_mean'], alive_fraction=endpoints['alive_fraction'],
        reserve_mean=float(trace['reserve'].mean()), service_mean=float(trace['service'].mean()),
        constraint_fraction=float(trace['constraint'].mean()),
        gate_active_fraction=float((trace['gate'] > 0).mean()), fallback_fraction=float(trace['fallback'].mean()),
        mask_violations=int(trace['mask_violation'].sum()), probes=int(trace['probe'].sum()),
        observed_updates=int(trace['workspace_update_count'].sum()), goal_events=int(trace['goal_event_count'].sum()))
    for key, value in recomputed.items():
        close(reported[key], value, 'Metric reconstructed: ' + key)
    require(reported['all_finite'] is True, 'All-finite metric false')
    require(np.isin(trace['feasible'], (0, 1)).all() and trace['feasible'].any(1).all(), 'Invalid feasible mask')
    chosen_feasible = trace['feasible'][np.arange(320), trace['action']]
    require(np.array_equal(~chosen_feasible.astype(bool), trace['mask_violation']), 'Mask violation diagnostic')
    greedy = np.argmax(np.where(trace['feasible'], trace['scores'], -np.inf), axis=1)
    rng_seed = int.from_bytes(hashlib.sha256(f"R9|{meta['seed']}|{meta['episode']}|policy".encode()).digest()[:8], 'little')
    rng = np.random.default_rng(rng_seed)
    probes = rng.random(320) < meta['epsilon']
    probe_actions = rng.integers(0, 4, 320)
    require(np.array_equal(trace['probe'], probes), 'Own exploration tape differs')
    require(np.array_equal(trace['action'], np.where(probes, probe_actions, greedy)), 'Own action selection differs')
    if not meta['learn']:
        require(meta['policy_parameter_sha256_before'] == meta['policy_parameter_sha256_after'], 'Frozen policy mutated')
    if checkpoint is not None:
        require(meta['policy_parameter_sha256_before'] == parameter_digest(checkpoint), 'Evaluation used wrong checkpoint')
        mode = checkpoint['gate_mode']
        if meta['variant'] == 'constant_gate':
            mode = checkpoint['constant_mode']
        if meta['gate_override'] is not None:
            mode = meta['gate_override']
        if mode in ('off', 'constant_quarter', 'constant_half', 'constant_one'):
            expected_gate = np.full(320, {'off': 0., 'constant_quarter': .25, 'constant_half': .5, 'constant_one': 1.}[mode])
        else:
            threshold = {'context_zero': 0., 'context_small': .001, 'context_large': .01}[mode]
            expected_gate = (context @ np.asarray(checkpoint['gate_coef']) > threshold).astype(float)
        if meta['variant'] == 'noCross':
            expected_gate[:] = 0.
        close(trace['gate'], expected_gate, 'Frozen gate decision', atol=0, rtol=0)
        index_obs = np.clip(policy_obs, 0., 1.)
        idx = np.minimum(4, (index_obs[:, 0] * 5).astype(int)) * 12
        idx += np.minimum(3, (index_obs[:, 1] * 4).astype(int)) * 3 + np.minimum(2, (index_obs[:, 2] * 3).astype(int))
        q = np.tanh(np.asarray(checkpoint['q'])[idx] / np.asarray(checkpoint['q_scale'])[None, :, None])
        close(trace['q_scores'], np.einsum('ti,tia->ta', trace['goals'][:, [2, 0, 1]], q), 'Frozen Q readout')
    weights = trace['goals']
    require((weights >= 0.).all(), 'Negative goal weights')
    close(weights.sum(1), np.ones(320), 'Goal normalization')
    immediate = weights[:, 0, None] * trace['predictions'][:, :, 1] + weights[:, 1, None] * trace['predictions'][:, :, 0] + weights[:, 2, None] * trace['predictions'][:, :, 3]
    delayed = weights[:, 0, None] * trace['memory_energy'] + weights[:, 1, None] * trace['memory_resource'] + weights[:, 2, None] * trace['predictions'][:, :, 3]
    close(trace['scores'], .5 * immediate + .3 * trace['q_scores'] + .2 * delayed, 'Actual native score')
    viability_uncertainty = []
    for step in range(320):
        past = policy_obs[step:step + 1] if meta['variant'] == 'noMemory' else policy_obs[max(0, step - 15):step + 1]
        coordinate = 2 if meta['variant'] == 'permuted_content' else 1
        value = uncertainty[step] + past[:, coordinate].std() / np.sqrt(len(past))
        viability_uncertainty.append(0. if meta['variant'] == 'block_viability' else value)
    lower = .75 * (trace['predictions'][:, :, 1] - margin[1])
    lower += .25 * (trace['memory_energy'] - np.asarray(viability_uncertainty)[:, None])
    close(trace['lower_energy'], lower, 'Public-content feasibility lower bound')
    feasible = lower >= .2
    fallback = ~feasible.any(1)
    feasible[np.flatnonzero(fallback), np.argmax(lower[fallback], axis=1)] = True
    require(np.array_equal(trace['feasible'], feasible), 'Feasibility mask differs from actual bound')
    require(np.array_equal(trace['fallback'], fallback), 'Fallback diagnostic differs')
    events = audit_events(directory / 'workspace_events.jsonl', trace, meta['variant'])
    return trace, recomputed, meta, events


def _calibration_hash(trace):
    digest = hashlib.sha256()
    for key in ('features', 'action', 'target'):
        digest.update(np.ascontiguousarray(trace[key]).tobytes())
    return digest.hexdigest()


def audit_calibration(checkpoint, uncertainty_trace, gate_trace):
    records = checkpoint['calibration_record']
    margins, gain = None, None
    for role, trace in (('uncertainty', uncertainty_trace), ('gate', gate_trace)):
        rec = records[role]
        require(rec['data_sha256'] == _calibration_hash(trace), role + ' calibration provenance')
        require(rec['rows'] == 320, role + ' calibration row count')
        require(rec['action_counts'] == np.bincount(trace['action'], minlength=4).tolist(), role + ' calibration action counts')
        local, cross = _model_predictions(checkpoint['model'], trace['features'], trace['policy_observation'])
        selected = (local + cross)[np.arange(320), trace['action']]
        if role == 'uncertainty':
            margins = np.quantile(np.abs(trace['target'] - selected), .9, axis=0)
            close(checkpoint['margin'], margins, 'Disjoint empirical residual calibration')
            close(rec['empirical_margin'], margins, 'Calibration margin record')
        else:
            selected_local = local[np.arange(320), trace['action']]
            gain = np.mean((trace['target'] - selected_local) ** 2 - (trace['target'] - selected) ** 2, axis=1)
            context = trace['gate_context']
            matrix = context.T @ context + .1 * np.diag([0., 1., 1., 1., 1., 1.])
            close(checkpoint['gate_coef'], np.linalg.solve(matrix, context.T @ gain), 'Separate gate calibration')
            close(rec['gain_mean'], gain.mean(), 'Gate training gain')
            close(context[0, -1], np.asarray(checkpoint['margin'])[:3].mean(), 'Initial gate uncertainty baseline')
    require(records['uncertainty']['data_sha256'] != records['gate']['data_sha256'], 'Calibration splits identical')


def audit_manifest(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / 'COMPLETE.json').read_text())
    require(manifest['complete'] is True, 'Missing complete marker')
    recorded = set()
    for row in manifest['files']:
        path = directory / row['path']
        require(not Path(row['path']).is_absolute() and path.resolve().is_relative_to(directory), 'Manifest escapes seed directory')
        require(row['path'] not in recorded, 'Duplicate manifest member')
        recorded.add(row['path'])
        require(path.is_file() and path.stat().st_size == row['bytes'], 'Missing or resized artifact: ' + row['path'])
        require(sha256(path) == row['sha256'], 'Artifact SHA256 differs: ' + row['path'])
    found = {str(path.relative_to(directory)) for path in directory.rglob('*') if path.is_file() and path.name != 'COMPLETE.json'}
    require(recorded == found, 'Manifest contains missing or unlisted seed artifacts')
    return manifest


def audit_pilot_snapshot(directory):
    """Bind pilot review to its saved scientific source, when supplied."""
    record_path = Path(directory).parent / 'SOURCE_RECORD.json'
    if not record_path.is_file():
        return dict(available=False, scope='No pilot source snapshot supplied in this data restoration')
    record = json.loads(record_path.read_text())
    source_root = Path(__file__).resolve().parent
    io_differences = []
    for filename, expected in record['sources'].items():
        require(Path(filename).name == filename, 'Unsafe snapshot source name')
        snapshot = record_path.parent / 'SOURCE_SNAPSHOT' / filename
        require(snapshot.is_file() and sha256(snapshot) == expected, 'Pilot source snapshot hash differs: ' + filename)
        if sha256(source_root / filename) != expected:
            # The pilot's plain JSONL and final deterministic gzip differ only
            # in persistence. Compare every runner definition outside that
            # function, including all acquisition and selection algorithms.
            require(filename == 'runner.py', 'Current scientific source differs from pilot snapshot: ' + filename)
            import ast
            def scientific_ast(path):
                tree = ast.parse(path.read_text())
                for node in tree.body:
                    if isinstance(node, ast.ImportFrom) and node.module == 'r8_completion.io':
                        node.names = [alias for alias in node.names if alias.name not in ('atomic_text', '_atomic_bytes')]
                tree.body = [node for node in tree.body
                             if not (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == 'save_rollout')
                             and not (isinstance(node, ast.Import) and all(alias.name in ('gzip', 'io') for alias in node.names))]
                return ast.dump(tree, include_attributes=False)
            require(scientific_ast(snapshot) == scientific_ast(source_root / filename),
                    'Runner scientific definitions differ beyond event persistence')
            io_differences.append(filename)
    return dict(available=True, source_files=len(record['sources']),
                record_sha256=sha256(record_path), snapshot_hashes_match=True,
                current_scientific_definitions_match=True,
                persistence_only_differences=io_differences)


def _select_modes(records):
    selected = []
    for names in (('off', 'context_zero', 'context_small', 'context_large'),
                  ('off', 'constant_quarter', 'constant_half', 'constant_one')):
        passing = [name for name in names if records[name]['alive_fraction'] >= .80]
        if passing:
            selected.append(sorted(passing, key=lambda n: (-records[n]['reward_mean'], names.index(n)))[0])
        else:
            selected.append(sorted(names, key=lambda n: (-records[n]['alive_fraction'], -records[n]['reward_mean'], names.index(n)))[0])
    return selected


def audit_q_history(checkpoint, training_traces):
    """Reconstruct the tabular arithmetic from saved own transitions only."""
    q, visits, scale = np.zeros((60, 3, 4)), np.zeros((60, 4), dtype=np.int64), np.ones(3)
    def index(obs):
        r, e, d = np.clip(obs, 0., 1.)
        return min(4, int(r * 5)) * 12 + min(3, int(e * 4)) * 3 + min(2, int(d * 3))
    for trace in training_traces:
        for step in range(320):
            obs, nxt = trace['policy_observation'][step], trace['next_observation'][step]
            state, successor = index(obs), index(nxt)
            weights = trace['goals'][step, [2, 0, 1]]
            close(trace['q_scores'][step], weights @ np.tanh(q[state] / scale[:, None]), 'Training preaction own-Q readout')
            future_action = np.argmax(weights @ np.tanh(q[successor] / scale[:, None]))
            components = np.array([trace['reward'][step], nxt[1] - obs[1], nxt[0] - obs[0]])
            continuation = 0. if step == 319 else .95
            target = components + continuation * q[successor, :, future_action]
            action = trace['action'][step]
            q[state, :, action] += .1 * (target - q[state, :, action])
            visits[state, action] += 1
            scale = np.maximum(.05, np.sqrt(np.mean(q * q, axis=(0, 2))))
    close(checkpoint['q'], q, 'Final own-Q checkpoint')
    close(checkpoint['visits'], visits, 'Final own-Q visits', atol=0, rtol=0)
    close(checkpoint['q_scale'], scale, 'Final own-Q scale')


def audit_seed(directory, *, phase, frozen=None, replay=False):
    from r9_completion.config import DOMAINS, KINDS, VARIANTS, PROTOCOL
    directory = Path(directory)
    manifest_hash = sha256(directory / 'COMPLETE.json')
    manifest = audit_manifest(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    seed = summary['seed']
    require(directory.name == f'seed_{seed}', 'Seed directory/name mismatch')
    require(summary['phase'] == manifest['metadata']['phase'] == phase, 'Campaign phase mismatch')
    require(manifest['metadata']['seed'] == seed, 'Manifest seed mismatch')
    if phase == 'final':
        require(summary['source'] == manifest['metadata']['source'] == frozen, 'Final freeze evidence differs')
    require(summary['trained_kinds'] == list(KINDS), 'Missing trained comparator')
    require(summary['evaluated_variants'] == list(VARIANTS), 'Missing evaluated lesion')
    domains = DOMAINS if phase == 'final' else DOMAINS[:2]
    require(summary['evaluated_domains'] == list(domains), 'Wrong reserved/pilot domain set')
    checkpoints, capacities, training_hashes, metrics = {}, {}, {}, {}
    rollout_count = memory_events = reconsolidations = 0
    for kind in KINDS:
        checkpoint = json.loads((directory / 'checkpoints' / f'{kind}.json').read_text())
        require(checkpoint['seed'] == seed and checkpoint['kind'] == kind, 'Wrong kind/seed checkpoint')
        require(checkpoint['model']['config']['mode'] == {'full': 'sparse', 'scalar': 'sparse', 'dense': 'dense', 'fixed': 'fixed'}[kind], 'Wrong predictor comparator')
        checkpoints[kind] = checkpoint
        trained, prior_model = [], None
        for episode in range(8):
            trace, _, meta, events = audit_rollout(directory / 'training' / kind / f'episode_{episode:02d}',
                model=prior_model, expected=dict(seed=seed, kind=kind, episode=100 + episode,
                domain='ecology_train', variant='full', learn=True, epsilon=.2, gate_override='constant_half'))
            close(trace['gate'], np.full(320, .5), 'Training provisional gate', atol=0, rtol=0)
            trained.append(trace)
            rollout_count += 1
            memory_events += events['memory_events']
            reconsolidations += events['reconsolidations']
            if (episode + 1) % 2 == 0:
                fit = json.loads((directory / 'training' / kind / f'fit_{episode+1:02d}.json').read_text())
                require(fit['fit']['episodes'] == episode + 1, 'Wrong fitting episode')
                require(fit['fit']['rows'] == 320 * (episode + 1), 'Fit includes wrong number of rows')
                arrays = [np.concatenate([t[key] for t in trained]) for key in ('features', 'action', 'target')]
                capacities[kind] = audit_model(fit['model'], *arrays)
                require(fit['fit']['diagnostics'] == fit['model']['fit_diagnostics'], 'Fitting diagnostics differ')
                require(checkpoint['fit_log'][episode // 2] == fit['fit'], 'Checkpoint fitting history differs')
                prior_model = fit['model']
        require(checkpoint['model'] == prior_model, 'Final checkpoint model differs from last training fit')
        audit_q_history(checkpoint, trained)
        training_hashes[kind] = training_digest(*arrays)
        require(summary['own_acquisition_digests'][kind] == training_hashes[kind], 'Own acquisition summary differs')
        require(int(np.asarray(checkpoint['visits']).sum()) == 8 * 320, 'Q own-transition visit budget differs')
        calibration = []
        for episode in range(2):
            result = audit_rollout(directory / 'calibration' / kind / f'episode_{episode:02d}', model=prior_model,
                expected=dict(seed=seed, kind=kind, episode=200 + episode, domain='ecology_train',
                              variant='full', learn=False, epsilon=.2, gate_override='constant_half'),
                margin=np.full(4, .1) if episode == 0 else np.asarray(checkpoint['margin']))
            close(result[0]['gate'], np.full(320, .5), 'Calibration provisional gate', atol=0, rtol=0)
            calibration.append(result[0])
            rollout_count += 1
            memory_events += result[3]['memory_events']
            reconsolidations += result[3]['reconsolidations']
        audit_calibration(checkpoint, *calibration)
        selection = {}
        for mode in PROTOCOL['selection_candidates']:
            # The checkpoint retains selected mode labels, whereas selection
            # cloned the same calibrated base with default mode labels.
            base = dict(checkpoint, gate_mode='context_zero', constant_mode='off')
            _, selection[mode], _, events = audit_rollout(directory / 'selection' / kind / mode,
                model=prior_model, checkpoint=base, expected=dict(seed=seed, kind=kind, episode=300,
                domain='ecology_train', variant='full', learn=False, epsilon=0., gate_override=mode))
            rollout_count += 1
            memory_events += events['memory_events']
            reconsolidations += events['reconsolidations']
            for key, value in selection[mode].items():
                close(checkpoint['selection_record']['candidates'][mode][key], value, 'Selection source metric ' + key)
        require([checkpoint['gate_mode'], checkpoint['constant_mode']] == _select_modes(selection), 'Frozen selection policy differs')
        sr = checkpoint['selection_record']
        require(sr['selected'] == checkpoint['gate_mode'] and sr['constant_selected'] == checkpoint['constant_mode'], 'Selection record labels differ')
        require(sr['candidate_workspace_updates_not_retained'] is True, 'Selection memory retention differs')
        for key, value in (('total_training_transitions', 2560), ('total_calibration_transitions', 640), ('total_selection_transitions', 2240)):
            require(sr[key] == value, 'Selection budget differs: ' + key)
    tapes = {}
    for domain in domains:
        metrics[domain] = {}
        for variant in VARIANTS:
            kind = variant if variant in ('dense', 'scalar', 'fixed') else 'dense' if variant == 'dense_shock' else 'full'
            checkpoint = checkpoints[kind]
            trace, values, meta, events = audit_rollout(directory / 'evaluation' / domain / variant,
                model=checkpoint['model'], checkpoint=checkpoint,
                expected=dict(seed=seed, kind=kind, episode=700, domain=domain, variant=variant,
                              learn=False, epsilon=0., gate_override=None))
            metrics[domain][variant] = values
            rollout_count += 1
            memory_events += events['memory_events']
            reconsolidations += events['reconsolidations']
            tape = (meta['environment_seed'], meta['environment_tape_sha256'])
            if domain in tapes:
                require(tapes[domain] == tape, 'Unpaired evaluation exogenous tapes')
            tapes[domain] = tape
            for key, value in values.items():
                close(summary['metrics'][domain][variant][key], value, 'Seed summary ' + key)
            if replay and variant in ('full', 'noCross', 'permuted_content', 'observation_shock', 'dense_shock'):
                replay_actor(checkpoint, trace, variant)
    require(tapes['ecology_train'] == tapes['ecology_delay9'], 'Ecology delay comparison not paired by exact tape')
    require(sha256(directory / 'COMPLETE.json') == manifest_hash, 'Completion manifest changed during audit')
    audit_manifest(directory)
    return dict(seed=seed, phase=phase, rollouts=rollout_count, raw_metrics=metrics,
                capacities=capacities, own_acquisition_sha256=training_hashes,
                distinct_own_acquisition_buffers=len(set(training_hashes.values())),
                memory_events=memory_events, reconsolidations=reconsolidations,
                manifest_sha256=manifest_hash, actor_replay=bool(replay))


def replay_actor(checkpoint, trace, variant):
    """Replay actor only against saved observations; never advance a simulator."""
    from r9_completion.agent import NativeAgent
    actor = NativeAgent.from_state_dict(checkpoint)
    actor.reset_episode(trace['observation'][0], variant)
    for step in range(320):
        policy = actor.prepare(trace['policy_observation'][step], variant)
        for key in ('features', 'scores', 'predictions', 'gate', 'memory_energy', 'memory_resource', 'goals'):
            close(policy[key], trace[key][step], 'Saved-observation actor replay: ' + key)
        require(policy['action'] == int(trace['action'][step]), 'Actor replay action differs')
        actor.complete_transition(trace['policy_observation'][step], int(trace['action'][step]),
                                  trace['next_observation'][step], float(trace['reward'][step]), policy, learn=False)
    require(actor.parameter_digest() == parameter_digest(checkpoint), 'Actor replay mutated frozen model')


def audit_campaign(directory, *, phase, replay_seed=None):
    from r9_completion.config import FINAL_SEEDS, PILOT_SEEDS
    from r9_completion.provenance import verify
    directory = Path(directory)
    from r8_completion.provenance import verify as verify_r8
    historical = verify_r8()
    pilot_snapshot = audit_pilot_snapshot(directory) if phase == 'pilot' else None
    found = sorted(int(path.name.removeprefix('seed_')) for path in directory.glob('seed_*') if path.is_dir())
    if phase == 'final':
        require(found == list(FINAL_SEEDS), 'Final audit requires all 80 exact seeds; no omissions/extras')
        frozen = verify()
    else:
        require(bool(found) and set(found).issubset(PILOT_SEEDS), 'Unknown/empty pilot seed set')
        frozen = None
    if replay_seed is not None:
        require(replay_seed in found, 'Replay seed absent from campaign')
    records = [audit_seed(directory / f'seed_{seed}', phase=phase, frozen=frozen, replay=seed == replay_seed)
               for seed in found]
    metrics = {row['seed']: row['raw_metrics'] for row in records}
    inference = independent_inference(metrics, found) if phase == 'final' else None
    # A second implementation comparison is an additional check; the raw
    # reconstruction and exact tests above do not depend on this compiler.
    from dataclasses import replace
    from r9_completion.statistics import DEFAULT_PLAN, compile_from_metrics
    if phase == 'final':
        expected = compile_from_metrics(metrics, plan=replace(DEFAULT_PLAN, n=len(found), seeds=tuple(found)))
        for a, b in zip(inference['rows'], expected['rows']):
            for key in ('claim', 'n', 'wins', 'seed_success', 'supported'):
                require(a[key] == b[key], 'Independent inference differs: ' + key)
            for key in ('p_one_sided_exact', 'p_holm'):
                close(a[key], b[key], 'Independent inference differs: ' + key, atol=1e-15, rtol=1e-12)
        require(verify() == frozen, 'Source drift during audit')
    return dict(schema='polar-r9-independent-audit-v1', all_checks_passed=True,
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--phase', choices=('pilot', 'final'), required=True)
    parser.add_argument('--replay-seed', type=int)
    parser.add_argument('--output', type=Path, help='Optional new immutable JSON report; otherwise stdout')
    args = parser.parse_args()
    result = audit_campaign(args.input, phase=args.phase, replay_seed=args.replay_seed)
    if args.output:
        from r8_completion.io import atomic_json
        atomic_json(args.output, result)
        print(json.dumps({key: result[key] for key in ('all_checks_passed', 'phase', 'seeds_audited', 'rollouts_audited')}, indent=2))
    else:
        print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
