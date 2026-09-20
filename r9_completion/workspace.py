"""Typed, factual, stateful workspace for the R9 operational agent.

This implements designed goals and an associative predictive memory.  It is not
an implementation or test of subjective awareness.  Only observations, an own
previous action, a previously issued prediction and supplied uncertainty enter.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, replace
import numpy as np


VERSION = 'R9.workspace.1'
HORIZONS = (1, 4, 12)
INITIAL_WEIGHTS = (.5, .3, .2)


def _tuple(value):
    a = np.asarray(value)
    return tuple(_tuple(v) if np.asarray(v).ndim else v.item() for v in a)


def _vector(value, name):
    x = np.asarray(value, dtype=float)
    if x.shape != (3,) or not np.isfinite(x).all():
        raise ValueError(f'{name} requires three finite public values')
    return x


@dataclass(frozen=True)
class StateEstimate:
    observation: tuple
    slope: tuple
    variability: tuple
    cue: tuple
    uncertainty: tuple
    prediction_error: tuple
    history_count: int


@dataclass(frozen=True)
class ActionConsequences:
    delayed_estimates: tuple  # [own action, horizon, reserve/health/demand]
    counts: tuple             # [own action, horizon]
    horizons: tuple = HORIZONS


@dataclass(frozen=True)
class GoalState:
    primary: str
    weights: tuple            # survival, reserve, task
    pressures: tuple
    age: int
    revision: int


@dataclass(frozen=True)
class WorkspaceMessage:
    episode_id: int
    time: int
    absolute_time: int
    state: StateEstimate
    consequences: ActionConsequences
    goals: GoalState
    pending_count: int
    memory_revision: int
    route: str = 'intact'

    def to_dict(self):
        return asdict(self)

    def payload_vector(self):
        """Fixed width numeric broadcast, including unchanged temporal fields."""
        s, c, g = self.state, self.consequences, self.goals
        return np.r_[self.episode_id, self.time, self.absolute_time,
                     s.observation, s.slope, s.variability, s.cue,
                     s.uncertainty, s.prediction_error, s.history_count,
                     np.asarray(c.delayed_estimates).ravel(),
                     np.asarray(c.counts).ravel(), c.horizons,
                     g.weights, g.pressures, g.age, g.revision,
                     self.pending_count, self.memory_revision].astype(float)


def route_message(message, mode='intact', consumer='viability'):
    """Deterministic content interventions with exactly the same payload width.

    Permutation changes semantic coordinate/action assignment, preserving every
    numeric multiset within a field. It does not change physical time or history.
    Lesion substitutes neutral content; it is not a bandwidth reduction.
    """
    if consumer not in ('viability', 'resources'):
        raise ValueError('Unknown consumer')
    if mode == 'intact':
        return message
    if mode not in ('content_lesion', 'permuted'):
        raise ValueError('Unknown message intervention')
    s, c, g = message.state, message.consequences, message.goals
    if mode == 'content_lesion':
        neutral = (.5, .5, 0.)
        state = replace(s, observation=neutral, slope=(0.,)*3,
                        variability=(0.,)*3, cue=(1, 1, 0),
                        uncertainty=(0.,)*3, prediction_error=(0.,)*3)
        consequences = replace(c, delayed_estimates=_tuple(np.tile(neutral, (4, 3, 1))),
                               counts=_tuple(np.zeros((4, 3), dtype=int)))
        goals = replace(g, primary='balanced', weights=INITIAL_WEIGHTS, pressures=(0.,)*3)
    else:
        coordinate = np.array([1, 2, 0]) if consumer == 'viability' else np.array([2, 0, 1])
        actions = np.array([2, 0, 3, 1])
        def permute(x):
            return _tuple(np.asarray(x)[coordinate])
        state = replace(s, observation=permute(s.observation), slope=permute(s.slope),
                        variability=permute(s.variability), cue=permute(s.cue),
                        uncertainty=permute(s.uncertainty), prediction_error=permute(s.prediction_error))
        consequences = replace(c,
            delayed_estimates=_tuple(np.asarray(c.delayed_estimates)[actions][:, :, coordinate]),
            counts=_tuple(np.asarray(c.counts)[actions]))
        goals = replace(g, weights=permute(g.weights), pressures=permute(g.pressures))
    return replace(message, state=state, consequences=consequences,
                   goals=goals, route=f'{consumer}:{mode}')


def _forecast(message, coordinate, horizon_index):
    s, c = message.state, message.consequences
    n = np.asarray(c.counts, float)[:, horizon_index]
    factual = np.asarray(c.delayed_estimates, float)[:, horizon_index, coordinate]
    # Shrink sparse associative recall toward own recent observed trend.
    trend = np.clip(s.observation[coordinate]+c.horizons[horizon_index]*s.slope[coordinate], 0., 1.)
    confidence = n/(n+3.)
    return np.clip(confidence*factual+(1.-confidence)*trend, 0., 1.)


def viability_consumer(message):
    """Independent recipient: near-term health assessment, without goal access."""
    s = message.state
    uncertainty = float(s.uncertainty[1]+s.variability[1]/np.sqrt(max(1, s.history_count)))
    return dict(energy_forecast=_forecast(message, 1, 0),
                energy_floor=.2, uncertainty=uncertainty)


def resource_consumer(message):
    """Independent recipient: delayed reserve assessment and persistent goals."""
    forecast = .5*(_forecast(message, 0, 1)+_forecast(message, 0, 2))
    return dict(resource_forecast=forecast,
                goal_weights=np.asarray(message.goals.weights, float).copy())


class CognitiveWorkspace:
    def __init__(self, observation_dim=3, history=16, horizons=HORIZONS,
                 no_memory=False, frozen_memory=False, no_goal_revision=False,
                 block_viability=False, block_resources=False,
                 initial_goal_weights=INITIAL_WEIGHTS):
        if observation_dim != 3 or tuple(horizons) != HORIZONS or int(history) < 2:
            raise ValueError('R9 requires three observations, history>=2 and horizons 1/4/12')
        weights = _vector(initial_goal_weights, 'goal weights')
        if (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('Goal weights must be nonnegative with positive total')
        self.history_limit = int(history)
        self.initial_goal_weights = weights/weights.sum()
        self.no_memory, self.frozen_memory = bool(no_memory), bool(frozen_memory)
        self.no_goal_revision = bool(no_goal_revision)
        self.block_viability, self.block_resources = bool(block_viability), bool(block_resources)
        self.memory = {}
        self.recalled = {}
        self.counters = dict(observations=0, formations=0, updates=0,
                             reconsolidations=0, suppressed_updates=0,
                             goal_formations=0, goal_revisions=0, goal_abandons=0)
        self.episode_id = -1
        self.absolute_time = -1
        self.memory_revision = 0
        self.reset_episode()

    def set_modes(self, *, no_memory=None, frozen_memory=None, no_goal_revision=None,
                  block_viability=None, block_resources=None):
        for name, value in locals().copy().items():
            if name != 'self' and value is not None:
                setattr(self, name, bool(value))
        if self.no_memory:
            # A structural lesion removes retained cue values as well as recall.
            self.memory.clear()
            self.recalled.clear()
            self.pending = []
            self._history = self._history[-1:]
        if self.no_goal_revision:
            self.goal_weights = self.initial_goal_weights.copy()
            self.goal_primary = 'balanced'

    def reset_episode(self):
        """Keep learned cue means; never bridge delayed targets across episodes."""
        self.episode_id += 1
        self.time = -1
        self._history = []
        self.pending = []
        self.recalled = {}
        self.previous_observation = None
        self.previous_cue = None
        self.previous_time = None
        self.previous_absolute_time = None
        self.goal_primary = None
        self.goal_weights = self.initial_goal_weights.copy()
        self.goal_pressures = np.zeros(3)
        self.goal_age = 0
        self.goal_revision = 0
        self.goal_candidate = None
        self.goal_candidate_age = 0
        self.last_events = []
        self.memory_updates = []
        self.last_message = None

    def history_array(self):
        return np.asarray(self._history, dtype=float).reshape(-1, 4).copy()

    @staticmethod
    def _cue(obs):
        return tuple(int(np.searchsorted(cuts, value)) for value, cuts in
                     zip(obs, ((.3, .65), (.25, .6), (.33, .67))))

    @staticmethod
    def _key(cue, action, horizon):
        return ':'.join(map(str, (*cue, action, horizon)))

    def _update_memory(self, record, horizon, obs):
        key = self._key(record['cue'], record['action'], horizon)
        entry = self.memory.get(key)
        delta = obs-np.asarray(record['observation'])
        before = np.zeros(3) if entry is None else np.asarray(entry['mean_delta'], float)
        count = 0 if entry is None else int(entry['count'])
        recalled_at = self.recalled.get(key, -1)
        reactivated = entry is not None and recalled_at >= record['absolute_time']
        error = float(np.mean(np.abs(delta-before)))
        reconsolidated = bool(reactivated and error >= .08 and not self.frozen_memory)
        rate = .6 if reconsolidated else 1./(count+1)
        after = before if self.frozen_memory else before+rate*(delta-before)
        event = dict(episode_id=self.episode_id, origin_time=record['time'],
                     destination_time=self.time, origin_absolute_time=record['absolute_time'],
                     destination_absolute_time=self.absolute_time, action=record['action'],
                     horizon=horizon, cue=list(record['cue']), key=key,
                     baseline=list(record['observation']), observed=obs.tolist(),
                     delta=delta.tolist(), before=before.tolist(), after=after.tolist(),
                     count_before=count, count_after=count if self.frozen_memory else count+1,
                     recalled=bool(reactivated), recalled_at=int(recalled_at),
                     prediction_error=error, learning_rate=0. if self.frozen_memory else rate,
                     reconsolidated=reconsolidated, update_applied=not self.frozen_memory)
        self.memory_updates.append(event)
        if self.frozen_memory:
            self.counters['suppressed_updates'] += 1
            return
        self.memory[key] = dict(mean_delta=after.tolist(), count=count+1,
                                last_episode=self.episode_id, last_time=self.time)
        self.memory_revision += 1
        self.counters['formations' if entry is None else 'updates'] += 1
        if reconsolidated:
            self.counters['reconsolidations'] += 1

    def _advance_goals(self, obs, slope):
        r, e, d = obs
        pressures = np.clip([(.55-e)/.55+max(0., -slope[1])*4.,
                             (.65-r)/.65+max(0., -slope[0])*4., d], 0., 1.)
        names = ('survival', 'reserve', 'task')
        candidate = names[int(np.argmax(pressures))] if pressures.max() >= .08 else 'balanced'
        weights = {'survival': np.array([.75, .15, .10]),
                   'reserve': np.array([.25, .65, .10]),
                   'task': np.array([.25, .15, .60]),
                   'balanced': self.initial_goal_weights}
        if self.no_goal_revision:
            candidate = 'balanced'
        if self.goal_primary is None:
            self.goal_primary = candidate
            self.goal_weights = (self.initial_goal_weights if self.no_goal_revision else weights[candidate]).copy()
            self.counters['goal_formations'] += 1
            self.last_events.append(dict(type='goal_formed', time=self.time, goal=candidate,
                                         weights=self.goal_weights.tolist(), reason='observed_needs'))
        else:
            self.goal_age += 1
            old_pressure = pressures[names.index(self.goal_primary)] if self.goal_primary in names else 0.
            new_pressure = pressures[names.index(candidate)] if candidate in names else 0.
            change = (not self.no_goal_revision and candidate != self.goal_primary and
                      (new_pressure > old_pressure+.15 or old_pressure < .05))
            if change:
                self.goal_candidate_age = self.goal_candidate_age+1 if candidate == self.goal_candidate else 1
                self.goal_candidate = candidate
            else:
                self.goal_candidate, self.goal_candidate_age = None, 0
            if change and self.goal_age >= 3 and self.goal_candidate_age >= 2:
                old, before = self.goal_primary, self.goal_weights.copy()
                self.goal_primary, self.goal_weights = candidate, weights[candidate].copy()
                self.goal_age, self.goal_candidate_age = 0, 0
                self.goal_candidate = None
                self.goal_revision += 1
                self.counters['goal_abandons'] += 1
                self.counters['goal_revisions'] += 1
                self.counters['goal_formations'] += 1
                self.last_events.extend([
                    dict(type='goal_abandoned', time=self.time, goal=old,
                         reason='resolved' if old_pressure < .05 else 'observed_reprioritization'),
                    dict(type='goal_revised', time=self.time, before=before.tolist(),
                         after=self.goal_weights.tolist(), evidence=pressures.tolist()),
                    dict(type='goal_formed', time=self.time, goal=candidate,
                         weights=self.goal_weights.tolist(), reason='observed_needs')])
            else:
                self.last_events.append(dict(type='goal_persisted', time=self.time,
                                             goal=self.goal_primary, age=self.goal_age))
        self.goal_pressures = pressures
        return GoalState(self.goal_primary, _tuple(self.goal_weights), _tuple(pressures),
                         self.goal_age, self.goal_revision)

    def observe(self, obs_t, prev_action=None, prev_prediction=None, uncertainty=0.):
        obs = _vector(obs_t, 'observation')
        if prev_action is not None and (int(prev_action) != prev_action or not 0 <= int(prev_action) < 4):
            raise ValueError('Previous own action must be an integer in 0..3')
        prediction = obs if prev_prediction is None else _vector(prev_prediction, 'previous prediction')
        unc = np.full(3, uncertainty, float) if np.asarray(uncertainty).ndim == 0 else _vector(uncertainty, 'uncertainty')
        if not np.isfinite(unc).all() or (unc < 0).any():
            raise ValueError('Uncertainty must be finite and nonnegative')
        self.time += 1
        self.absolute_time += 1
        self.counters['observations'] += 1
        self.last_events, self.memory_updates = [], []
        cue = self._cue(obs)
        if not self.no_memory:
            if self.previous_observation is not None and prev_action is not None:
                self.pending.append(dict(time=self.previous_time, absolute_time=self.previous_absolute_time,
                    observation=self.previous_observation.tolist(), cue=list(self.previous_cue),
                    action=int(prev_action), remaining=list(HORIZONS)))
            pending = []
            for record in self.pending:
                remain = []
                for horizon in record['remaining']:
                    if self.time-record['time'] == horizon:
                        self._update_memory(record, horizon, obs)
                    else:
                        remain.append(horizon)
                if remain:
                    record['remaining'] = remain
                    pending.append(record)
            self.pending = pending
        row = np.r_[obs, -1. if prev_action is None else float(prev_action)].tolist()
        self._history = [row] if self.no_memory else (self._history+[row])[-self.history_limit:]
        history = np.asarray(self._history)[:, :3]
        if len(history) > 1:
            t = np.arange(len(history), dtype=float)
            t -= t.mean()
            slope = (t[:, None]*history).sum(0)/(t@t)
            variability = history.std(0)
        else:
            slope, variability = np.zeros(3), np.zeros(3)
        estimates, counts = np.tile(obs, (4, 3, 1)), np.zeros((4, 3), dtype=int)
        if not self.no_memory:
            for a in range(4):
                for j, horizon in enumerate(HORIZONS):
                    key = self._key(cue, a, horizon)
                    if key in self.memory:
                        entry = self.memory[key]
                        estimates[a, j] = np.clip(obs+entry['mean_delta'], 0., 1.)
                        counts[a, j] = entry['count']
                        self.recalled[key] = self.absolute_time
        goals = self._advance_goals(obs, slope)
        message = WorkspaceMessage(self.episode_id, self.time, self.absolute_time,
            StateEstimate(_tuple(obs), _tuple(slope), _tuple(variability), cue,
                          _tuple(unc), _tuple(obs-prediction), len(history)),
            ActionConsequences(_tuple(estimates), _tuple(counts)), goals,
            len(self.pending), self.memory_revision)
        self.previous_observation, self.previous_cue = obs.copy(), cue
        self.previous_time, self.previous_absolute_time = self.time, self.absolute_time
        self.last_message = message
        return message

    def consume(self, message=None, viability_mode=None, resource_mode=None):
        message = self.last_message if message is None else message
        if message is None:
            raise ValueError('No observation has been broadcast')
        vm = viability_mode or ('content_lesion' if self.block_viability else 'intact')
        rm = resource_mode or ('content_lesion' if self.block_resources else 'intact')
        return (viability_consumer(route_message(message, vm, 'viability')),
                resource_consumer(route_message(message, rm, 'resources')))

    def state_dict(self):
        """Bounded working state plus cue table, never cumulative event traces."""
        return copy.deepcopy(dict(version=VERSION, history_limit=self.history_limit,
            initial_goal_weights=self.initial_goal_weights.tolist(),
            modes={k: getattr(self, k) for k in ('no_memory', 'frozen_memory',
                    'no_goal_revision', 'block_viability', 'block_resources')},
            memory=self.memory, recalled=self.recalled, counters=self.counters,
            episode_id=self.episode_id, time=self.time, absolute_time=self.absolute_time,
            memory_revision=self.memory_revision, history=self._history, pending=self.pending,
            previous_observation=None if self.previous_observation is None else self.previous_observation.tolist(),
            previous_cue=self.previous_cue, previous_time=self.previous_time,
            previous_absolute_time=self.previous_absolute_time, goal_primary=self.goal_primary,
            goal_weights=self.goal_weights.tolist(), goal_pressures=self.goal_pressures.tolist(),
            goal_age=self.goal_age, goal_revision=self.goal_revision,
            goal_candidate=self.goal_candidate, goal_candidate_age=self.goal_candidate_age,
            last_events=self.last_events, memory_updates=self.memory_updates))

    @classmethod
    def from_state_dict(cls, state):
        if state['version'] != VERSION:
            raise ValueError('Unknown workspace version')
        obj = cls(history=state['history_limit'],
                  initial_goal_weights=state['initial_goal_weights'], **state['modes'])
        arrays = ('goal_weights', 'goal_pressures', 'previous_observation')
        excluded = ('version', 'history_limit', 'initial_goal_weights', 'modes', 'history')
        for key, value in state.items():
            if key not in excluded:
                setattr(obj, key, np.asarray(value, float) if key in arrays and value is not None else copy.deepcopy(value))
        obj._history = copy.deepcopy(state['history'])
        if obj.previous_cue is not None:
            obj.previous_cue = tuple(obj.previous_cue)
        return obj
