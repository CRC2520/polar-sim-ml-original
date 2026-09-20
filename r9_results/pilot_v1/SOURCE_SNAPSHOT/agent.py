"""One R9 native agent with explicit public-information and content contracts.

The model predicts observations and factual reward; its support is not an
identified environmental causal graph. Goals, rewards and safety floors are
designed. During evaluation the associative workspace updates from past own
events; model coefficients, Q tables, gates and calibration remain frozen.
"""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np
from r9_completion.config import PROTOCOL, VERSION
from r9_completion.network import SparseActionModel, tension_vector
from r9_completion.workspace import (CognitiveWorkspace, route_message,
                                     viability_consumer, resource_consumer)


def encode_poles(obs, history, visits=0):
    """Eight operational pairs, with no claim of eight independent constructs."""
    o, h = np.asarray(obs, float), np.asarray(history, float)
    r, e, d = o
    last = h[-1, :3]
    actions = h[:, 3]
    return np.clip(np.array([
        [e, max(0., last[1]-e)*8.],
        [d, max(0., d-actions[-1]/3.)],
        [r, max(0., last[0]-r)*8.],
        [actions[-1]/3., np.mean(actions == 0)],
        [max(0., .65-e)/.65, max(0., .65-r)/.65],
        [1./np.sqrt(1.+visits), np.mean(np.diff(actions) == 0)],
        [h[:, 1].min(), abs(actions[-1]-actions.mean())/3.],
        [max(0., d-last[2])*4., max(0., last[2]-d)*4.],
    ]), 0., 1.)


def local_support():
    mask = np.zeros((48, 4), bool)
    for output, pairs in enumerate(((2,), (0,), (1,), (1, 3))):
        for pair in pairs:
            mask[pair*2:pair*2+2, output] = True
            mask[16+pair*4:16+pair*4+4, output] = True
    return mask


LOCAL_SUPPORT = local_support()


def state_index(obs):
    r, e, d = np.clip(np.asarray(obs, float), 0, 1)
    return int(min(4, int(r*5))*12+min(3, int(e*4))*3+min(2, int(d*3)))


def gate_features(obs, tensions, uncertainty):
    return np.r_[1., np.asarray(obs, float), np.mean(tensions[:, 0]), float(uncertainty)]


class NativeAgent:
    def __init__(self, seed, kind='full'):
        if kind not in ('full', 'dense', 'scalar', 'fixed'):
            raise ValueError(kind)
        self.seed, self.kind = int(seed), kind
        mode = {'full': 'sparse', 'dense': 'dense', 'scalar': 'sparse', 'fixed': 'fixed'}[kind]
        self.model = SparseActionModel(48, 4, mode=mode, l1=PROTOCOL['l1'],
                                      ridge=PROTOCOL['ridge'], iterations=PROTOCOL['network_iterations'])
        self.q = np.zeros((60, 3, 4))
        self.visits = np.zeros((60, 4), dtype=np.int64)
        self.q_scale = np.ones(3)
        self.workspace = CognitiveWorkspace()
        self.margin = np.full(4, .10)
        self.gate_coef = np.zeros(6)
        self.gate_mode = 'context_zero'
        self.constant_mode = 'off'
        self.fit_log = []
        self.calibration_record = {}
        self.selection_record = {}
        self.reset_episode(np.array([.7, .7, .5]))

    def reset_episode(self, observation, variant='full'):
        self.workspace.set_modes(no_memory=variant == 'noMemory',
                                 frozen_memory=variant == 'frozenMemory',
                                 no_goal_revision=variant == 'noGoalRevision')
        self.workspace.reset_episode()
        obs = np.asarray(observation, float)
        self.history = np.tile(np.r_[obs, 0.], (16, 1))
        self.previous_action = None
        self.previous_prediction = None
        self.previous_pole_prediction = None
        self.previous_tension = np.zeros((8, 4))
        self.recent_errors = np.zeros((16, 3))
        self.time = 0

    def _pack(self, poles, tensions):
        if self.kind == 'scalar':
            # Same allocated input width/products, lower effective dimension.
            # Zero padding avoids changing ridge shrinkage by duplicate columns.
            summary = tensions.sum(-1)
            tensions = np.zeros_like(tensions)
            tensions[:, 0] = summary
        return np.r_[poles.ravel(), tensions.ravel()]

    def _predictions(self, x, obs):
        if not self.model.fitted:
            fallback = np.tile(np.r_[obs, .5], (4, 1))
            return fallback, np.zeros_like(fallback)
        c = self.model.predict_components(x, LOCAL_SUPPORT)
        return c['local'], c['cross']

    def _gate(self, context, mode):
        if mode == 'off':
            return 0.
        if mode == 'constant_quarter':
            return .25
        if mode == 'constant_half':
            return .5
        if mode == 'constant_one':
            return 1.
        threshold = {'context_zero': 0., 'context_small': .001, 'context_large': .01}[mode]
        return float(float(context@self.gate_coef) > threshold)

    def prepare(self, observation, variant='full', gate_override=None):
        obs = np.asarray(observation, float)
        if obs.shape != (3,) or not np.isfinite(obs).all():
            raise ValueError('Policy requires exactly three finite public observations')
        history = self.history
        if variant == 'noMemory':
            history = np.tile(np.r_[obs, 0.], (16, 1))
        idx = state_index(obs)
        poles = encode_poles(obs, history, self.visits[idx].sum())
        old_prediction = poles if self.previous_pole_prediction is None else self.previous_pole_prediction
        error = np.zeros(3) if self.previous_prediction is None else obs-self.previous_prediction
        if variant == 'noMemory':
            self.recent_errors[:] = error
        else:
            self.recent_errors = np.concatenate((self.recent_errors[1:], error[None]))
        uncertainty = float(np.mean(self.margin[:3])+np.sqrt(np.mean(self.recent_errors**2)))
        preliminary_tension = np.zeros((8, 4))
        preliminary_tension[:, 0] = np.mean(np.abs(poles-old_prediction), axis=-1)
        context = gate_features(obs, preliminary_tension, uncertainty)
        context_used = context.copy()
        if variant == 'permuted_gate':
            context_used[1:4] = context_used[[2, 3, 1]]
        mode = self.gate_mode if gate_override is None else gate_override
        if variant == 'constant_gate':
            mode = self.constant_mode
        gate = 0. if variant == 'noCross' else self._gate(context_used, mode)
        lag_x = self._pack(poles, self.previous_tension)
        lag_local, lag_cross = self._predictions(lag_x, obs)
        lag_forecasts = np.clip((lag_local+gate*lag_cross)[:, :3], 0., 1.)
        predicted_poles = np.stack([encode_poles(p, history, self.visits[idx].sum())
                                    for p in lag_forecasts])
        tensions = tension_vector(poles, old_prediction, predicted_poles,
                                  np.full(8, uncertainty), coactivation_weight=.2)
        x = self._pack(poles, tensions)
        local, cross = self._predictions(x, obs)
        prediction_raw = local+gate*cross
        prediction = np.clip(prediction_raw, 0., 1.)

        message = self.workspace.observe(obs, prev_action=self.previous_action,
                                         prev_prediction=self.previous_prediction,
                                         uncertainty=uncertainty)
        viability_mode = ('content_lesion' if variant == 'block_viability' else
                          'permuted' if variant == 'permuted_content' else 'intact')
        resource_mode = ('content_lesion' if variant == 'block_resources' else
                         'permuted' if variant == 'permuted_content' else 'intact')
        v = viability_consumer(route_message(message, mode=viability_mode, consumer='viability'))
        r = resource_consumer(route_message(message, mode=resource_mode, consumer='resources'))
        weights = np.asarray(r['goal_weights'], float)
        weights = weights/weights.sum()
        memory_energy = np.asarray(v['energy_forecast'], float)
        memory_resource = np.asarray(r['resource_forecast'], float)
        q_values = np.tanh(self.q[idx]/self.q_scale[:, None])
        q_score = weights[[2, 0, 1]]@q_values
        immediate = weights[0]*prediction[:, 1]+weights[1]*prediction[:, 0]+weights[2]*prediction[:, 3]
        delayed = weights[0]*memory_energy+weights[1]*memory_resource+weights[2]*prediction[:, 3]
        score = .5*immediate+.3*q_score+.2*delayed
        # Calibrated residual is an empirical heuristic, not guaranteed coverage.
        lower_energy = .75*(prediction[:, 1]-self.margin[1])+.25*(memory_energy-float(v['uncertainty']))
        mask = lower_energy >= float(v['energy_floor'])
        fallback = not bool(mask.any())
        if fallback:
            mask[int(np.argmax(lower_energy))] = True
        selected = int(np.argmax(np.where(mask, score, -np.inf)))
        issued_poles = []
        for candidate_action in range(4):
            hypothetical_history = np.concatenate((history[1:], np.r_[obs, float(candidate_action)][None]))
            predicted_index = state_index(prediction[candidate_action, :3])
            issued_poles.append(encode_poles(prediction[candidate_action, :3], hypothetical_history,
                                             self.visits[predicted_index].sum()))
        return dict(action=selected, features=x, poles=poles, tension=tensions,
                    predictions=prediction, raw_predictions=prediction_raw,
                    local_predictions=local, cross_predictions=cross,
                    gate=gate, gate_context=context, gate_context_used=context_used,
                    scores=score, q_scores=q_score, lower_energy=lower_energy,
                    feasible=mask, fallback=fallback, goals=weights,
                    memory_energy=memory_energy, memory_resource=memory_resource,
                    issued_pole_predictions=np.asarray(issued_poles),
                    lag_predictions=lag_forecasts,
                    state_index=idx, previous_prediction=(obs if self.previous_prediction is None
                                                          else self.previous_prediction.copy()),
                    message=message)

    def complete_transition(self, observation, action, next_observation, reward, policy, learn=False, terminal=False):
        obs, nxt = np.asarray(observation, float), np.asarray(next_observation, float)
        state, successor = state_index(obs), state_index(nxt)
        if learn:
            weights = policy['goals'][[2, 0, 1]]
            successor_action = int(np.argmax(weights@np.tanh(self.q[successor]/self.q_scale[:, None])))
            rewards = np.array([reward, nxt[1]-obs[1], nxt[0]-obs[0]])
            target = rewards+(0. if terminal else PROTOCOL['q_gamma'])*self.q[successor, :, successor_action]
            self.q[state, :, action] += PROTOCOL['q_alpha']*(target-self.q[state, :, action])
            self.visits[state, action] += 1
            self.q_scale = np.maximum(.05, np.sqrt(np.mean(self.q*self.q, axis=(0, 2))))
        self.history = np.concatenate((self.history[1:], np.r_[obs, float(action)][None]), axis=0)
        self.previous_action = int(action)
        self.previous_prediction = policy['predictions'][action, :3].copy()
        self.previous_pole_prediction = policy['issued_pole_predictions'][action].copy()
        self.previous_tension = policy['tension'].copy()
        self.time += 1

    def fit_model(self, training, episode_count):
        x, a, y = training['features'], training['action'], training['target']
        self.model.fit(x, a, y)
        self.fit_log.append(dict(episodes=int(episode_count), rows=len(x),
                                 diagnostics=copy.deepcopy(self.model.fit_diagnostics)))

    def calibrate_uncertainty(self, calibration):
        x, a, y = calibration['features'], calibration['action'], calibration['target']
        all_pred = self.model.predict_all(x)
        idx = np.arange(len(x))
        pred = all_pred[idx, a]
        errors = np.abs(y-pred)
        self.margin = np.quantile(errors, PROTOCOL['calibration_quantile'], axis=0)
        digest = hashlib.sha256()
        for value in (x, a, y):
            digest.update(np.ascontiguousarray(value).tobytes())
        self.calibration_record['uncertainty'] = dict(rows=len(x), data_sha256=digest.hexdigest(),
            action_counts=np.bincount(a, minlength=4).tolist(), empirical_margin=self.margin.tolist(),
            scope='own-policy ecology episode 200 only; empirical quantile, no coverage guarantee')

    def fit_gate(self, calibration):
        x, a, y = calibration['features'], calibration['action'], calibration['target']
        all_pred = self.model.predict_all(x)
        local_pred = self.model.predict_local_all(x, LOCAL_SUPPORT)
        idx = np.arange(len(x))
        pred, local = all_pred[idx, a], local_pred[idx, a]
        gain = np.mean((y-local)**2-(y-pred)**2, axis=-1)
        context = np.asarray(calibration['gate_context']).copy()
        system = context.T@context+.1*np.diag([0., 1., 1., 1., 1., 1.])
        self.gate_coef = np.linalg.solve(system, context.T@gain)
        digest = hashlib.sha256()
        for value in (x, a, y):
            digest.update(np.ascontiguousarray(value).tobytes())
        self.calibration_record['gate'] = dict(rows=len(x), data_sha256=digest.hexdigest(),
            action_counts=np.bincount(a, minlength=4).tolist(), gain_mean=float(gain.mean()),
            scope='own-policy ecology episode 201 only; disjoint from uncertainty calibration')

    def parameter_digest(self):
        record = dict(model=self.model.state_dict(), q=self.q.tolist(), visits=self.visits.tolist(),
                      q_scale=self.q_scale.tolist(), margin=self.margin.tolist(),
                      gate_coef=self.gate_coef.tolist(), gate_mode=self.gate_mode,
                      constant_mode=self.constant_mode)
        return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()

    def state_dict(self):
        return dict(version=VERSION, seed=self.seed, kind=self.kind, model=self.model.state_dict(),
                    q=self.q.tolist(), visits=self.visits.tolist(), q_scale=self.q_scale.tolist(),
                    margin=self.margin.tolist(), gate_coef=self.gate_coef.tolist(),
                    gate_mode=self.gate_mode, constant_mode=self.constant_mode,
                    workspace=self.workspace.state_dict(), fit_log=copy.deepcopy(self.fit_log),
                    calibration_record=copy.deepcopy(self.calibration_record),
                    selection_record=copy.deepcopy(self.selection_record))

    @classmethod
    def from_state_dict(cls, state):
        if state['version'] != VERSION:
            raise ValueError('Unknown agent version')
        obj = cls(state['seed'], state['kind'])
        obj.model = SparseActionModel.from_state_dict(state['model'])
        for key in ('q', 'visits', 'q_scale', 'margin', 'gate_coef'):
            setattr(obj, key, np.asarray(state[key], dtype=np.int64 if key == 'visits' else float))
        for key in ('gate_mode', 'constant_mode', 'fit_log', 'calibration_record', 'selection_record'):
            setattr(obj, key, copy.deepcopy(state[key]))
        obj.workspace = CognitiveWorkspace.from_state_dict(state['workspace'])
        return obj
