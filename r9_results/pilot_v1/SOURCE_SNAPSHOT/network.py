"""R9 action-conditioned predictive sparsity and observable tension components.

This is a new predictor, not a causal graph learner or the P1/R8 regulator.
Factual next observations are training labels only. Callers own temporal
provenance, calibration, physical constraints, gating and the train/test split.
"""
from __future__ import annotations

import copy
import hashlib
import json
import numpy as np

VERSION = 'r9-sparse-action-model-1'
ACTIONS = 4
COMPONENT_NAMES = ('mismatch', 'coactivation_weighted', 'predicted_opposition', 'uncertainty')
MODES = ('sparse', 'dense', 'fixed')
FIXED_SUPPORT_SEED = 1729


def _finite(value, name):
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError(name+' must contain only finite values')
    return arr


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(name+' must be a positive integer')
    return int(value)


def _soft_threshold(value, threshold):
    return np.sign(value)*np.maximum(np.abs(value)-threshold, 0.)


def tension_vector(poles, previous_prediction, predicted_poles_all, uncertainty,
                   *, coactivation_weight=.2, n_pairs=8):
    """Return [..., n_pairs, 4] components known before the current action.

    `previous_prediction` was issued before current poles were observed.
    `predicted_poles_all` is a forecast, based on currently observable inputs,
    for each of four candidate actions. It is NOT factual future data.
    `uncertainty` is a nonnegative predictive-error scale calibrated on prior
    train/development observations, not variation between candidate actions.
    Arrays alone cannot certify timing; record their provenance in the caller.
    The third component is predicted opposed response, not causal conflict.
    """
    p = _finite(poles, 'poles')
    previous = _finite(previous_prediction, 'previous_prediction')
    candidates = _finite(predicted_poles_all, 'predicted_poles_all')
    if p.ndim < 2 or p.shape[-1] != 2 or p.shape[-2] < 1:
        raise ValueError('poles must have shape [...,n_pairs,2]')
    pairs = p.shape[-2]
    if _positive_int(n_pairs, 'n_pairs') != pairs:
        raise ValueError('Declared n_pairs differs from supplied poles')
    if previous.shape != p.shape or candidates.shape != p.shape[:-2]+(ACTIONS, pairs, 2):
        raise ValueError('Forecast shapes must match poles and four candidate actions')
    if any(np.any((a < 0.) | (a > 1.)) for a in (p, previous, candidates)):
        raise ValueError('Encoded poles and their forecasts must lie in [0,1]')
    try:
        weight = np.broadcast_to(_finite(coactivation_weight, 'coactivation_weight'), p.shape[:-1])
        calibrated = np.broadcast_to(_finite(uncertainty, 'uncertainty'), p.shape[:-1])
    except ValueError as exc:
        raise ValueError('Weights and uncertainty must broadcast to [...,n_pairs]') from exc
    if np.any((weight < 0.) | (weight > 1.)) or np.any(calibrated < 0.):
        raise ValueError('Coactivation weights must be in [0,1], uncertainty nonnegative')
    delta = candidates[..., 1:, :, :]-candidates[..., :1, :, :]
    opposition = np.mean(np.maximum(-delta[..., 0]*delta[..., 1], 0.), axis=-2)
    return np.stack((np.mean(np.abs(p-previous), axis=-1),
                     weight*p[..., 0]*p[..., 1], opposition, calibrated), axis=-1)


class SparseActionModel:
    """Four elastic-net linear predictors with a fixed proximal-step budget.

    J_a = ||[1,Z] Theta_a - Y||_F^2/(2*n_a)
          + ridge*||B_a||_F^2/2 + effective_l1*||B_a||_1.

    Z uses global training-only feature means/scales. Intercepts are unpenalized.
    `sparse` uses L1 and an unrestricted candidate support. `dense` uses no L1.
    `fixed` uses no L1 and a data-independent half-degree support. Every mode
    executes the same number and shape of dense gradient matrix products;
    effective support and statistical capacity are explicitly different.
    Predictions are unbounded; any physical projection belongs to the caller.
    """

    def __init__(self, n_features, n_outputs, mode='sparse', l1=.01,
                 ridge=.01, iterations=80, normalization_floor=1e-6):
        self.n_features = _positive_int(n_features, 'n_features')
        self.n_outputs = _positive_int(n_outputs, 'n_outputs')
        if mode not in MODES:
            raise ValueError('Unknown mode: '+str(mode))
        self.mode = mode
        self.iterations = _positive_int(iterations, 'iterations')
        for name, value in (('l1', l1), ('ridge', ridge), ('normalization_floor', normalization_floor)):
            if not np.isfinite(value) or value < 0 or (name == 'normalization_floor' and value == 0):
                raise ValueError(name+' has an invalid value')
            setattr(self, name, float(value))
        self.effective_l1 = self.l1 if mode == 'sparse' else 0.
        self.allowed_support_ = np.ones((ACTIONS, self.n_features, self.n_outputs), dtype=bool)
        if mode == 'fixed':
            rng = np.random.Generator(np.random.PCG64(FIXED_SUPPORT_SEED))
            degree = max(1, self.n_features//2)
            self.allowed_support_[:] = False
            for action in range(ACTIONS):
                for output in range(self.n_outputs):
                    self.allowed_support_[action, rng.permutation(self.n_features)[:degree], output] = True
        self.fitted = False

    def _features(self, X):
        x = _finite(X, 'X')
        if x.ndim < 1 or x.shape[-1] != self.n_features:
            raise ValueError('X must have trailing n_features coordinates')
        return x

    def _training_arrays(self, X, action, Y):
        x, y, a = self._features(X), _finite(Y, 'Y'), np.asarray(action)
        if x.ndim != 2 or not len(x) or y.shape != (len(x), self.n_outputs):
            raise ValueError('Require X[n,d] and Y[n,k] with n>0')
        if a.shape != (len(x),) or a.dtype.kind not in 'iu' or np.any((a < 0) | (a >= ACTIONS)):
            raise ValueError('action must be integer own actions in {0,1,2,3}')
        return x, a, y

    @staticmethod
    def _loss(theta, gram, cross, y_square, ridge, l1):
        # Same squared-error objective in sufficient-statistic form. The
        # recorded final value is also independently recomputed from residuals.
        residual = .5*np.sum(theta*(gram@theta))-np.sum(theta*cross)+y_square
        return float(residual+.5*ridge*np.sum(theta[1:]**2)+l1*np.sum(np.abs(theta[1:])))

    def fit(self, X, action, Y, *, split='train'):
        if split != 'train':
            raise ValueError('Parameter fitting is restricted to the train split')
        x, a, y = self._training_arrays(X, action, Y)
        counts = np.bincount(a, minlength=ACTIONS)
        if np.any(counts < 2):
            raise ValueError('At least two factual training observations per action are required')
        self.feature_mean_ = x.mean(axis=0)
        raw_scale = x.std(axis=0)
        self.feature_scale_ = np.maximum(raw_scale, self.normalization_floor)
        z = (x-self.feature_mean_)/self.feature_scale_
        self.coef_ = np.zeros((ACTIONS, self.n_features, self.n_outputs))
        self.intercept_ = np.zeros((ACTIONS, self.n_outputs))
        self.objective_history_ = np.empty((ACTIONS, self.iterations+1))
        self.lipschitz_ = np.empty(ACTIONS)
        self.residual_rmse_ = np.empty((ACTIONS, self.n_outputs))
        penalty = np.diag(np.r_[0., np.ones(self.n_features)])
        for ai in range(ACTIONS):
            subset = a == ai
            design = np.column_stack((np.ones(int(counts[ai])), z[subset]))
            target = y[subset]
            gram = design.T@design/counts[ai]
            cross = design.T@target/counts[ai]
            y_square = .5*np.sum(target**2)/counts[ai]
            hessian = gram+self.ridge*penalty
            # A tiny conservative multiplier avoids a numerical underestimate
            # of the largest eigenvalue; every mode computes the same bound.
            lipschitz = float(np.linalg.eigvalsh(hessian)[-1])*(1.+1e-12)
            if not np.isfinite(lipschitz) or lipschitz <= 0:
                raise ValueError('Training produced an invalid Lipschitz bound')
            step = 1./lipschitz
            theta = np.zeros((self.n_features+1, self.n_outputs))
            theta[0] = target.mean(axis=0)
            self.lipschitz_[ai] = lipschitz
            self.objective_history_[ai, 0] = self._loss(theta, gram, cross, y_square, self.ridge, self.effective_l1)
            for iteration in range(self.iterations):
                gradient = hessian@theta-cross
                proposal = theta-step*gradient
                proposal[1:] = _soft_threshold(proposal[1:], step*self.effective_l1)
                proposal[1:] *= self.allowed_support_[ai]
                theta = proposal
                self.objective_history_[ai, iteration+1] = self._loss(theta, gram, cross, y_square, self.ridge, self.effective_l1)
            self.intercept_[ai], self.coef_[ai] = theta[0], theta[1:]
            self.residual_rmse_[ai] = np.sqrt(np.mean((design@theta-target)**2, axis=0))
        self.fitted = True
        self.support_ = self.coef_ != 0.
        digest = hashlib.sha256()
        for value in (x, a, y):
            arr = np.ascontiguousarray(value)
            digest.update(str(arr.shape).encode()+arr.dtype.str.encode()+arr.tobytes())
        self.fit_diagnostics = {
            'split': 'train', 'samples': len(x), 'action_counts': counts.tolist(),
            'training_arrays_sha256': digest.hexdigest(), 'mode': self.mode,
            'effective_l1': self.effective_l1, 'ridge': self.ridge,
            'feature_rank_centered': int(np.linalg.matrix_rank(z)),
            'feature_count': self.n_features, 'output_count': self.n_outputs,
            'features_below_scale_floor': int(np.count_nonzero(raw_scale < self.normalization_floor)),
            'allocated_coefficients_including_intercepts': int(ACTIONS*(self.n_features+1)*self.n_outputs),
            'allowed_slope_coefficients': int(self.allowed_support_.sum()),
            'active_slope_coefficients': int(self.support_.sum()),
            'active_by_action_output': self.support_.sum(axis=1).tolist(),
            'fixed_support_seed': FIXED_SUPPORT_SEED,
            'gradient_steps_per_action': self.iterations,
            'gradient_steps_total': ACTIONS*self.iterations,
            'dense_gradient_shape': [self.n_features+1, self.n_features+1, self.n_outputs],
            'objective_evaluations_total': ACTIONS*(self.iterations+1),
            'lipschitz_by_action': self.lipschitz_.tolist(),
            'objective_initial_by_action': self.objective_history_[:, 0].tolist(),
            'objective_final_by_action': self.objective_history_[:, -1].tolist(),
            'objective_largest_step_increase': float(np.max(np.diff(self.objective_history_, axis=1))),
            'final_objective_direct': self.objective_loss(x, a, y),
            'causal_graph_identified': False,
            'residual_rmse_scope': 'in-sample training diagnostic, not calibrated uncertainty',
        }
        return self

    def _standardized(self, X):
        if not self.fitted:
            raise RuntimeError('Fit or restore a fitted model before prediction')
        return (self._features(X)-self.feature_mean_)/self.feature_scale_

    def predict_all(self, X):
        z = self._standardized(X)
        return np.einsum('...d,adk->...ak', z, self.coef_)+self.intercept_

    def _local_mask(self, mask):
        value = np.asarray(mask)
        if value.shape != (self.n_features, self.n_outputs) or not np.isin(value, (0, 1)).all():
            raise ValueError('local_mask must be binary [n_features,n_outputs]')
        return value.astype(bool)

    def predict_local_all(self, X, local_mask):
        """Remove cross coefficients without refitting or changing centering."""
        z, mask = self._standardized(X), self._local_mask(local_mask)
        return np.einsum('...d,adk->...ak', z, self.coef_*mask[None])+self.intercept_

    def predict_components(self, X, local_mask):
        z, mask = self._standardized(X), self._local_mask(local_mask)
        local = np.einsum('...d,adk->...ak', z, self.coef_*mask[None])+self.intercept_
        cross = np.einsum('...d,adk->...ak', z, self.coef_*(~mask)[None])
        return {'local': local, 'cross': cross, 'full': self.predict_all(X)}

    def objective_loss(self, X=None, action=None, Y=None):
        if X is None and action is None and Y is None:
            if not self.fitted:
                raise RuntimeError('Fit model before requesting its objective')
            return float(np.mean(self.objective_history_[:, -1]))
        if X is None or action is None or Y is None:
            raise ValueError('Supply all of X, action and Y, or none')
        x, a, y = self._training_arrays(X, action, Y)
        prediction = self.predict_all(x)
        values = []
        for ai in range(ACTIONS):
            chosen = a == ai
            if not chosen.any():
                raise ValueError('The equal-action objective requires all four actions')
            residual = prediction[chosen, ai]-y[chosen]
            values.append(.5*np.mean(np.sum(residual**2, axis=-1))
                          +.5*self.ridge*np.sum(self.coef_[ai]**2)
                          +self.effective_l1*np.sum(np.abs(self.coef_[ai])))
        return float(np.mean(values))

    def state_dict(self):
        state = {'version': VERSION, 'config': {name: getattr(self, name) for name in
                 ('n_features', 'n_outputs', 'mode', 'l1', 'ridge', 'iterations', 'normalization_floor')},
                 'fitted': self.fitted, 'allowed_support': self.allowed_support_.tolist()}
        if self.fitted:
            for name in ('feature_mean_', 'feature_scale_', 'coef_', 'intercept_',
                         'objective_history_', 'lipschitz_', 'residual_rmse_'):
                state[name] = getattr(self, name).tolist()
            state['fit_diagnostics'] = copy.deepcopy(self.fit_diagnostics)
        return state

    @classmethod
    def from_state_dict(cls, state):
        state = copy.deepcopy(state)
        if state.get('version') != VERSION:
            raise ValueError('Unsupported model checkpoint version')
        obj = cls(**state['config'])
        if not np.array_equal(state['allowed_support'], obj.allowed_support_):
            raise ValueError('Checkpoint support differs from deterministic mode contract')
        if not isinstance(state.get('fitted'), bool):
            raise ValueError('Checkpoint fitted flag must be boolean')
        if state['fitted']:
            shapes = {'feature_mean_': (obj.n_features,), 'feature_scale_': (obj.n_features,),
                      'coef_': (ACTIONS, obj.n_features, obj.n_outputs),
                      'intercept_': (ACTIONS, obj.n_outputs),
                      'objective_history_': (ACTIONS, obj.iterations+1),
                      'lipschitz_': (ACTIONS,), 'residual_rmse_': (ACTIONS, obj.n_outputs)}
            for name, shape in shapes.items():
                value = _finite(state[name], name)
                if value.shape != shape:
                    raise ValueError('Checkpoint shape mismatch: '+name)
                setattr(obj, name, value.copy())
            if np.any(obj.feature_scale_ <= 0) or np.any(obj.lipschitz_ <= 0) or np.any(obj.residual_rmse_ < 0):
                raise ValueError('Invalid checkpoint scales')
            if np.any(obj.coef_[~obj.allowed_support_] != 0):
                raise ValueError('Checkpoint has coefficients outside allowed support')
            obj.support_ = obj.coef_ != 0
            obj.fit_diagnostics = copy.deepcopy(state['fit_diagnostics'])
            obj.fitted = True
        return obj

    def parameter_digest(self):
        serialized = json.dumps(self.state_dict(), sort_keys=True, separators=(',', ':'), allow_nan=False)
        return hashlib.sha256(serialized.encode()).hexdigest()
