"""Action-conditioned learned inter-pair predictor for the integrated R8 agent.

This NEW predictor is not P1-T's configured regulator. W/K are predictive
coefficients on a fixed support, not discovered psychological or causal edges.
Only factual observed transitions enter fit; global contribution is selected
on development. Two channels per pair remain independently represented.
"""
from __future__ import annotations
import copy
import hashlib
import json
import numpy as np

VERSION = 'learned-network-0.1'
GATES = (0., .25, .5, 1.)
VARIANTS = ('learned', 'off', 'fixed', 'random', 'scrambled', 'generic_flat')
POLARITY_NAMES = (
    'Poder / Vulnerabilidad', 'Placer / Dolor', 'Integración / Fragmentación',
    'Control / Rendición', 'Deseo / Límite', 'Libertad / Orden',
    'Preservación / Transformación', 'Reconocimiento / Autenticidad',
)


def _finite(value, name):
    result = np.asarray(value, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(name+' must contain finite values')
    return result


def _poles(value, name='poles'):
    result = _finite(value, name)
    if result.shape[-2:] != (8, 2) or np.any((result < 0) | (result > 1)):
        raise ValueError(name+' must have trailing shape (8,2) in [0,1]')
    return result


def operational_tension(poles, desired, chi=0.):
    """Only use an observed goal or a prediction issued BEFORE poles arrived.

    This helper cannot establish temporal provenance: callers record the
    previous prediction, current observation and update order in their traces.
    Future targets must never be supplied as `desired`.
    """
    p, d = _poles(poles), _poles(desired, 'desired')
    if p.shape != d.shape:
        raise ValueError('desired and poles must have the same shape')
    incompatibility = np.broadcast_to(_finite(chi, 'chi'), p.shape[:-1])
    if np.any((incompatibility < 0) | (incompatibility > 1)):
        raise ValueError('chi must be in [0,1]')
    return np.mean(np.abs(d-p), axis=-1)+incompatibility*p[..., 0]*p[..., 1]


def to_signed_intensity(poles):
    p = _poles(poles)
    return p[..., 0]-p[..., 1], p.sum(axis=-1)


def from_signed_intensity(signed, intensity):
    signed, intensity = np.broadcast_arrays(_finite(signed, 'signed'), _finite(intensity, 'intensity'))
    if signed.shape[-1:] != (8,):
        raise ValueError('signed/intensity must have eight coordinates')
    poles = np.stack(((intensity+signed)/2, (intensity-signed)/2), axis=-1)
    if np.any((poles < -1e-12) | (poles > 1+1e-12)):
        raise ValueError('Infeasible signed/intensity pair')
    return np.clip(poles, 0, 1)


class LearnedNetwork:
    """Ridge local predictor plus a learned between-pair residual predictor.

    fit receives [sample,8,2] poles and [sample,8] pre-action tension, own
    actions and *later observed* poles. Predict accepts arbitrary batch axes.
    W/K are published in original feature units, centered by training means.
    """

    fit_mode = 'sequential_local_then_cross_residual'

    def __init__(self, actions=4, seed=0, ridge=.1):
        if not isinstance(actions, int) or actions < 2:
            raise ValueError('actions must be an integer >=2')
        if not isinstance(seed, (int, np.integer)) or int(seed) < 0:
            raise ValueError('seed must be a nonnegative integer')
        if not np.isfinite(ridge) or ridge <= 0:
            raise ValueError('ridge must be positive and finite')
        self.actions, self.seed, self.ridge = actions, int(seed), float(ridge)
        self.fitted = False
        self.gate = 0.
        self.gate_selection = {'status': 'not_selected', 'split': None}

    def _check_inputs(self, poles, tension):
        p = _poles(poles)
        tau = _finite(tension, 'tension')
        if tau.shape != p.shape[:-1] or np.any((tau < 0) | (tau > 2+1e-12)):
            raise ValueError('tension must match [...,8] and lie in [0,2]')
        return p, tau

    def fit(self, poles, tension, action, next_poles, *, split='train'):
        if split != 'train':
            raise ValueError('Parameter learning is allowed only on the train split')
        p, tau = self._check_inputs(poles, tension)
        y = _poles(next_poles, 'next_poles')
        if p.ndim != 3 or y.shape != p.shape or len(p) < self.actions:
            raise ValueError('fit expects at least actions samples of [N,8,2]')
        a = np.asarray(action)
        if a.shape != (len(p),) or a.dtype.kind not in 'iu' or np.any((a < 0) | (a >= self.actions)):
            raise ValueError('action must contain valid integer own actions [N]')
        counts = np.bincount(a, minlength=self.actions)
        if np.any(counts < 3):
            raise ValueError('At least three factual observations per action are required')
        flat = p.reshape(len(p), 16)
        targets = y.reshape(len(p), 16)
        self.poles_mean, self.tension_mean = flat.mean(axis=0), tau.mean(axis=0)
        # Floors are predeclared numerical guards, not tuned on held-out data.
        self.poles_scale = np.maximum(flat.std(axis=0), .05)
        self.tension_scale = np.maximum(tau.std(axis=0), .05)
        z = (flat-self.poles_mean)/self.poles_scale
        ztau = (tau-self.tension_mean)/self.tension_scale
        self.local = np.zeros((self.actions, 16, 3))
        self.W = np.zeros((self.actions, 16, 16))
        self.K = np.zeros((self.actions, 16, 8))
        local_condition = np.zeros((self.actions, 16))
        network_condition = np.zeros((self.actions, 16))
        for ai in range(self.actions):
            chosen = a == ai
            for dest in range(16):
                pair = dest//2
                own = np.array([2*pair, 2*pair+1])
                xlocal = np.column_stack((np.ones(int(chosen.sum())), z[chosen][:, own]))
                system = xlocal.T@xlocal+self.ridge*np.diag([0., 1., 1.])
                other_p = np.flatnonzero(np.arange(16)//2 != pair)
                other_tau = np.flatnonzero(np.arange(8) != pair)
                xnetwork = np.column_stack((z[chosen][:, other_p], ztau[chosen][:, other_tau]))
                if self.fit_mode == 'joint_ridge':
                    # Ordinary joint ridge on exactly the same 24 coefficients:
                    # intercept, own two channels, other 14 channels, seven τ.
                    joint = np.column_stack((xlocal, xnetwork))
                    system_network = joint.T@joint+self.ridge*np.diag([0.]+[1.]*23)
                    coef_all = np.linalg.solve(system_network, joint.T@targets[chosen, dest])
                    coef, coef_net = coef_all[:3], coef_all[3:]
                else:
                    coef = np.linalg.solve(system, xlocal.T@targets[chosen, dest])
                    residual = targets[chosen, dest]-xlocal@coef
                    system_network = xnetwork.T@xnetwork+self.ridge*np.eye(21)
                    coef_net = np.linalg.solve(system_network, xnetwork.T@residual)
                self.local[ai, dest] = coef
                self.W[ai, dest, other_p] = coef_net[:14]/self.poles_scale[other_p]
                self.K[ai, dest, other_tau] = coef_net[14:]/self.tension_scale[other_tau]
                local_condition[ai, dest] = np.linalg.cond(system)
                network_condition[ai, dest] = np.linalg.cond(system_network)
        digest = hashlib.sha256()
        for arr in (p, tau, a, y):
            value = np.ascontiguousarray(arr)
            digest.update(str(value.shape).encode()+value.dtype.str.encode()+value.tobytes())
        self.fit_diagnostics = {
            'split': 'train', 'samples': len(p), 'action_counts': counts.tolist(),
            'training_arrays_sha256': digest.hexdigest(),
            'joint_standardized_feature_rank': int(np.linalg.matrix_rank(np.column_stack((z, ztau)))),
            'feature_dimension': 24,
            'local_condition_max': float(local_condition.max()),
            'regularized_network_condition_max': float(network_condition.max()),
            'constant_or_near_constant_poles': int(np.count_nonzero(flat.std(axis=0) < .05)),
            'constant_or_near_constant_tension': int(np.count_nonzero(tau.std(axis=0) < .05)),
            'local_parameters': int(self.local.size),
            'cross_parameters': int(self.actions*16*21),
            'support': 'fixed_dense_between_pairs',
            'fit_mode': self.fit_mode,
            'causal_graph_identified': False,
        }
        self._build_controls()
        self.fitted = True
        # A refit invalidates any prior development selection.
        self.gate = 0.
        self.gate_selection = {'status': 'not_selected', 'split': None}
        return self

    def _build_controls(self):
        rng = np.random.default_rng([self.seed, 749])
        self.permutation = rng.permutation(8)
        permutation_channels = np.column_stack((2*self.permutation, 2*self.permutation+1)).ravel()
        self.scrambled_W = self.W[:, permutation_channels][:, :, permutation_channels].copy()
        self.scrambled_K = self.K[:, permutation_channels][:, :, self.permutation].copy()
        self.random_W = rng.normal(size=self.W.shape)
        self.random_K = rng.normal(size=self.K.shape)
        owner = np.arange(16)//2
        self.random_W[:, owner[:, None] == owner[None, :]] = 0
        self.random_K[:, np.arange(16), owner] = 0
        for ai in range(self.actions):
            learned_norm = np.sqrt(np.sum(self.W[ai]**2)+np.sum(self.K[ai]**2))
            random_norm = np.sqrt(np.sum(self.random_W[ai]**2)+np.sum(self.random_K[ai]**2))
            scale = learned_norm/max(random_norm, 1e-30)
            self.random_W[ai] *= scale
            self.random_K[ai] *= scale
        self.fixed_W, self.fixed_K = np.zeros_like(self.W), np.zeros_like(self.K)
        for source in range(7):
            dest = (source+1)%7
            self.fixed_W[:, 2*dest, 2*source] = .12
            self.fixed_W[:, 2*dest+1, 2*source+1] = .08
            self.fixed_K[:, 2*dest, source] = .18
            self.fixed_K[:, 2*dest+1, source] = -.14

    def _matrices(self, variant):
        if not self.fitted:
            raise RuntimeError('fit or restore the model before predicting')
        if variant not in VARIANTS:
            raise ValueError('Unknown network variant: '+str(variant))
        if variant in ('learned', 'generic_flat'):
            return self.W, self.K
        if variant == 'off':
            return np.zeros_like(self.W), np.zeros_like(self.K)
        return getattr(self, variant+'_W'), getattr(self, variant+'_K')

    def predict_components(self, poles, tension, *, variant='learned', gate=None):
        p, tau = self._check_inputs(poles, tension)
        w, k = self._matrices(variant)
        weight = self.gate if gate is None else float(gate)
        if not np.isfinite(weight) or not 0 <= weight <= 1:
            raise ValueError('gate must lie in [0,1]')
        batch = p.shape[:-2]
        flat = p.reshape(-1, 16)
        z = (flat-self.poles_mean)/self.poles_scale
        # Select the two channels belonging to each receiving channel's pair.
        own_first = (np.arange(16)//2)*2
        own_second = own_first+1
        xlocal = np.stack((np.ones_like(z), z[:, own_first], z[:, own_second]), axis=-1)
        local = np.einsum('nic,aic->nai', xlocal, self.local)
        centered_p = flat-self.poles_mean
        centered_tau = tau.reshape(-1, 8)-self.tension_mean
        if variant == 'generic_flat':
            # Generic vector implementation of the exact same function.
            state = np.stack([centered_p@w[ai].T for ai in range(self.actions)], axis=1)
            tens = np.stack([centered_tau@k[ai].T for ai in range(self.actions)], axis=1)
        else:
            state = np.einsum('nj,aij->nai', centered_p, w)
            tens = np.einsum('nj,aij->nai', centered_tau, k)
        combined = local+weight*(state+tens)
        shape = batch+(self.actions, 8, 2)
        return {'local_raw': local.reshape(shape), 'state_term': state.reshape(shape),
                'tension_term': tens.reshape(shape), 'network_term': (state+tens).reshape(shape),
                'unbounded': combined.reshape(shape), 'prediction': np.clip(combined, 0, 1).reshape(shape),
                'gate': weight, 'variant': variant}

    def predict_all(self, poles, tension, *, variant='learned', gate=None):
        return self.predict_components(poles, tension, variant=variant, gate=gate)['prediction']

    def predict_all_signed(self, signed, intensity, tension, *, variant='learned', gate=None):
        return self.predict_all(from_signed_intensity(signed, intensity), tension, variant=variant, gate=gate)

    def select_gate(self, development_losses, *, split='development'):
        """Record external NATIVE policy losses; lower is better, ties choose off.

        Caller evaluates all four candidates with equal development data/budget.
        This method never evaluates a test split or changes learned weights.
        """
        if not self.fitted or split != 'development':
            raise ValueError('Gate selection requires a fitted model and development split')
        losses = {float(key): float(value) for key, value in development_losses.items()}
        if set(losses) != set(GATES) or not np.isfinite(list(losses.values())).all():
            raise ValueError('Supply all four finite development candidate losses')
        self.gate = min(GATES, key=lambda gate: (losses[gate], gate))
        self.gate_selection = {'status': 'selected', 'split': split,
                               'criterion': 'external_native_loss_supplied_by_runner',
                               'candidate_losses': {str(k): losses[k] for k in GATES},
                               'selected_gate': self.gate, 'autonomous_state_gating': False}
        return self.gate

    def lesioned(self, *, remove_state=False, remove_tension=False, state_edges=(), tension_edges=()):
        """Return an independent lesion without refitting or changing local memory."""
        if not self.fitted:
            raise RuntimeError('A lesion requires a fitted model')
        clone = copy.deepcopy(self)
        if remove_state:
            clone.W.fill(0)
        if remove_tension:
            clone.K.fill(0)
        for dest, source in state_edges:
            if not (0 <= dest < 16 and 0 <= source < 16):
                raise ValueError('Invalid state edge')
            clone.W[:, dest, source] = 0
        for dest, source in tension_edges:
            if not (0 <= dest < 16 and 0 <= source < 8):
                raise ValueError('Invalid tension edge')
            clone.K[:, dest, source] = 0
        # Keep randomized-control arrays from the original fit: the lesion is a
        # specific learned-graph intervention, not a new randomized benchmark.
        return clone

    def state_dict(self):
        if not self.fitted:
            raise RuntimeError('Cannot export an unfitted network')
        arrays = ('local', 'W', 'K', 'poles_mean', 'tension_mean', 'poles_scale', 'tension_scale',
                  'random_W', 'random_K', 'scrambled_W', 'scrambled_K', 'fixed_W', 'fixed_K', 'permutation')
        return {'version': VERSION, 'fit_mode': self.fit_mode,
                'actions': self.actions, 'seed': self.seed, 'ridge': self.ridge,
                'gate': self.gate, 'gate_selection': copy.deepcopy(self.gate_selection),
                'fit_diagnostics': copy.deepcopy(self.fit_diagnostics),
                **{key: getattr(self, key).copy() for key in arrays}}

    @classmethod
    def from_state_dict(cls, state):
        if state.get('version') != VERSION:
            raise ValueError('Unsupported network state version')
        mode = state.get('fit_mode', 'sequential_local_then_cross_residual')
        if mode not in ('sequential_local_then_cross_residual', 'joint_ridge'):
            raise ValueError('Unsupported fitting rule')
        model_class = JointRidgePredictor if mode == 'joint_ridge' else LearnedNetwork
        obj = model_class(actions=int(state['actions']), seed=int(state['seed']), ridge=float(state['ridge']))
        shapes = {'local': (obj.actions, 16, 3), 'W': (obj.actions, 16, 16), 'K': (obj.actions, 16, 8),
                  'poles_mean': (16,), 'tension_mean': (8,), 'poles_scale': (16,), 'tension_scale': (8,),
                  'random_W': (obj.actions, 16, 16), 'random_K': (obj.actions, 16, 8),
                  'scrambled_W': (obj.actions, 16, 16), 'scrambled_K': (obj.actions, 16, 8),
                  'fixed_W': (obj.actions, 16, 16), 'fixed_K': (obj.actions, 16, 8), 'permutation': (8,)}
        for key, shape in shapes.items():
            arr = _finite(state[key], key)
            if arr.shape != shape:
                raise ValueError('Invalid serialized shape for '+key)
            setattr(obj, key, arr.astype(int) if key == 'permutation' else arr.copy())
        if np.any(obj.poles_scale <= 0) or np.any(obj.tension_scale <= 0):
            raise ValueError('Nonpositive stored feature scales')
        owner = np.arange(16)//2
        for w in (obj.W, obj.random_W, obj.scrambled_W, obj.fixed_W):
            if np.any(w[:, owner[:, None] == owner[None, :]] != 0):
                raise ValueError('Stored W contains a forbidden own-pair route')
        for k in (obj.K, obj.random_K, obj.scrambled_K, obj.fixed_K):
            if np.any(k[:, np.arange(16), owner] != 0):
                raise ValueError('Stored K contains a forbidden own-pair route')
        obj.gate = float(state['gate'])
        if not np.isfinite(obj.gate) or not 0 <= obj.gate <= 1:
            raise ValueError('Invalid stored gate')
        obj.gate_selection = copy.deepcopy(state['gate_selection'])
        obj.fit_diagnostics = copy.deepcopy(state['fit_diagnostics'])
        obj.fitted = True
        return obj

    def parameter_digest(self):
        state = self.state_dict()
        digest = hashlib.sha256()
        for key in sorted(state):
            value = state[key]
            digest.update(key.encode())
            if isinstance(value, np.ndarray):
                digest.update(value.dtype.str.encode()+str(value.shape).encode()+value.tobytes())
            else:
                digest.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
        return digest.hexdigest()


class JointRidgePredictor(LearnedNetwork):
    """Effective generic competitor, same information and parameter count.

    Each action/output fits jointly on the same 23 observed features plus
    intercept; the sequential POLAR predictor first fits local, then cross
    residuals. This is a learning-rule comparison, not a representation claim.
    Memory, actor/readouts and development evaluation budgets are supplied by
    the integrated runner and must match. Arithmetic cost need not be identical.
    """

    fit_mode = 'joint_ridge'
