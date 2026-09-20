"""Prospective R9 seed-level criteria and inference; no simulator or file IO.

Every primary win is a conjunction of practical reward gain, an absolute
survival floor, and survival noninferiority in all declared conditions.
Parameters are protocol choices, not validated measures of consciousness.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from numbers import Integral
from typing import Mapping

import numpy as np
from scipy.stats import binom


ECOLOGY_DOMAINS = ("ecology_train", "ecology_delay9")
TRANSFER_DOMAINS = ("inventory_transfer", "thermal_transfer")
FINAL_SEEDS = tuple(range(952001, 952081))
DEVELOPMENT_SEEDS = (950101, 950102, 950103)


@dataclass(frozen=True)
class ClaimSpec:
    name: str
    comparators: tuple[str, ...]
    domains: tuple[str, ...]
    reward_margin: float


CLAIM_SPECS = (
    ClaimSpec("REL", ("noCross",), ECOLOGY_DOMAINS, .02),
    ClaimSpec("GATE", ("constant_gate", "permuted_gate"), ECOLOGY_DOMAINS, .01),
    ClaimSpec("TENSION", ("scalar",), ECOLOGY_DOMAINS, .01),
    ClaimSpec("CONTENT", ("permuted_content",), ECOLOGY_DOMAINS, .01),
    ClaimSpec("GENERIC", ("dense",), ECOLOGY_DOMAINS, .01),
    ClaimSpec("TRANSFER", ("dense",), TRANSFER_DOMAINS, .01),
)
CLAIMS = tuple(spec.name for spec in CLAIM_SPECS)


@dataclass(frozen=True)
class StatisticalPlan:
    n: int = 80
    seeds: tuple[int, ...] = FINAL_SEEDS
    claims: tuple[ClaimSpec, ...] = CLAIM_SPECS
    alpha: float = .05
    absolute_alive_floor: float = .80
    maximum_alive_loss: float = .005
    null_win_probability: float = .50
    full_variant: str = "full"

    def __post_init__(self):
        if isinstance(self.n, bool) or not isinstance(self.n, Integral) or self.n < 1:
            raise ValueError("n must be a positive integer")
        if len(self.seeds) != self.n or len(set(self.seeds)) != self.n:
            raise ValueError("Seed count must equal frozen n and contain no duplicates")
        if any(isinstance(s, bool) or not isinstance(s, Integral) for s in self.seeds):
            raise ValueError("Every seed must be an integer")
        if len(self.claims) != 6 or len({c.name for c in self.claims}) != 6:
            raise ValueError("The frozen confirmatory family must contain six distinct claims")
        for spec in self.claims:
            if not spec.name or not spec.comparators or not spec.domains:
                raise ValueError("Each claim needs a name, comparator and domain")
            if len(set(spec.comparators)) != len(spec.comparators) or len(set(spec.domains)) != len(spec.domains):
                raise ValueError("Duplicate claim conditions are not permitted")
            if not np.isfinite(spec.reward_margin) or not 0 < spec.reward_margin <= 1:
                raise ValueError("Reward margins must be fixed, positive fractions")
        if not np.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0,1)")
        if not 0 <= self.absolute_alive_floor <= 1 or not 0 <= self.maximum_alive_loss <= 1:
            raise ValueError("Survival guards must be fractions in [0,1]")
        if self.null_win_probability != .5:
            raise ValueError("R9 tests the frozen null Pr(seed win)<=0.5")


DEFAULT_PLAN = StatisticalPlan()


def _fraction_array(value, name):
    array = np.asarray(value, dtype=float)
    if array.ndim < 1 or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name} needs nonempty, finite seed observations")
    if ((array < 0) | (array > 1)).any():
        raise ValueError(f"{name} must lie in [0,1]; do not fit normalization on test results")
    return array


def episode_endpoints(reward, alive_after, *, horizon=320):
    """Compute two separate endpoints over the entire fixed native horizon.

    Inputs must already contain every step, including absorbing zeros after
    death. A partial trace is an incomplete artifact, not a shorter denominator.
    Rewards have the environment's analytically defined [0,1] scale.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    reward = _fraction_array(reward, "reward")
    alive = _fraction_array(alive_after, "alive_after")
    if reward.ndim != 1 or alive.shape != reward.shape or len(reward) != horizon:
        raise ValueError("Both traces must be one-dimensional and span the fixed horizon")
    if not np.isin(alive, (0., 1.)).all():
        raise ValueError("Alive status must be binary at every post-transition step")
    if np.any(np.diff(alive) > 0):
        raise ValueError("Death must be absorbing within a native episode")
    dead = np.flatnonzero(alive == 0)
    if len(dead) and np.any(reward[dead[0]:] != 0):
        raise ValueError("Post-death absorbing steps must have reward zero")
    return {"reward_mean": float(reward.mean()), "alive_fraction": float(alive.mean()),
            "horizon": int(horizon), "terminal_alive": bool(alive[-1])}


def utility_criterion(full_reward, comparator_reward, full_alive, comparator_alive, *,
                      reward_margin, absolute_alive_floor=.80, maximum_alive_loss=.005,
                      expected_n=None):
    """Return component guards and one conjunctive boolean per independent seed.

    The first axis is seed. Every remaining axis indexes mandatory domains or
    comparators; none are averaged to turn a failed condition into a success.
    Missing/nonfinite observations raise instead of being silently discarded.
    """
    values = [_fraction_array(value, name) for value, name in zip(
        (full_reward, comparator_reward, full_alive, comparator_alive),
        ("full_reward", "comparator_reward", "full_alive", "comparator_alive"))]
    if len({v.shape for v in values}) != 1:
        raise ValueError("All four endpoint arrays must have identical shapes")
    if expected_n is not None and values[0].shape[0] != expected_n:
        raise ValueError("Observed seed count differs from the frozen plan")
    if not np.isfinite(reward_margin) or not 0 < reward_margin <= 1:
        raise ValueError("A positive prespecified reward margin is required")
    if not 0 <= absolute_alive_floor <= 1 or not 0 <= maximum_alive_loss <= 1:
        raise ValueError("Invalid survival guard")
    fr, cr, fa, ca = values
    reward_difference, alive_difference = fr-cr, fa-ca
    reward_pass = reward_difference >= reward_margin
    floor_pass = fa >= absolute_alive_floor
    noninferiority_pass = alive_difference >= -maximum_alive_loss
    condition_success = reward_pass & floor_pass & noninferiority_pass
    seed_success = condition_success.reshape(len(fr), -1).all(axis=1)
    return {"delta_reward": reward_difference, "delta_alive": alive_difference,
            "reward_margin_pass": reward_pass, "absolute_alive_floor_pass": floor_pass,
            "alive_noninferiority_pass": noninferiority_pass,
            "condition_success": condition_success, "seed_success": seed_success}


def exact_binomial_tail(wins, n):
    """Exact one-sided null tail at p=.5, computed from integer binomial sums."""
    if any(isinstance(v, bool) or not isinstance(v, Integral) for v in (wins,n)):
        raise ValueError("wins and n must be integers")
    if n < 1 or wins < 0 or wins > n:
        raise ValueError("Require n>=1 and 0<=wins<=n")
    return sum(comb(int(n), k) for k in range(int(wins), int(n)+1)) / (2**int(n))


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or ((values<0)|(values>1)).any():
        raise ValueError("p-values must be a finite one-dimensional vector in [0,1]")
    order = np.argsort(values, kind="stable")
    adjusted, running = np.empty_like(values), 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., float((len(values)-rank)*values[index])))
        adjusted[index] = running
    return adjusted


def conservative_power(*, n=80, family_size=6, alpha=.05,
                       probabilities=(.60,.65,.70,.75,.80)):
    if isinstance(family_size, bool) or not isinstance(family_size, Integral) or family_size < 1:
        raise ValueError("family_size must be a positive integer")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Invalid alpha")
    # Also validate n through the exact tail rather than silently rounding it.
    exact_binomial_tail(0,n)
    candidates = [k for k in range(n+1) if exact_binomial_tail(k,n) <= alpha/family_size]
    threshold = candidates[0] if candidates else None
    power = {}
    for p in probabilities:
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("Power scenarios must be probabilities")
        power[str(p)] = 0. if threshold is None else float(binom.sf(threshold-1,n,p))
    return {"n":int(n), "family_size":int(family_size), "alpha":float(alpha),
            "minimum_wins_first_holm_step":threshold,
            "null_tail_at_threshold":None if threshold is None else exact_binomial_tail(threshold,n),
            "power_by_joint_seed_win_probability":power,
            "scope":"Conservative first-Holm-step bound, not a power guarantee for each endpoint"}


def summarize(success_by_claim, *, seeds, plan=DEFAULT_PLAN):
    names = tuple(spec.name for spec in plan.claims)
    if set(success_by_claim) != set(names):
        raise ValueError("All six frozen hypotheses must be included, with no additions or omissions")
    if tuple(seeds) != tuple(plan.seeds):
        raise ValueError("Observed seeds or their order differ from the frozen plan")
    vectors = {}
    for name in names:
        raw = np.asarray(success_by_claim[name])
        if raw.shape != (plan.n,) or not np.isin(raw,(0,1)).all():
            raise ValueError(f"{name} requires exactly n complete binary seed outcomes")
        vectors[name] = raw.astype(bool)
    tails = [exact_binomial_tail(int(vectors[name].sum()),plan.n) for name in names]
    adjusted = holm_adjust(tails)
    rows = [{"claim":name,"n":plan.n,"wins":int(vectors[name].sum()),
             "seed_success":vectors[name].tolist(),"p_one_sided_exact":tails[i],
             "p_holm":float(adjusted[i]),"supported":bool(adjusted[i] <= plan.alpha)}
            for i,name in enumerate(names)]
    return {"estimand":"Probability that an independent seed meets all practical conditions",
            "seed_order":list(plan.seeds),"family_alpha":plan.alpha,"null_win_probability":.5,
            "multiplicity":"Holm over all six frozen claims","rows":rows,
            "power":conservative_power(n=plan.n,family_size=6,alpha=plan.alpha)}


def paired_mean_interval(differences, *, confidence=.95, resamples=20000,
                         bootstrap_seed=953900, expected_n=None):
    values = np.asarray(differences,dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("A complete finite vector of paired seed differences is required")
    if expected_n is not None and len(values) != expected_n:
        raise ValueError("Descriptive interval would omit planned seeds")
    if not 0 < confidence < 1 or isinstance(resamples,bool) or not isinstance(resamples,Integral) or resamples < 1000:
        raise ValueError("Invalid confidence or fewer than 1000 resamples")
    if isinstance(bootstrap_seed,bool) or not isinstance(bootstrap_seed,Integral) or bootstrap_seed < 0:
        raise ValueError("Invalid bootstrap stream")
    draws = np.random.default_rng(bootstrap_seed).integers(0,len(values),(resamples,len(values)))
    lower,upper = np.quantile(values[draws].mean(1),[(1-confidence)/2,1-(1-confidence)/2])
    return {"mean":float(values.mean()),"lower":float(lower),"upper":float(upper),
            "n_seeds":len(values),"confidence":confidence,"resamples":int(resamples),
            "bootstrap_seed":int(bootstrap_seed),
            "method":"Paired-seed percentile bootstrap; descriptive and not multiplicity adjusted"}


def compile_from_metrics(data: Mapping, *, reward_key="reward_mean", alive_key="alive_fraction",
                         variant_names=None, domain_names=None, plan=DEFAULT_PLAN):
    """Compile nested data[seed][domain][variant][metric] under a fixed plan.

    ``variant_names`` and ``domain_names`` map protocol roles to runner labels;
    they change names, never criteria. Seeds may be integer keys or decimal
    strings. Primary computations retain each domain/comparator condition.
    """
    variant_names,domain_names = dict(variant_names or {}),dict(domain_names or {})
    normalized = {}
    for key, value in data.items():
        if isinstance(key,Integral) and not isinstance(key,bool):
            seed = int(key)
        elif isinstance(key,str) and key.isdecimal():
            seed = int(key)
        else:
            raise ValueError("Seed keys must be integers or decimal strings")
        if seed in normalized:
            raise ValueError("Duplicate seed after normalizing integer/string keys")
        normalized[seed] = value
    if set(normalized) != set(plan.seeds):
        raise ValueError("Dataset must contain exactly the frozen independent seeds")
    components,successes = {},{}
    for spec in plan.claims:
        conditions = [(domain,comparator) for domain in spec.domains for comparator in spec.comparators]
        arrays = [np.empty((plan.n,len(conditions))) for _ in range(4)]
        for i,seed in enumerate(plan.seeds):
            for j,(domain,comparator) in enumerate(conditions):
                domain_key = domain_names.get(domain,domain)
                full_key = variant_names.get(plan.full_variant,plan.full_variant)
                comp_key = variant_names.get(comparator,comparator)
                try:
                    full = normalized[seed][domain_key][full_key]
                    other = normalized[seed][domain_key][comp_key]
                    for array,value in zip(arrays,(full[reward_key],other[reward_key],full[alive_key],other[alive_key])):
                        array[i,j] = value
                except (KeyError,TypeError,ValueError) as error:
                    raise ValueError(f"Missing/invalid metric for seed={seed}, domain={domain}, comparator={comparator}") from error
        criterion = utility_criterion(*arrays,reward_margin=spec.reward_margin,
            absolute_alive_floor=plan.absolute_alive_floor,maximum_alive_loss=plan.maximum_alive_loss,expected_n=plan.n)
        successes[spec.name] = criterion["seed_success"]
        components[spec.name] = {"conditions":[{"domain":d,"comparator":c} for d,c in conditions],
            "reward_margin":spec.reward_margin,"absolute_alive_floor":plan.absolute_alive_floor,
            "maximum_alive_loss":plan.maximum_alive_loss,
            **{key:value.tolist() for key,value in criterion.items()},
            "full_reward":arrays[0].tolist(),"comparator_reward":arrays[1].tolist(),
            "full_alive":arrays[2].tolist(),"comparator_alive":arrays[3].tolist()}
    return {**summarize(successes,seeds=plan.seeds,plan=plan),"components":components,
            "input_metric_keys":{"reward":reward_key,"alive":alive_key},
            "variant_names":variant_names,"domain_names":domain_names}
