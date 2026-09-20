"""Frozen six-claim seed-robustness inference, separate from mean effect CIs.

Scientific margins and within-seed conjunctions live in the design documents.
A success vector contains one boolean per independent final seed. Its null is
Pr(seed meets the full prespecified practical criterion) <= 0.5, not a zero
mean effect. Ties and missing/invalid outcomes are not successes. An invalid
campaign is additionally flagged and cannot be reported as complete.
"""
from __future__ import annotations
import math
import numpy as np

CLAIMS = ('Q1', 'Q2', 'E_credit', 'E_utility', 'T_utility', 'I_generic')


def binomial_upper(wins, n, probability=.5):
    if isinstance(wins, bool) or isinstance(n, bool):
        raise ValueError('Counts must be integers')
    if int(wins) != wins or int(n) != n or not 0 <= wins <= n or n < 1:
        raise ValueError('Require integer counts 0 <= wins <= n and n >= 1')
    if not 0 <= probability <= 1:
        raise ValueError('Probability must be in [0,1]')
    return min(1., math.fsum(math.comb(int(n), k)*probability**k*
                            (1-probability)**(int(n)-k)
                            for k in range(int(wins), int(n)+1)))


def holm_adjust(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Require a nonempty finite vector')
    if np.any((values < 0) | (values > 1)):
        raise ValueError('P-values must lie in [0,1]')
    order = np.argsort(values, kind='stable')
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(1., np.maximum.accumulate(
        values[order]*(len(values)-np.arange(len(values)))))
    return adjusted


def confirmation_family(successes, *, expected_n=30):
    if set(successes) != set(CLAIMS):
        raise ValueError('All six prespecified claims must be supplied')
    records = []
    for claim in CLAIMS:
        vector = np.asarray(successes[claim])
        if vector.dtype.kind != 'b' or vector.shape != (expected_n,):
            raise ValueError(f'{claim}: require {expected_n} explicit seed booleans')
        wins = int(vector.sum())
        records.append({'claim': claim, 'wins': wins, 'n': expected_n,
                        'seed_success': vector.tolist(),
                        'p_one_sided_exact': binomial_upper(wins, expected_n)})
    adjusted = holm_adjust([row['p_one_sided_exact'] for row in records])
    for row, p in zip(records, adjusted):
        row['p_holm'] = float(p)
        row['supported_at_family_alpha_0_05'] = bool(p <= .05)
    return {'estimand': 'probability that a seed meets its full practical criterion',
            'null_probability_upper_bound': .5, 'family_alpha': .05,
            'multiplicity': 'Holm step-down across all six prespecified claims',
            'rows': records}


def mean_interval(differences, *, confidence=.95, resamples=20000, bootstrap_seed=949000):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError('Paired seed differences must be a finite vector of length >=2')
    if not 0 < confidence < 1 or resamples < 1000:
        raise ValueError('Invalid confidence or too few bootstrap resamples')
    rng = np.random.default_rng(bootstrap_seed)
    draws = values[rng.integers(0, len(values), size=(resamples, len(values)))].mean(axis=1)
    tail = (1-confidence)/2
    low, high = np.quantile(draws, (tail, 1-tail))
    return {'mean': float(values.mean()), 'lower': float(low), 'upper': float(high),
            'confidence': confidence, 'n_seeds': len(values),
            'method': 'paired-seed percentile bootstrap; descriptive, not multiplicity adjusted',
            'resamples': resamples, 'bootstrap_seed': bootstrap_seed}


def conservative_power(n=30, family_size=6, alpha=.05, probabilities=(.6,.7,.8,.9)):
    critical = next((k for k in range(n+1) if binomial_upper(k, n) <= alpha/family_size), None)
    return {'n': n, 'family_size': family_size, 'alpha': alpha,
            'minimum_wins_bonferroni': critical,
            'description': 'Conservative first-Holm-step bound; not a power guarantee',
            'power_by_true_seed_success_probability': {
                str(p): 0. if critical is None else binomial_upper(critical, n, p)
                for p in probabilities}}
