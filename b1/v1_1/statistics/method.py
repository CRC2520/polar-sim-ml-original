"""Fixed-sample, bounded paired-mean inference, independent of Polar tasks.

Theorem 4 of Maurer and Pontil (2009), rescaled to [lower, upper].
Two tails each receive alpha/(2*family_size). No optional stopping.
"""
from math import ceil, isfinite, log, sqrt

ALPHA = 0.05
FAMILY_SIZE = 9
DELTA = 1 / 64
EPSILON = 1 / 128
TARGET_HALF_WIDTH = 1 / 256
DEVELOPMENT_BUNDLES = 32
PLANNING_ENDPOINTS = 6
DEVELOPMENT_ERROR = 0.05
PRECISION_ERROR = 0.05
VARIANCE_FLOOR = 1 / 64


def radius(n, sample_variance, support=(-1.0, 1.0), alpha=ALPHA,
           family_size=FAMILY_SIZE):
    if not isinstance(n, int) or isinstance(n, bool) or n < 2:
        raise ValueError("At least two independent bundles are required")
    if not isfinite(sample_variance) or sample_variance < 0:
        raise ValueError("Invalid unbiased sample variance")
    lower, upper = support
    if not all(isfinite(x) for x in support) or lower >= upper:
        raise ValueError("Invalid support")
    if not 0 < alpha < 1 or not isinstance(family_size, int) or family_size < 1:
        raise ValueError("Invalid error allocation")
    t = log(4 * family_size / alpha)
    return sqrt(2 * sample_variance * t / n) + 7 * (upper - lower) * t / (3 * (n - 1))


def interval(values, support=(-1.0, 1.0), *, required_n=None):
    values = tuple(values)
    n = len(values)
    if required_n is not None and n != required_n:
        raise ValueError("Collection incomplete or sample size differs from fixed N")
    if any(not isfinite(x) or not support[0] <= x <= support[1] for x in values):
        raise ValueError("Missing, non-finite, or out-of-support bundle outcome")
    if n < 2:
        raise ValueError("Insufficient independent bundles")
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    half = radius(n, variance, support)
    return {"n": n, "mean": mean, "sample_variance": variance,
            "half_width": half, "lower": max(support[0], mean - half),
            "upper": min(support[1], mean + half)}


def utility_status(ci, *, admissible=True):
    """Precision is a prerequisite; inference never borrows planning variance."""
    if not admissible or ci["half_width"] > TARGET_HALF_WIDTH:
        return "inconclusive"
    if ci["upper"] < -DELTA:
        return "improved"
    if ci["lower"] > DELTA:
        return "worsened"
    if ci["lower"] > -EPSILON and ci["upper"] < EPSILON:
        return "equivalent"
    return "inconclusive"


def positive_gates(*, instrument_pass, mechanism_kappa1_supported,
                   mechanism_kappa2_equivalent, source_causal_admissible,
                   comparator_status, context_supported, pilot="P7"):
    """Gatekeeping restricts positive claims; negative reports remain visible.

    Source joint manipulation must not be re-labelled Gamma-specific causality.
    source_causal_admissible includes the frozen joint-manipulation guard.
    """
    eligible = pilot == "P7"
    mechanism = bool(eligible and instrument_pass and source_causal_admissible
                     and mechanism_kappa1_supported and mechanism_kappa2_equivalent)
    utility = bool(mechanism and all(comparator_status.get(c) == "improved"
                                    for c in ("C0", "C4")))
    pairing = bool(utility and all(comparator_status.get(c) == "improved"
                                  for c in ("C0", "C2", "C3", "C4")))
    return {"H_mechanism": mechanism, "H_utility": utility,
            "H_pairing": pairing, "H_context": bool(pairing and context_supported)}


def planning_second_moment_upper(nonzero_counts, n=DEVELOPMENT_BUNDLES):
    """Exact simultaneous zero-event bound; otherwise safe worst-case fallback.

    For X in [-1,1], E[X^2] <= Pr(X != 0). Zero successes in n trials gives
    the one-sided exact bound 1-delta**(1/n). Failure to observe a zero-only
    development sample triggers a distribution-free upper of 1, not a fitted
    convenient variance. This rule is locked before final seed generation.
    """
    if len(nonzero_counts) != PLANNING_ENDPOINTS or n < 1:
        raise ValueError("Unexpected calibration family")
    if any(not isinstance(k, int) or not 0 <= k <= n for k in nonzero_counts):
        raise ValueError("Invalid nonzero count")
    bound = (1 - (DEVELOPMENT_ERROR / PLANNING_ENDPOINTS) ** (1 / n)
             if all(k == 0 for k in nonzero_counts) else 1.0)
    return max(VARIANCE_FLOOR, bound)


def planning_variance(n, second_moment_upper):
    # X^2 in [0,1]; Hoeffding and union over the six tight-precision endpoints.
    # s^2 <= n/(n-1) * mean(X^2), independently of the unknown mean.
    upper = second_moment_upper + sqrt(log(PLANNING_ENDPOINTS / PRECISION_ERROR) / (2 * n))
    return n / (n - 1) * min(1.0, upper)


def fixed_n(second_moment_upper):
    if not VARIANCE_FLOOR <= second_moment_upper <= 1:
        raise ValueError("Invalid planning second-moment upper")
    lower, upper = 2, 2
    def sufficient(n):
        tight = radius(n, planning_variance(n, second_moment_upper)) <= TARGET_HALF_WIDTH
        # Popoviciu plus unbiased sample correction guarantees these broad margins.
        source = radius(n, n / (n - 1), (-1.0, 1.0)) <= 0.125
        context = radius(n, 4 * n / (n - 1), (-2.0, 2.0)) <= 0.125
        return tight and source and context
    while not sufficient(upper):
        upper *= 2
    while lower < upper:
        mid = (lower + upper) // 2
        if sufficient(mid):
            upper = mid
        else:
            lower = mid + 1
    return lower


def v1_n():
    return ceil(4 * log(2 * 27 / ALPHA) / (2 * TARGET_HALF_WIDTH ** 2))
