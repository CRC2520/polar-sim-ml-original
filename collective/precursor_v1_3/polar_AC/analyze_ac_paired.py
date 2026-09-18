"""Seed-paired sensitivity analysis for the preserved A/C precursor study."""

import glob
import json

import numpy as np
from scipy import stats


def load(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as stream:
            rows.extend(json.load(stream))
    return np.asarray([np.mean(row["prop"][20:25]) for row in rows])


def exact_wilcoxon(x, y=None, alternative="two-sided"):
    return stats.wilcoxon(x, y, alternative=alternative, method="exact").pvalue


a_base = load("AC_A_AMBOS-base_*.json")
a_equal = load("AC_A_AMBOS-eq_*.json")
c_base = load("AC_A_CONF-base_*.json")
c_equal = load("AC_A_CONF-eq_*.json")
a_interaction = (a_equal - a_base) - (c_equal - c_base)

print("A: paired success-imitation effect")
print(f"mean={np.mean(a_equal-a_base):+.6f} p_exact_one_sided={exact_wilcoxon(a_equal, a_base, 'greater'):.7f}")
print("A: paired difference-in-differences")
print(f"mean={np.mean(a_interaction):+.6f} p_exact_one_sided={exact_wilcoxon(a_interaction, alternative='greater'):.7f}")

base = load("AC_C_BASE_*.json")
print("C: paired architecture sensitivities")
for variant in ("SIN_ANCLA", "HORIZONTE_CORTO", "SIN_DIRIGIDA"):
    values = load(f"AC_C_{variant}_*.json")
    print(
        f"{variant} mean={np.mean(values-base):+.6f} "
        f"p_exact_two_sided={exact_wilcoxon(values, base):.7f}"
    )
