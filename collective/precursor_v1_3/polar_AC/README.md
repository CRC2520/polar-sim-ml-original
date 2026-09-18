# Collective A/C precursor study (v1.3)

This directory preserves the exact A/C source files and raw JSON outputs from
`POLAR_collective_viability_v1.3.zip` (SHA-256
`6e4c4bd42e523888926cc007752140c293edd2a59cbf9b7ce427c03f9d1c083d`).

The study is evidence preceding the strengthened confirmatory design in
`docs/COLLECTIVE_BRIDGE_PREREGISTRATION.md`; it is not an execution of that
design. Important differences are eight seeds per cell, one initial composition
per study, final restraint as the primary outcome, an inherited binary trait,
and no signed-plus-intensity coordinate control.

`analyze_ac.py` is the preserved original analysis. `analyze_ac_paired.py`
adds the seed-paired exact Wilcoxon sensitivity analysis without replacing or
rewriting the registered analysis.

The equalization intervention changes the fitness signal used by
success-biased imitation. It does not equalize realised survival or ecological
payoff. Under conformity that fitness signal is not consulted, making the
conformity cell a structural negative control.

Run from this directory:

```bash
python analyze_ac.py
python analyze_ac_paired.py
```

No result in this directory measures consciousness.
