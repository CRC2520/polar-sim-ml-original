# E2 PREREG — Polarity Discovery

Freeze: 2026-09-21 America/Lima.

Development seeds excluded: 1321001–1321104.
Untouched confirmatory seeds: 1322001–1322012.

The environment contains three hidden continuous causal axes embedded by independent random orthogonal input/output mixtures. No pair labels, pole signs, historical catalogue, or axis identities are provided to the learner.

The learner is a generic 3-dimensional nonlinear action bottleneck trained only to predict factual consequences. After training, its learned action directions are compared post hoc to the hidden generating axes up to permutation and sign.

Frozen median criteria:
- mean recovered-axis alignment >=0.98
- minimum recovered-axis alignment >=0.95
- held-out consequence MSE <=0.003
- cosine between predicted consequences of opposite learned extremes <=-0.98
- context-dependent sign-control ratio >=0.98
- alignment gain over equal-size random axes >=0.25

At least 9/12 seeds must satisfy mean alignment >=0.95, min alignment >=0.90, gain >=0.18 and MSE <=0.006.

A PASS supports discovery of bipolar causal directions when such axes genuinely generate consequences. It does not establish universal polarity or that all domains contain low-rank bipolar structure.
