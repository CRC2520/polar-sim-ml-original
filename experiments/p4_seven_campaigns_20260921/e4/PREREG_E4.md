# E4 PREREG — Naturalistic Organizational Necessity

Freeze: 2026-09-21 America/Lima.
Development seeds excluded: 1341001–1341104.
Untouched confirmatory seeds: 1342001–1342012.

Families are standard dynamical equations chosen independently of POLAR semantics: SIR epidemic dynamics, Lotka–Volterra predator–prey, bimolecular mass-action reaction, and two-species competition.

Each family is learned from only 64 samples and evaluated under an OOD parameter shift.

Comparators:
- INTEGRATED: generic full degree-2 interactions;
- FACTORIZED: degree-1 through degree-5 univariate features only, with feature count >= integrated in every family;
- MONOLITHIC: generic two-hidden-layer MLP.

Frozen median criteria across confirmatory seeds:
- median integrated/factorized OOD-MSE ratio <=0.05
- median of each seed's worst-family integrated/factorized ratio <=0.20
- integrated beats factorized in all 4 families
- factorized feature budget >= integrated in all families
- median integrated/monolithic OOD-MSE ratio <=0.80
- integrated beats monolithic in >=3/4 families

At least 9/12 seeds must have 4/4 integrated wins over factorized, factorized-ratio median <=0.10, monolithic-ratio median <=1.0, >=2/4 wins over monolithic, and larger/equal factorized feature budget.

A PASS supports the need for cross-variable organization and its sample efficiency in these independent equations. It is not a POLAR-specific superiority result; generic relational models instantiate the same functional class.
