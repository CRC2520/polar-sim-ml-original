# R51 — fresh external robustness battery for D+C+R

Status: prospective before R51 scientific development seeds are opened.
Date: 25 September 2026, America/Lima.

## Starting boundary

R37/R38 failed the previous external observation-robustness gate. Those panels
are frozen and must not be tuned again.

R47 later established a same-program external D+C+R conjunction on unmodified
POPGym ConcentrationHard, while R48 failed to establish reward-only neural
relation learnability under its frozen CountRecallMedium design.

R51 asks a separate robustness question:

> Does a persistent D+C+R organization retain absolute capability and selective
> D/C/R causal dependence when observations from unmodified third-party tasks
> are made unreliable by prospectively frozen sensor-interface perturbations?

R51 is not a neural-learning experiment and does not rescue R48. It uses
functional persistent controllers so that observation robustness can be tested
without confounding it with optimization failure.

POLAR Core v1.1 remains unchanged:

[
mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]

## External environment source

Repository:

`proroklab/popgym`

Exact source:

`e397e5eac9965f9963d18c9f455cd1983bca14fb`

The environment source is not modified.

Two externally defined tasks are used:

1. `ConcentrationHard`
2. `CountRecallHard`

## Frozen observation perturbations

Perturbations are applied only at the agent observation interface. Native
environment state, reward, transition, termination and action spaces remain
unchanged.

Every condition within one episode receives the same deterministic perturbation
schedule for that task/episode seed.

### CLEAN

Native observation.

### SUBSTITUTE10

Each active identity observation is independently substituted with probability
0.10 by a uniformly selected different native identity.

For Concentration this applies to each revealed card identity.
For CountRecall it applies independently to dealt and queried identities.

### STALE_BURST

Two deterministic sensor-staleness bursts are generated per episode from the
episode seed.

During a burst:
- Concentration receives the previous perceived revealed-card identity;
- CountRecall receives the previous perceived complete two-identity
  observation.

Burst length is four observation events.

### MIXED

A milder mixed corruption:
- identity substitution probability 0.05;
- two stale bursts of length two.

No perturbation parameter is tuned after scientific development begins.

## Task A — ConcentrationHard controller

The intact controller is the R47-style online position↔identity memory
controller, rewritten in the R51 runner.

Conditions:

- `FULL_DCR`
- `NO_D_COLLAPSE`: all perceived ranks map to one internal identity;
- `NO_C_PAIR_RESET`: persistent hidden card/location memory is cleared after
  every pair attempt while current pair state remains;
- `WRONG_R_CYCLIC`: lookup uses ((s+1)mod13);
- `GENERIC_ISO`: exact internal bijection (phi(s)=12-s).

Primary capability endpoint:
- completion fraction = successful matches / 26.

## Task B — CountRecallHard controller

The intact controller maintains an online count vector indexed by perceived
identity. At each observation it:
1. updates the count for the dealt identity;
2. retrieves the count indexed by the current query identity;
3. emits that count as the native action.

Conditions:

- `FULL_DCR`
- `NO_D_COLLAPSE`: dealt/query identities map to one internal identity;
- `NO_C_STEP_RESET`: the persistent count vector is cleared before every
  observation;
- `WRONG_R_QUERY_CYCLE`: retrieval uses ((q+1)mod13);
- `GENERIC_ISO`: both identities are recoded by (phi(s)=12-s).

Primary capability endpoint:
- action accuracy = fraction of steps with positive native reward.

## Per-cell causal endpoints

For each task, perturbation and statistical seed:

[
Delta_D=M_{FULL}-M_{NO_D}
]

[
Delta_C=M_{FULL}-M_{NO_C}
]

[
Delta_R=M_{FULL}-M_{WRONG_R}
]

where (M) is completion for Concentration and accuracy for CountRecall.

Also record:
- isomorphic metric gap;
- exact FULL↔GENERIC_ISO action agreement;
- robustness ratio (M_{perturbed}/M_{clean}).

## Frozen cell gates

### CLEAN cells

For each task:
- FULL metric >= 0.90;
- D effect >= +0.20;
- C effect >= +0.30;
- R effect >= +0.20;
- iso gap <= 1e-12;
- iso action agreement >= 1 - 1e-12.

### Perturbed cells

For each task and each of SUBSTITUTE10 / STALE_BURST / MIXED:
- FULL metric >= 0.55;
- robustness ratio >= 0.55;
- D effect >= +0.10;
- C effect >= +0.15;
- R effect >= +0.10;
- iso gap <= 1e-12;
- iso action agreement >= 1 - 1e-12.

These thresholds apply identically to both task families.

## Statistical unit

Each statistical seed contains:
- 3 paired episodes per task;
- all four perturbations;
- all five D/C/R/isomorphic controller conditions.

Episode seeds are deterministically derived from the statistical seed and task
index. Perturbation schedules are derived independently from the same episode
seed and perturbation label.

## Development

Seeds:

`2216001–2216012`

Development authorizes confirmation only if:

1. every one of the eight task×perturbation cells passes its median frozen cell
   gates; and
2. at least **10/12** statistical seeds pass every one of their eight cells.

Development resolutions:

- `R51_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `R51_DEVELOPMENT_FAIL_NO_CONFIRM`

## Reserved confirmation

Seeds:

`2217001–2217032`

They remain unopened unless development passes.

Confirmation requires:

1. every task×perturbation cell passes its median frozen gate;
2. at least **26/32** statistical seeds pass every cell.

Final resolutions:

- `R51_EXTERNAL_ROBUSTNESS_PASS_SAME_PROGRAM`
- `R51_EXTERNAL_ROBUSTNESS_FAIL`

## Interpretation boundaries

A PASS supports bounded same-program observation robustness of functional
persistent D+C+R organization across the two tested external POPGym tasks and
the preregistered corruption family.

It does not establish:
- reward-only neural learnability;
- general robustness to arbitrary corruptions;
- biological robustness;
- global minimality;
- POLAR algorithmic superiority;
- E6b;
- E7;
- identity/selfhood;
- consciousness.

A FAIL is retained and does not authorize tuning on the confirmatory panel.
