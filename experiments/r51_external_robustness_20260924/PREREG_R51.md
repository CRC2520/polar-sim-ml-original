# R51 — fresh external robustness battery

Status: prospective before scientific development seeds are opened.
Date: 24 September 2026, America/Lima.

## Starting boundary

R37/R38 showed that the previous recurrent observer did not satisfy its frozen
absolute robustness criterion under noisy partial observations. Those panels
remain adverse and are not reused for tuning.

R51 evaluates a new external robustness question on **different POPGym tasks**
and with frozen functional D+C+R controllers. It does not change POLAR Core
v1.1:

[
mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]

## External source

Repository: `proroklab/popgym`

Exact source commit:

`e397e5eac9965f9963d18c9f455cd1983bca14fb`

No environment source is modified.

## Task families

Development:
- `AutoencodeMedium`
- `RepeatPreviousMedium`

Reserved confirmation:
- `AutoencodeHard`
- `RepeatPreviousHard`

These tasks are not the PositionOnlyCartPole noisy panel used by R37/R38.

## Functional organization

### Autoencode

The intact controller:
- differentiates the four observed suit identities (D);
- stores the observed sequence across the WATCH phase (C);
- uses the correct sequence-position relation during PLAY (R).

The external historical implementation requires LIFO/reverse replay.

### RepeatPrevious

The intact controller:
- differentiates suit identities (D);
- retains the observation sequence over the episode (C);
- retrieves the identity at the externally specified temporal lag (R).

Medium uses lag 32. Hard uses lag 64.

No environment hidden state is read by either controller.

## Frozen causal lesions

### NO_D_COLLAPSE

All observed suit identities are collapsed to internal identity 0 before
storage.

### NO_C_RESET

Persistent sequence memory is erased at each input event, retaining only the
current item.

### WRONG_R

- Autoencode: use FIFO relation instead of the correct LIFO relation.
- RepeatPrevious: use lag (k-1) instead of the externally defined lag (k).

### GENERIC_ISO

Apply the fixed bijection

[
phi(i)=3-i
]

to internal suit identities and invert the bijection at the native action
boundary.

On identical external observations, GENERIC_ISO must produce exactly the same
actions as FULL.

## Observation perturbations

Perturbations are applied outside the third-party environment and do not alter
its transition/reward code.

### CLEAN

No perturbation.

### NOISE_05

With probability 0.05 per input symbol, replace the symbol by one of the other
three identities, chosen from a frozen per-episode RNG.

### OCCLUSION_10

With probability 0.10 per input symbol, hide the symbol. The controller knows
that the symbol is unavailable but receives no oracle replacement.

### STALE_05

With probability 0.05, replace the current symbol with the immediately previous
raw symbol, representing one-step observation delay/staleness.

### SEMANTIC_SHIFT_HALF

After half of the input sequence, apply the fixed involution
(phi(i)=3-i) to observed symbols without changing native action semantics.

This is an unannounced observation-semantic regime shift.

## Statistical unit

Each statistical seed contains two paired episodes per task, perturbation and
causal condition.

All FULL / lesion / GENERIC_ISO conditions use the same external episode seeds
and the same perturbation RNG streams.

Scored-step accuracy is:

- Autoencode: accuracy over non-zero-reward PLAY steps;
- RepeatPrevious: accuracy over non-zero-reward steps after the lag becomes
  active.

## Per-seed endpoints

For each task and perturbation:

[
A_{full}
]

and causal effects:

[
Delta_D=A_{full}-A_{noD},
]

[
Delta_C=A_{full}-A_{noC},
]

[
Delta_R=A_{full}-A_{wrongR}.
]

For CLEAN:

[
G_{iso}=|A_{full}-A_{iso}|
]

plus exact greedy action agreement.

## Frozen robustness criteria

For **each task** within a statistical seed:

### Absolute capability

- CLEAN FULL accuracy >= **0.98**;
- NOISE_05 FULL accuracy >= **0.82**;
- OCCLUSION_10 FULL accuracy >= **0.82**;
- STALE_05 FULL accuracy >= **0.82**;
- SEMANTIC_SHIFT_HALF FULL accuracy >= **0.45**.

### Causal retention

CLEAN:
- D effect >= **+0.55**;
- C effect >= **+0.55**;
- R effect >= **+0.55**.

Each mild perturbation (NOISE_05, OCCLUSION_10, STALE_05):
- D effect >= **+0.35**;
- C effect >= **+0.35**;
- R effect >= **+0.35**.

SEMANTIC_SHIFT_HALF:
- D effect >= **+0.15**;
- C effect >= **+0.15**;
- R effect >= **+0.15**.

### Isomorphic control

On CLEAN:
- iso accuracy gap <= **1e-12**;
- exact action agreement >= **1 - 1e-12**.

A statistical seed passes only if **both tasks** satisfy every frozen criterion.

## Development

Seeds:

`2216001–2216016`

Development authorizes confirmation only if:
- at least **13/16** seed guards pass;
- all median task/perturbation capability and causal-retention criteria pass;
- CLEAN isomorphic criteria pass exactly for both tasks.

Development resolutions:

- `R51_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `R51_DEVELOPMENT_FAIL_NO_CONFIRM`

## Reserved confirmation

Seeds:

`2217001–2217032`

Confirmation uses the Hard task variants and remains unopened until development
passes.

Final PASS requires:
- at least **28/32** seed guards;
- all frozen median capability/causal-retention criteria;
- exact CLEAN isomorphic equivalence.

Final resolutions:

- `R51_EXTERNAL_ROBUSTNESS_BATTERY_PASS`
- `R51_EXTERNAL_ROBUSTNESS_BATTERY_FAIL`

## Interpretation

A PASS supports bounded robustness of D+C+R-like sequence organization under
the tested observation perturbations across two external POMDP families.

A FAIL preserves the external-robustness gap.

Neither result establishes:
- neural learnability;
- global minimality;
- E6b;
- E7;
- higher-order metacognition/planning integration;
- AGI/ASI;
- phenomenal consciousness.
