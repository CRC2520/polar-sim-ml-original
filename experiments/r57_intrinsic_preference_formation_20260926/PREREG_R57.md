# R57 — bounded intrinsic preference formation from self-maintenance consequences

Status: prospective before scientific development seeds are opened.
Date: 26 September 2026, America/Lima.

## Motivation

R55 does not establish strong long-horizon autonomy under its frozen causal
endpoint. R56 does not establish the full social-interaction conjunction.

R57 tests a narrower next frontier: whether an agent can **learn stable internal
preferences** from the consequences of experience without receiving an external
scalar reward.

This experiment does not test moral values, human motivation, free will or
consciousness.

## Operational definition

R57 uses the term **intrinsic preference formation** only in this bounded sense:

> A neutral internal value table starts at zero. Through experience, cue/action
> values are updated from internally computed changes in the agent's own
> homeostatic viability. Those learned values later guide novel, reward-free
> choices; deleting, misbinding or transplanting the value state causally
> disrupts those choices.

The homeostatic variables and generic viability function are part of the
agent's design. R57 therefore does **not** claim that ultimate goals emerge from
nothing.

## Internal body state

Two homeostatic dimensions:

- energy;
- integrity.

Both lie in [0,1].

Main-agent target:

[
h^*_{main}=[0.75,0.75].
]

A donor body has a different target:

[
h^*_{donor}=[0.25,0.25].
]

Internal instantaneous viability is:

[
U(h;h^*)=-|h-h^*|_2^2.
]

No external reward is supplied.

## Experience contexts

Three main-agent state contexts:

1. energy deficit:
   ([0.45,0.75]);
2. integrity deficit:
   ([0.75,0.45]);
3. joint deficit:
   ([0.52,0.52]).

Donor contexts are symmetric around the donor target:

1. energy excess:
   ([0.55,0.25]);
2. integrity excess:
   ([0.25,0.55]);
3. joint excess:
   ([0.48,0.48]).

Small preregistered Gaussian state noise is added during experience.

## Six neutral cue classes

Before learning, all cue values are exactly zero.

Mean homeostatic consequences:

- cue 0: [+0.15,  0.00]
- cue 1: [ 0.00, +0.15]
- cue 2: [+0.10, +0.10]
- cue 3: [-0.15,  0.00]
- cue 4: [ 0.00, -0.15]
- cue 5: [-0.10, -0.10]

Outcome noise is Gaussian and seed-paired across conditions.

The cue IDs themselves have no externally assigned reward or valence.

## Learning phase

3,600 forced experience trials.

On each trial:

1. a context is sampled;
2. a cue is sampled uniformly;
3. the cue produces its noisy homeostatic consequence;
4. the agent observes before/after internal state;
5. it computes:
   [
   Delta U=U(h_{after})-U(h_{before});
   ]
6. context-specific cue value is updated:
   [
   V_{c,k}leftarrow(1-alpha)V_{c,k}+alphaDelta U
   ]
   with frozen (alpha=0.08).

There is:
- no external reward;
- no teacher-provided cue label;
- no oracle ranking;
- no free-choice behavior during learning.

The donor value table is learned independently from the same cue consequences
but the donor target/contexts.

## Washout

After learning, the main agent undergoes 800 steps with:

- internal-state observations;
- no cue-outcome pairing;
- no external reward.

Value magnitudes receive a fixed passive decay of 0.9998 per washout step.

No value update from outcome occurs during washout.

This tests persistence of learned ranking without reinforcement.

## Reward-free novel-choice probe

The probe phase provides **no outcome and no reward**.

For each context, all 15 unordered pairs of the six cues are presented multiple
times in a seed-randomized order.

The agent chooses the cue with the higher stored context-specific value.

The evaluator alone computes which cue has the higher expected homeostatic
improvement under the frozen mean cue consequences.

Pairs tied within (10^{-9}) expected utility are excluded prospectively from
accuracy.

The exact cue pair itself was never presented during learning; learning exposed
only individual cue consequences.

## Frozen conditions

### FULL_VALUE

Main agent's learned value table after washout.

### VALUE_RESET

Same probe and body, but all learned values are set to zero before probe.

Tie-breaking is deterministic by smaller cue ID.

### PERMUTED_BINDING

Main learned value table is retained, but cue→value lookup uses:

[
k'=(k+1)mod 6.
]

### DONOR_VALUE_TRANSPLANT

Main body/probe receives the independently learned donor value table.

### OUTCOME_BLIND

During the experience phase, cue values remain zero because the agent is not
allowed to use before/after homeostatic consequence.

All other exposure counts and probe mechanics are identical.

### GENERIC_ISO

Cue identities are recoded by:

[
phi(k)=5-k
]

and the learned value table is transformed by the exact conjugate
permutation. External choices are mapped back to native cue IDs.

Behavior must be exactly equivalent to FULL_VALUE.

### SHAM_STATE

FULL_VALUE plus an unused state matrix with the same dimensions as the learned
value table.

## Endpoints

Per seed:

- FULL reward-free pairwise choice accuracy;
- FULL post-washout pairwise choice accuracy;
- context-sensitive reversal count:
  number of cue pairs whose preferred cue differs across at least two contexts;
- Spearman rank correlation between learned cue values and expected intrinsic
  utility, averaged across contexts;
- VALUE_RESET accuracy;
- PERMUTED_BINDING accuracy;
- DONOR_VALUE_TRANSPLANT accuracy;
- OUTCOME_BLIND accuracy;
- GENERIC_ISO accuracy/action agreement;
- SHAM accuracy gap.

Primary causal effects:

[
Delta_{memory}=A_{FULL}-A_{RESET}
]

[
Delta_{binding}=A_{FULL}-A_{PERMUTED}
]

[
Delta_{self}=A_{FULL}-A_{DONOR}
]

[
Delta_{outcome}=A_{FULL}-A_{OUTCOME_BLIND}.
]

## Seed-level guard

A seed passes only if:

- FULL post-washout choice accuracy >= **0.90**;
- learned-value / expected-utility mean Spearman >= **0.85**;
- context-sensitive reversal count >= **3**;
- memory effect >= **+0.30**;
- binding effect >= **+0.25**;
- donor-transplant effect >= **+0.20**;
- outcome effect >= **+0.30**;
- GENERIC_ISO accuracy gap <= **1e-12**;
- GENERIC_ISO action agreement >= **1 - 1e-12**;
- SHAM accuracy gap <= **1e-12**.

## Development

Seeds:

`2266001–2266016`.

Development authorizes confirmation only if:

- at least 13/16 seed guards pass;
- median FULL accuracy >=0.93;
- median Spearman >=0.90;
- median memory effect >=+0.35;
- median binding effect >=+0.30;
- median donor effect >=+0.25;
- median outcome effect >=+0.35;
- median iso/sham gaps <=1e-12.

Labels:

- `R57_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `R57_DEVELOPMENT_FAIL_NO_CONFIRM`.

## Reserved confirmation

Seeds:

`2267001–2267032`.

Remain unopened unless development passes.

Final PASS requires:

- at least 28/32 seed guards;
- the same median criteria as development.

Final labels:

- `R57_BOUNDED_INTRINSIC_PREFERENCE_FORMATION_PASS`
- `R57_BOUNDED_INTRINSIC_PREFERENCE_FORMATION_FAIL`.

## Valid interpretation of PASS

A PASS supports only:

> stable context-sensitive preferences over initially neutral cues are learned
> from internally computed homeostatic consequences, persist without reward,
> generalize to novel reward-free pairwise choices, and causally depend on the
> agent's own learned value state and correct cue-value binding.

It does not establish:

- moral values;
- normative ethics;
- spontaneous creation of ultimate goals;
- emotions;
- desire in the phenomenal sense;
- free will;
- consciousness;
- AGI/ASI;
- E6b;
- full E7 biological correspondence.
