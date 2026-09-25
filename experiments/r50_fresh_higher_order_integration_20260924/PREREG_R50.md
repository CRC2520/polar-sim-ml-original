# R50 — fresh higher-order integration in a persistent agent

Status: prospective before scientific development seeds are opened.
Date: 24 September 2026, America/Lima.

## Motivation

R36 and R36-R1 demonstrated that all declared components can pass individually,
while robust same-seed coexistence remained below the frozen threshold.
R36-R2 then showed that tuning context speed, memory horizon or planning margin
inside that same family produces cross-component interference rather than a
stable joint solution.

R50 therefore does **not** reuse the R36-R2 panel and does not select among
variants. It uses a fresh task family, fresh seeds and one preregistered
partitioned persistent-agent architecture.

POLAR Core v1.1 remains:

[
mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]

## Primary question

Can one persistent agent sustain, on the **same seed and same continuous
lifetime**, all of the following under a fresh design?

- D differentiated causal state;
- C recurrent context state;
- R learned relational dynamics;
- persistent reentry memory;
- own-history specificity;
- delayed source attribution;
- temporal-order attribution;
- calibrated metacognitive confidence;
- functionally useful confidence-dependent planning;
- multi-step model-based planning.

## Fresh environment

State dimension: 3.

Actions: seven bounded axis-aligned actions.

Five recurring dynamical regimes.

Block length: 180 steps.

Schedule:

[
[0,1,2,3,4,0,3,1,4,2].
]

One lifetime therefore contains exactly 1,800 transitions.

Each regime has:
- a distinct stable 3x3 transition matrix;
- distinct off-diagonal relational structure;
- a distinct target;
- a distinct two-dimensional noisy cue center.

No regime ID is given to the agent.

The cue centers are arranged on a pentagon rather than the four-quadrant
encoding used by R36.

Rare exogenous shocks create a source-attribution problem.

Each agent also has a persistent hidden actuator profile. A donor agent with a
complementary profile is used only for the own-history transplant comparator.

## Fresh architecture

R50 freezes one architecture before development.

### Two-timescale recurrent context

The agent maintains:
- a fast cue state;
- a slow cue state.

The context estimate is a fixed blend of the two and selects the nearest of the
five cue prototypes. Current-cue-only classification is used only as the C
lesion.

### Partitioned persistent dynamics memory

Each inferred context owns its own dynamics model and retained transition
buffer.

The dynamics buffer and metacognitive reliability buffer are separate.

### Longer independent reliability memory

Metacognitive reliability retains a longer history than the dynamics fit. This
is intended to test whether separating epistemic memory from dynamics-fitting
memory reduces the interference seen in R36-R2. This is a single frozen
architecture, not a development sweep.

### Confidence-dependent three-step planning

The agent computes myopic and multi-step candidates from its learned model.
Planning is used only when:
- estimated reliability exceeds the frozen confidence threshold; and
- predicted planning advantage exceeds the frozen uncertainty margin.

The reversed-gate counterfactual tests functional metacognitive use.

## Same-agent interventions

### D

Collapse differentiated state-transition columns to their row-wise mean while
preserving action and context structure.

### C

Use the current noisy cue only, instead of the two-timescale recurrent context,
to select the learned persistent model.

### R

Remove off-diagonal learned state relations while preserving diagonal dynamics
and action effects.

### Reentry memory

On revisits, compare the persistent context model with a block-local cold model
during the early reentry window.

### Own-history specificity

Apply a model learned by a donor agent with a complementary hidden actuator
profile to the main agent's current trajectory.

### Source attribution

Infer externally shocked transitions from prediction residuals and evaluate
balanced delayed accuracy.

### Time attribution

Query pairwise temporal order from episodic entries carrying the persistent
internal clock.

### Metacognitive calibration

High-confidence transitions must have lower factual prediction error than
low-confidence transitions.

### Functional metacognitive gate

Compare the true multi-step cost of the action selected by the confidence gate
with the action selected by the reversed gate.

### Planning

On states where planning is invoked, compare the chosen planned first action
with the myopic first action under the true external multi-step cost, used only
for evaluation.

## Frozen thresholds

A seed passes only if **all** criteria pass simultaneously:

- D prediction damage >= **+0.0040**;
- C recurrent-context gain >= **+0.0005**;
- R relation-lesion damage >= **+0.0005**;
- memory reentry gain >= **+0.00010**;
- own-history transplant damage >= **+0.0008**;
- source balanced accuracy >= **0.85**;
- time-order accuracy >= **0.98**;
- metacognitive calibration gap > **0**;
- metacognitive gate gain > **0**;
- planning gain >= **+0.00020**;
- planning invocation rate in **[0.02, 0.50]**;
- stable fraction >= **0.995**;
- same-agent steps = **1800**;
- learned context models >= **5**.

No threshold may be changed after development seeds are opened.

## Development

Seeds:

`2206001–2206006`

Development authorizes confirmation only if:
- at least **4/6** seeds pass the full same-agent conjunction;
- every individual component passes in at least **4/6** seeds;
- median stable fraction >=0.995;
- median learned-model count >=5.

If development fails:

`R50_DEVELOPMENT_FAIL_NO_CONFIRM`

If development authorizes confirmation:

`R50_DEVELOPMENT_AUTHORIZE_CONFIRM`

## Reserved confirmation

Seeds:

`2207001–2207016`

They remain unopened unless development passes.

Confirmatory PASS requires:
- at least **12/16** full same-agent conjunction passes;
- every individual component passes in at least **12/16** seeds;
- median stable fraction >=0.995;
- median learned-model count >=5.

Final resolutions:

- `R50_FRESH_HIGHER_ORDER_INTEGRATION_PASS`
- `R50_FRESH_HIGHER_ORDER_INTEGRATION_FAIL`

## Interpretation boundaries

A PASS would support robust same-program coexistence of the tested computational
capacities in one bounded synthetic persistent agent under a fresh design.

It would not establish:
- phenomenal consciousness;
- an intrinsic self;
- global minimality;
- E6b;
- E7;
- open-ended autonomy;
- intrinsic value formation.

A FAIL must be retained and must not trigger tuning on the confirmatory panel.
