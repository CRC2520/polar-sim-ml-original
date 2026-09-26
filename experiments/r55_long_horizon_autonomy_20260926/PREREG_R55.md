# R55 — bounded long-horizon functional autonomy

Status: prospective before scientific development seeds are opened.
Date: 26 September 2026, America/Lima.

## Motivation

R54 cannot execute because no author-provided external checkpoint is publicly
available.

R55 begins the next roadmap tier: long-horizon autonomy.

The target is deliberately operational and bounded. R55 does **not** test
phenomenal consciousness, free will, personhood or moral agency.

## Primary question

Can one persistent agent maintain its own viability over a long, changing
lifetime by:

- selecting which internal need to address;
- maintaining a goal long enough to execute a multi-step policy;
- learning where different resources can be obtained;
- exploring when knowledge is insufficient;
- adapting after external resource regimes change;

without receiving an externally specified goal at each time step?

## Environment

Pure NumPy environment; no hidden oracle information is supplied to the agent.

Lifetime:

- 12,000 steps.

World:

- four locations arranged in a ring;
- three internally relevant resources:
  - energy;
  - integrity;
  - knowledge;
- each resource is bounded to [0,1].

Passive drains every step:

- energy: -0.0065;
- integrity: -0.0030;
- knowledge: -0.0018.

Actions:

1. harvest energy in current zone;
2. perform repair in current zone;
3. survey/explore in current zone;
4. travel clockwise;
5. travel counter-clockwise;
6. rest.

Travel consumes additional energy.

Switching directly between different work modes (energy/repair/survey) imposes
a small setup cost. The environment does not observe the agent's declared goal;
the cost depends only on external actions.

## External nonstationarity

The environment cycles through four hidden resource regimes.

Regime duration:
- 600 steps.

Regime schedule repeats but resource-yield assignments differ by regime.

Each regime changes which zones are best for:
- energy;
- repair;
- survey.

The agent never receives the regime ID.

Yield noise and hazard events are pre-generated from the seed, so all
conditions on the same seed receive identical exogenous disturbances.

Knowledge reduces hazard exposure but does not reveal future events.

## Agent observations

The agent observes only:

- current energy;
- current integrity;
- current knowledge;
- current zone;
- last realized work yield;
- last action.

No external target or goal label is observed.

## FULL_AUTONOMY

The intact agent contains:

### Persistent internal needs

Deficits in energy, integrity and knowledge create candidate goal urgencies.

### Persistent goal state

Once a goal is selected, it is maintained for a minimum commitment interval
unless an emergency energy/integrity threshold is crossed.

The goal terminates when its corresponding resource reaches a frozen satiation
threshold.

### Learned resource model

For every zone × work-mode pair the agent learns:

- exponentially weighted expected yield;
- visit count / uncertainty.

Expected yield and uncertainty determine a frozen UCB-style target-zone score.

### Exploration

When knowledge is sufficiently low or uncertainty sufficiently high, survey can
be selected without an external instruction.

### Navigation

For the active internal goal, the agent travels toward the zone with the
highest learned score for the corresponding work action.

## Frozen comparison conditions

All conditions receive identical environment disturbances for a seed.

### FULL_AUTONOMY

Persistent goal + learned resource model + exploration.

### NO_GOAL_PERSISTENCE

Same observations, resource model and action machinery, but internal goal is
recomputed every step with no commitment/hysteresis.

### NO_LEARNED_MODEL

Persistent goal remains, but the zone-resource model is reset to its uniform
prior every step. The agent therefore cannot retain learned zone/resource
relations.

### EXTERNAL_SCRIPT

No internal deficit-based goal selection. Goals follow the fixed repeating
sequence:

energy → integrity → knowledge

with the same nominal commitment duration.

This is a capacity-matched structured comparator, not a random policy.

### SHAM_STATE

Same as FULL_AUTONOMY, but carries an additional unused state vector of the same
dimension as the persistent goal/model summary. It must remain behaviorally
identical to FULL.

## Endpoints

Per seed and condition:

- survival fraction = completed steps / 12,000;
- safety fraction = fraction of steps with energy >=0.15 and integrity >=0.15;
- mean viability margin = mean(min(energy, integrity));
- catastrophic events = steps where energy <=0.05 or integrity <=0.05;
- goal switch rate;
- travel fraction;
- work-mode switch rate;
- post-regime-shift safety over the first 120 steps after each shift;
- model prediction absolute error on experienced work yields.

Primary causal effects:

[
Delta_{goal}=Safety_{FULL}-Safety_{NO_GOAL_PERSISTENCE}
]

[
Delta_{model}=Safety_{FULL}-Safety_{NO_LEARNED_MODEL}
]

[
Delta_{self}=Safety_{FULL}-Safety_{EXTERNAL_SCRIPT}.
]

Sham gap:

[
G_{sham}=|Safety_{FULL}-Safety_{SHAM}|.
]

## Seed-level guard

A seed passes only if:

- FULL survival fraction = 1.0;
- FULL safety fraction >= 0.93;
- FULL mean viability margin >=0.32;
- FULL catastrophic events = 0;
- FULL post-shift safety >=0.85;
- goal switch rate <=0.12;
- (Delta_{goal} >= +0.04);
- (Delta_{model} >= +0.10);
- (Delta_{self} >= +0.06);
- sham safety gap <=1e-12.

The comparator effects are intentionally causal/functional. A comparator is
not required to die; it must be measurably worse on the frozen viability
endpoint.

## Development panel

Seeds:

`2246001–2246016`

Development authorizes confirmation only if:

- at least 13/16 seeds pass the complete guard;
- median FULL safety >=0.95;
- median goal effect >=+0.06;
- median model effect >=+0.12;
- median self-selection effect >=+0.08;
- median sham gap <=1e-12.

Development labels:

- `R55_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `R55_DEVELOPMENT_FAIL_NO_CONFIRM`

## Reserved confirmation

Seeds:

`2247001–2247032`

Remain unopened unless development passes.

Confirmatory PASS requires:

- at least 28/32 complete seed guards;
- the same median criteria as development.

Final labels:

- `R55_BOUNDED_LONG_HORIZON_AUTONOMY_PASS`
- `R55_BOUNDED_LONG_HORIZON_AUTONOMY_FAIL`.

## Interpretation of PASS

A PASS supports bounded functional autonomy in the tested sense:

> the agent sustains viability over a long changing lifetime using persistent
> internally selected goals, learned resource relations and exploration, and
> these mechanisms have preregistered causal value relative to matched
> comparators.

It does not establish:

- consciousness;
- free will;
- intrinsic moral agency;
- human-like motivation;
- universal autonomy;
- E6b;
- E7;
- AGI/ASI.

R55 also does not establish learned value formation. That is reserved for a
later experiment.
