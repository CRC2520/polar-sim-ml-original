# R52 — bounded minimality challenge across external memory-task families

Status: prospective before R52 scientific development seeds are opened.
Date: 24 September 2026, America/Lima.

## Question

R52 does **not** attempt to prove global mathematical minimality.

It tests a bounded falsifiable question:

> Across four preregistered external task families that require identity-specific
> historical organization, can D, C or R be removed while preserving capability
> when information access and representational/storage capacity are otherwise
> retained?

POLAR Core v1.1 remains:

[
mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]

## External source

Repository: `proroklab/popgym`

Exact source commit:

`e397e5eac9965f9963d18c9f455cd1983bca14fb`

No external environment source is modified.

## Frozen task families

1. **ConcentrationHard** — identity matching across hidden card positions.
2. **AutoencodeHard** — sequence replay relation.
3. **RepeatPreviousHard** — fixed temporal-lag relation.
4. **CountRecallHard** — query identity to accumulated historical count.

The first three reuse the already frozen R47/R51 controllers exactly through
their committed runner modules. R52 does not retune them.

CountRecall receives a new transparent functional controller specified here.

## Matched conditions

For every task, the controller exposes:

### FULL

Uses differentiated identities, persistent history and the task-correct
identity/position/order/query relation.

### NO_D

All observed identities are mapped to one internal equivalence class while
retaining the same storage structures and action interface.

### NO_C

Persistent historical storage is erased according to the task:
- Concentration: R47 C1_ONLY lesion;
- Autoencode: retain only the most recent input item;
- RepeatPrevious: retain only the current item;
- CountRecall: reset count memory every step.

### NO_R

Preserve differentiated inputs and persistent storage but apply a fixed wrong
relation:
- Concentration: cyclic wrong rank-to-position lookup;
- Autoencode: FIFO instead of the correct LIFO sequence relation;
- RepeatPrevious: lag k-1 instead of k;
- CountRecall: query identity q+1 mod K instead of q.

### GENERIC_ISO

Use a bijective relabeling of identity coordinates with exact inverse at the
native action boundary.

No condition gets additional information, hidden environment state or a larger
memory store.

## Task scores

- ConcentrationHard: completion fraction.
- AutoencodeHard: scored-step action accuracy.
- RepeatPreviousHard: scored-step action accuracy.
- CountRecallHard: scored-step count-query accuracy.

All scores lie in [0,1].

## Per-task causal effects

[
Delta_D=S_{FULL}-S_{NO_D},
quad
Delta_C=S_{FULL}-S_{NO_C},
quad
Delta_R=S_{FULL}-S_{NO_R}.
]

Isomorphic gap:

[
G_{iso}=|S_{FULL}-S_{ISO}|.
]

## Frozen per-seed criterion

A task passes for a statistical seed only if:

- FULL score >= **0.95**;
- D effect >= **+0.40**;
- C effect >= **+0.40**;
- R effect >= **+0.20**;
- iso score gap <= **1e-12**;
- iso action agreement >= **1 - 1e-12**.

A statistical seed passes the bounded family guard only if **all four tasks**
pass simultaneously.

## Development

Seeds:

`2226001–2226016`

Two paired episodes per task/condition/seed.

Development authorizes confirmation only if:
- at least **13/16** full four-task guards pass;
- every task's median FULL/D/C/R/iso criteria pass.

Development resolutions:

- `R52_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `R52_DEVELOPMENT_FAIL_NO_CONFIRM`

## Reserved confirmation

Seeds:

`2227001–2227032`

Confirmation remains unopened unless development passes.

Final PASS requires:
- at least **28/32** full four-task guards;
- every task's frozen median criteria pass.

Final resolutions:

- `R52_BOUNDED_MINIMALITY_CHALLENGE_PASS`
- `R52_BOUNDED_MINIMALITY_CHALLENGE_FAIL`

## Interpretation of PASS

A PASS supports **bounded task-family necessity** of D, C and R across the four
tested external memory-task families under the declared functional
decomposition.

It does not establish:
- global minimality;
- universal sufficiency;
- uniqueness of the D/C/R factorization;
- neural learnability;
- POLAR implementation superiority;
- E6b;
- E7;
- consciousness.

## Theory-revision trigger

A capable matched controller that reproducibly violates one frozen D/C/R
criterion in this preregistered family is scientifically meaningful and must
remain in the record. Such a result would weaken the bounded necessity claim and
could motivate, but would not automatically force, Core v1.2.
