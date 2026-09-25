# R52 — bounded minimality challenge across external task families

Status: prospective before scientific development seeds are opened.
Date: 25 September 2026, America/Lima.

## Question

Global mathematical minimality cannot be established empirically.

R52 asks the bounded falsifiable question:

> Across a preregistered family of capable external tasks, can D, C or R be removed
> while preserving task capability when storage/computational machinery is kept
> structurally matched?

POLAR Core v1.1 remains unchanged before execution:

[
mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]

## External source

Repository:

`proroklab/popgym`

Exact source commit:

`e397e5eac9965f9963d18c9f455cd1983bca14fb`

No third-party environment source is modified.

## External task family

Four Hard task families are frozen:

1. `ConcentrationHard`
2. `CountRecallHard`
3. `AutoencodeHard`
4. `RepeatPreviousHard`

These tasks differ in the relation demanded:
- position↔identity binding;
- query↔historical count retrieval;
- sequence-position↔reverse replay;
- temporal-lag↔historical identity retrieval.

## Conditions

Every condition retains the same action interface and task-specific controller
data structures.

### FULL_DCR

Capable task-specific controller with differentiated identities, persistent
history and correct relation lookup.

### NO_D_MATCHED

Identity-sensitive storage and relation machinery remain allocated, but every
observed categorical identity is replaced by one internal equivalence class.

The controller still performs the same number of storage/query operations.

### NO_C_MATCHED

The same persistent buffers remain allocated, but past causal content is erased
or replaced by current-step content before the next decision.

Current observation and current pair/phase context remain available.

### NO_R_MATCHED

Differentiated observations and persistent memory remain intact, but retrieval
uses a fixed preregistered wrong relation:

- Concentration: cyclic rank relation ((s+1)mod13);
- CountRecall: cyclic query identity ((q+1)mod13);
- Autoencode: FIFO instead of the required LIFO relation;
- RepeatPrevious: lag (k-1) instead of externally specified lag (k).

### SHAM_CAPACITY

FULL behavior plus the same extra dummy state/buffer operations used to match
the lesion controllers' bookkeeping. The sham must not alter causal identity,
history or relation lookup.

### GENERIC_ISO

A fixed bijective recoding of categorical identity coordinates:
- 13-class tasks: (phi(i)=12-i);
- four-suit tasks: (phi(i)=3-i).

The bijection is inverted at the native action boundary where appropriate.

GENERIC_ISO must exactly match FULL external actions.

## Per-task endpoint

- ConcentrationHard: completion fraction (matched pairs / 26).
- CountRecallHard: fraction of positive-reward count decisions.
- AutoencodeHard: accuracy over scored PLAY steps.
- RepeatPreviousHard: accuracy over scored lag-active steps.

Each statistical seed averages three paired episodes per task and condition.

## Primary causal effects

For each task:

[
Delta_D=M_{FULL}-M_{NO_D}
]

[
Delta_C=M_{FULL}-M_{NO_C}
]

[
Delta_R=M_{FULL}-M_{NO_R}
]

[
Delta_{sham}=|M_{FULL}-M_{SHAM}|.
]

Also record exact FULL↔GENERIC_ISO action agreement and metric gap.

## Frozen task-cell gate

A task cell passes only if all are true:

- FULL metric >= **0.90**;
- D effect >= **+0.20**;
- C effect >= **+0.20**;
- R effect >= **+0.20**;
- sham metric gap <= **0.02**;
- each lesion effect exceeds sham gap by >= **+0.18**;
- iso metric gap <= **1e-12**;
- iso action agreement >= **1 - 1e-12**.

A statistical seed passes only if all four task cells pass.

## Development

Seeds:

`2226001–2226016`

Development authorizes confirmation only if:
- every task passes all median frozen gates; and
- at least **13/16** statistical seeds pass all four task cells.

Development resolutions:

- `R52_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `R52_DEVELOPMENT_FAIL_NO_CONFIRM`

## Reserved confirmation

Seeds:

`2227001–2227032`

They remain unopened unless development passes.

Final confirmation requires:
- every task passes every median frozen gate;
- at least **28/32** statistical seeds pass all four task cells.

Final resolutions:

- `R52_BOUNDED_DCR_NECESSITY_PASS`
- `R52_BOUNDED_DCR_NECESSITY_FAIL`

## Interpretation

PASS supports **bounded necessity across the four tested external task families**:
within those tasks and matched functional controllers, removing any one of D, C
or R causes substantial capability loss while a sham matched intervention does
not.

PASS does not establish:
- global mathematical minimality;
- universal sufficiency;
- unique decomposition;
- neural learnability;
- POLAR implementation superiority;
- E6b;
- E7;
- consciousness.

A task in which a lesion preserves capability is retained as evidence against
bounded necessity under this panel and may motivate Core v1.2 review rather than
post-hoc threshold changes.
