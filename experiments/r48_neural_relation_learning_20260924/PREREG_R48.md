# R48 — neural relation discovery from reward in external POPGym tasks

Status: prospective before scientific development seeds are opened.
Date: 24 September 2026, America/Lima.

## Starting boundary

R47 supports a bounded external same-task D+C+R conjunction in one functional
agent, but its relation update/query rule is designed explicitly.

R48 asks a narrower but stronger learnability question:

> Can a generic recurrent neural agent trained only from task reward discover
> relation-specific organization that later shows separable D, C and R causal
> signatures?

R48 does not change POLAR Core v1.1 before execution:

[
mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]

## External benchmark source

Repository: `proroklab/popgym`.

Exact historical source commit:

`e397e5eac9965f9963d18c9f455cd1983bca14fb`.

No POPGym environment code is modified.

Development task:
- `CountRecallMedium`

Reserved confirmation task:
- `CountRecallHard`

These are externally defined variants with different episode length, identity
cardinality and action cardinality.

## Why CountRecall tests neural relation learning

At each step the agent observes:
- a dealt identity;
- a queried identity.

The correct action is the number of times the queried identity has appeared so
far. The environment gives only scalar reward for the chosen count.

A capable recurrent neural policy therefore has to preserve differentiated
identity information, maintain history over the episode, and relate the current
query identity to the correct accumulated history.

Training code never reads:
- `env.get_state()`;
- hidden counts from the environment;
- oracle actions;
- labels for D/C/R;
- relation tables.

## Neural models

Two generic recurrent architectures are frozen:

1. **LSTM** — primary architecture.
2. **GRU** — held-out architecture in confirmatory Hard.

Both receive the same observation representation:
- one-hot dealt identity;
- one-hot query identity;
- concatenated into one input vector.

The recurrent hidden size is 128.

The output head is linear to the native CountRecall action space.

No handcrafted associative-memory table, key-value lookup or count accumulator
is available to the policy.

## Training algorithm

Pure reward-based categorical policy gradient.

For each episode:
- actions are sampled from the policy;
- only native environment reward is used;
- reward signs are rescaled from the native per-step normalization back to
  ({-1,+1}) before the policy-gradient loss;
- the baseline is the episode mean rescaled reward;
- entropy regularization is fixed at 0.01;
- Adam learning rate is 3e-4;
- gradient norm clipping is 1.0.

No hyperparameter sweep is authorized after scientific development begins.

Training budgets:
- Medium development: **12,000 episodes/checkpoint**;
- Hard confirmatory: **18,000 episodes/checkpoint**.

## Frozen scientific seeds

### Development / CountRecallMedium / LSTM

Training seeds:
- 2196001
- 2196002
- 2196003

Held-out evaluation seeds:
- 2196101–2196112

Two evaluation episodes per held-out seed.

### Confirmatory / CountRecallHard

Training seeds:
- 2197001
- 2197002
- 2197003

Architectures:
- LSTM
- GRU

Held-out evaluation seeds:
- 2197101–2197112

Two evaluation episodes per held-out seed.

The confirmatory task/seeds remain unopened unless development authorizes them.

## Conditions

Each trained checkpoint is evaluated under:

### INTACT

Native held-out observation stream and recurrent state.

### NO_D_COLLAPSE

Both dealt and query identity channels are mapped to identity 0 before one-hot
encoding. Recurrent state and output head remain intact.

This removes identity differentiation while preserving state capacity.

### NO_C_STEP_RESET

Native identity observations remain intact, but recurrent hidden state is reset
before every step.

This removes persistent historical state.

### WRONG_R_QUERY_CYCLE

Dealt identities are unchanged. Only the query identity is replaced by

[
q'=(q+1)mod K
]

where (K) is the task's number of identity classes.

All query identities remain differentiated; the intervention changes only the
identity-to-history relation used for retrieval.

### GENERIC_ISO

A fixed bijection

[
phi(i)=K-1-i
]

is applied to both external identity channels.

The trained recurrent model is transformed by the exact corresponding
permutation of the dealt/query input-weight columns. Recurrent and output
weights remain unchanged.

This condition must produce the same greedy action sequence as INTACT up to
floating-point tolerance.

## Per-checkpoint endpoints

Held-out action accuracy is the fraction of native CountRecall steps receiving
positive reward.

For each checkpoint:

[
Delta_D=A_{intact}-A_{noD}
]

[
Delta_C=A_{intact}-A_{noC}
]

[
Delta_R=A_{intact}-A_{wrongR}
]

and

[
G_{iso}=|A_{intact}-A_{iso}|.
]

The evaluator also records exact greedy action agreement between INTACT and
GENERIC_ISO.

## Development eligibility

A Medium LSTM checkpoint is eligible only if:

- intact held-out accuracy >= **0.90**;
- D effect >= **+0.25**;
- C effect >= **+0.25**;
- R effect >= **+0.20**;
- iso accuracy gap <= **1e-12**;
- iso action agreement >= **1.0 - 1e-12**.

Development authorizes Hard confirmation only if at least **2/3** checkpoints
are eligible.

Development resolutions:
- `R48_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `R48_DEVELOPMENT_FAIL_NO_CONFIRM`

## Confirmatory eligibility

For each architecture separately, a Hard checkpoint is eligible only if:

- intact held-out accuracy >= **0.80**;
- D effect >= **+0.15**;
- C effect >= **+0.30**;
- R effect >= **+0.15**;
- iso accuracy gap <= **1e-12**;
- iso action agreement >= **1.0 - 1e-12**.

An architecture confirms if at least **2/3** checkpoints are eligible.

## Final R48 adjudication

### Full PASS

`R48_NEURAL_RELATION_LEARNABILITY_PASS`

if both LSTM and held-out GRU confirm on CountRecallHard.

### Partial

`R48_NEURAL_RELATION_LEARNABILITY_PARTIAL`

if exactly one architecture confirms.

### Fail

`R48_NEURAL_RELATION_LEARNABILITY_FAIL`

if neither confirms after development authorized confirmation.

A failed development gate remains:

`R48_DEVELOPMENT_FAIL_NO_CONFIRM`.

## Valid interpretation of PASS

A full PASS supports that two generic recurrent neural architectures, trained
only from external reward in the tested CountRecall family, can learn a policy
whose held-out behavior depends causally on:

- differentiated identities D;
- persistent recurrent history C;
- the correct query-to-history relation R.

It also supports coordinate non-privilege under exact isomorphic recoding.

## Boundaries

R48 does not establish:

- cross-domain transport outside CountRecall;
- a unique internal neural representation of R;
- global minimality or universal sufficiency;
- POLAR algorithmic superiority;
- E6b independent replication;
- E7 biological correspondence;
- identity/autobiographical continuity;
- AGI/ASI;
- phenomenal consciousness.

Cross-domain learned transport is reserved for R49.
