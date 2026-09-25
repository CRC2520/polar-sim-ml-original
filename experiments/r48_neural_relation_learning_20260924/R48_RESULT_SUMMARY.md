# R48 result — neural relation discovery from reward

Frozen resolution:

`R48_DEVELOPMENT_FAIL_NO_CONFIRM`

## Canonical science

- canonical scientific run: `36089447230`
- scientific head SHA: `33ef845eca64ad80ea956b8ecc5a758f6a883843`
- external source: `proroklab/popgym@e397e5eac9965f9963d18c9f455cd1983bca14fb`
- development task: `CountRecallMedium`
- architecture: generic LSTM, hidden size 128
- training signal: native task reward only
- training budget: 12,000 episodes/checkpoint
- training seeds: 2196001–2196003
- held-out evaluation seeds: 2196101–2196112

The original development-gate job failed only because its aggregation job did
not install numpy/torch. All three scientific checkpoint artifacts were already
complete. They were mechanically re-adjudicated without reopening training in
run `36090760519`.

## Frozen development results

| train seed | intact accuracy | D effect | C effect | R effect | iso gap | iso action agreement | eligible |
|---|---:|---:|---:|---:|---:|---:|---|
| 2196001 | 0.186489 | +0.000809 | +0.146036 | +0.000000 | 0 | 1.0 | no |
| 2196002 | 0.190939 | -0.003236 | +0.160599 | -0.001618 | 0 | 1.0 | no |
| 2196003 | 0.216424 | +0.002832 | +0.186084 | +0.001214 | 0 | 1.0 | no |

Eligibility required simultaneously:

- intact accuracy >= 0.90;
- D effect >= +0.25;
- C effect >= +0.25;
- R effect >= +0.20;
- exact GENERIC_ISO accuracy/action equivalence.

Observed:

- eligible checkpoints: **0/3**;
- required: **2/3**.

Therefore Hard confirmation was not authorized and all reserved
2197001–2197003 / 2197101–2197112 confirmatory seeds remain unopened.

## Interpretation

R48 does **not** support reward-only neural relation learnability under this
frozen design.

The repeated signature is informative rather than a generic null:

- step-reset C lesions reduce accuracy by approximately +0.15 to +0.19;
- collapsing identity D has approximately zero effect;
- cycling the query-to-history relation R has approximately zero effect;
- isomorphic recoding is exact.

Thus the trained policies acquired a history-dependent temporal/global heuristic,
but not a competent identity-specific counting policy whose behavior depends on
the correct query↔history relation.

This distinguishes generic recurrent-state use from relation-specific neural
organization.

## Governance

No hyperparameter sweep or same-panel retraining is authorized after this
development failure.

R49 learned-core cross-domain transport is **not opened**, because its
prerequisite was a valid learned R mechanism from R48.

## Boundaries

R48 does not weaken the R47 functional result. R47 and R48 ask different
questions:

- R47: can D+C+R coexist and perform causal work in one external functional
  agent? **supported same-program**.
- R48: can a generic LSTM discover the required identity-specific relational
  organization from this reward-only training design? **not supported**.

Still open:
- other genuinely new neural relation-learning designs/tasks;
- E6b;
- E7;
- global minimality;
- higher-order integration;
- external robustness;
- identity continuity;
- consciousness.
