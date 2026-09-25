# R48 engineering amendment and adjudication note

Canonical scientific run: `36089447230`

Scientific execution commit:

`33ef845eca64ad80ea956b8ecc5a758f6a883843`

## What failed technically

All three preregistered Medium development jobs completed successfully and
uploaded their frozen checkpoint/result artifacts.

The downstream `development_gate` job failed before aggregation with:

`ModuleNotFoundError: No module named 'numpy'`

The gate job checked out the repository but did not install the runtime
dependency imported by `r48_neural_relation.py`.

This is an aggregation/infrastructure defect. It did not alter:

- training;
- evaluation;
- checkpoints;
- scientific seeds;
- held-out episode seeds;
- thresholds;
- scientific endpoints.

## Canonical development artifacts

- seed 2196001 artifact: `10845099062`
- seed 2196002 artifact: `10846010240`
- seed 2196003 artifact: `10845033735`

All three artifacts record:

- execution commit:
  `33ef845eca64ad80ea956b8ecc5a758f6a883843`
- protocol SHA-256:
  `150d568477f7efa264806527f76d814973f8b8f97144243dc6a004af9fada761`
- runner SHA-256:
  `e9085f24d2af036fc8fffa43727b7b35a1daaaea17a61b5e854527f1807c2738`

## Deterministic gate recomputation

The preregistered development eligibility rule is:

- intact accuracy >= 0.90;
- D effect >= +0.25;
- C effect >= +0.25;
- R effect >= +0.20;
- iso accuracy gap <= 1e-12;
- iso action agreement >= 1 - 1e-12.

Observed canonical results:

| train seed | intact | D effect | C effect | R effect | iso gap | iso action agreement | eligible |
|---|---:|---:|---:|---:|---:|---:|---|
| 2196001 | 0.186489 | +0.000809 | +0.146036 | +0.000000 | 0 | 1.0 | NO |
| 2196002 | 0.190939 | -0.003236 | +0.160599 | -0.001618 | 0 | 1.0 | NO |
| 2196003 | 0.216424 | +0.002832 | +0.186084 | +0.001214 | 0 | 1.0 | NO |

Eligible checkpoints: **0/3**.

Therefore the scientific resolution is uniquely determined by the frozen rule:

`R48_DEVELOPMENT_FAIL_NO_CONFIRM`

## Confirmatory protection

No Hard confirmatory training/evaluation job executed.

The following reserved sets remain unopened:

- train seeds 2197001–2197003;
- held-out evaluation seeds 2197101–2197112.

No repair run is authorized to reopen this R48 panel.

## Interpretation

The reward-only LSTM training shows a partial recurrent-state effect under the
C step-reset lesion, but it does not achieve absolute task competence and does
not show meaningful D or R lesion sensitivity.

This does **not** support neural relation learnability under the frozen R48
design.

The adverse result must be retained. A future neural relation-learning
experiment must use a new preregistered design/task/training method and fresh
seeds rather than tuning against the R48 held-out panel.
