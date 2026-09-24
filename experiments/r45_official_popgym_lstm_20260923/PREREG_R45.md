# R45 — official POPGym LSTM reproduction and paired checkpoint qualification

Status: prospectively frozen before R45 scientific training seeds are opened.
Date: 23 September 2026, America/Lima.

## Starting boundary

R44 established **published benchmark-ceiling compatibility** on the historical POPGym StatelessCartPole benchmark, but it did not replay an external LSTM checkpoint on the same held-out episodes. The current POLAR Core remains:

[
mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]

R45 addresses only the external recurrent-comparator boundary. It does not alter the core specification.

## External implementation fixed before execution

Upstream repository: `proroklab/popgym`.
Historical paper commit: `e397e5e`.

The R45 recurrent comparator uses the upstream implementation at that commit:

- `popgym/baselines/ray_models/ray_lstm.py` — upstream LSTM model;
- the PPO architecture/configuration from `popgym/baselines/ppo.py`;
- RLlib `ray[rllib]==2.0.0`;
- historical Gym API `gym==0.24.0`;
- `Antialias(PreviousAction(env))`, matching the paper launcher.

The upstream repository does not provide a reusable pretrained paper checkpoint in the campaign. Therefore R45 trains **new checkpoints with the official upstream model/configuration**. They are reproduced official-code checkpoints, not author-provided published checkpoints.

## Fixed compute boundary

The paper reports 15M environment timesteps per experiment. R45 does **not** claim to reproduce that full training budget.

For a bounded qualification study, every R45 checkpoint receives exactly:

- 1,048,576 environment timesteps;
- LSTM hidden size 256;
- linear pre/post layers and PPO settings from the upstream launcher;
- BPTT cutoff 1024;
- aggregate train batch 65,536;
- SGD minibatch 8,192;
- gamma 0.99;
- complete-episode sampling;
- no hyperparameter sweep.

Infrastructure adaptation only: R45 uses 2 rollout workers × 32 environments/worker instead of the upstream launcher's 4 × 16, preserving the same aggregate train batch. No result-dependent retuning is allowed.

## Scientific question

At the fixed bounded training budget, can the official upstream POPGym LSTM implementation produce a competent recurrent baseline on historical StatelessCartPole and thereby support a direct, same-episode comparison against the frozen R44 CORE_C controller?

Comparator competence is adjudicated **before** interpreting POLAR-versus-LSTM differences.

## Controllers / conditions

1. **UPSTREAM_LSTM_INTACT** — reproduced official-code checkpoint with recurrent state carried through the episode.
2. **UPSTREAM_LSTM_STEP_RESET** — the same checkpoint but recurrent hidden state reset before every action; one-step previous-action input remains because it belongs to the official wrapper.
3. **CORE_C** — frozen R44 recurrent observer/LQR, unchanged.
4. **GENERIC_ISO** — frozen R44 coordinate-isomorphic control, used as an implementation non-privilege guard.

No R45 training modifies CORE_C or GENERIC_ISO.

## Seed separation

### Development

Training seeds:
- 2166001
- 2166002
- 2166003

Environment: `StatelessCartPoleEasy`.

Held-out evaluation seeds shared by all three checkpoints:
- 2166101–2166112

Each evaluation seed contains four deterministic episodes per condition using episode seeds `seed*1000 + episode_index`.

A reproduced LSTM checkpoint is **eligible** only if:

- median intact normalized duration >= 0.90;
- median intact-minus-step-reset normalized duration >= +0.15.

Development authorizes confirmation only if at least **2/3** independently trained checkpoints are eligible.

CORE_C/LSTM differences are recorded in development but cannot rescue an invalid external comparator.

If development fails, all confirmatory training/evaluation seeds remain unopened.

### Confirmatory

Only if development authorizes confirmation.

Training seeds, independently trained per difficulty:
- 2167001
- 2167002
- 2167003

Difficulties:
- `StatelessCartPoleMedium`
- `StatelessCartPoleHard`

Held-out evaluation seeds:
- 2167101–2167112

Per difficulty, confirmation requires:

1. at least 2/3 reproduced LSTM checkpoints satisfy the same competence gate;
2. median CORE_C score >= 0.90;
3. median absolute CORE_C–GENERIC_ISO gap <= 1e-12;
4. paired CORE_C minus intact-LSTM median >= -0.05.

The last criterion is a noninferiority-style compatibility guard for the frozen CORE_C against the **reproduced official-code checkpoints** at the bounded R45 budget. It is not a superiority test.

R45 passes the direct reproduced-checkpoint boundary only if both Medium and Hard satisfy all four conditions.

## Frozen interpretations

Possible resolutions:

- `R45_OFFICIAL_LSTM_REPRODUCTION_DEV_FAIL_NO_CONFIRM`
- `R45_DIRECT_REPRODUCED_CHECKPOINT_COMPATIBILITY_PASS`
- `R45_DIRECT_REPRODUCED_CHECKPOINT_COMPATIBILITY_FAIL`

A PASS may establish a direct paired comparison against newly trained checkpoints using the official upstream POPGym LSTM implementation and launcher configuration at the fixed R45 budget.

A PASS does **not** establish:

- comparison against an author-provided paper checkpoint;
- reproduction of the paper's 15M-timestep endpoint;
- POLAR superiority;
- E6b independent replication;
- external D or R;
- global minimality;
- E7;
- phenomenal consciousness.

A development FAIL means only that this bounded official-code reproduction did not yield a sufficiently competent comparator. It does not falsify the published POPGym result.

## Governance

- Protocol and scientific runner are committed before scientific seeds are executed.
- Development and confirmatory seeds are disjoint.
- Confirmatory jobs are conditionally blocked unless the development gate passes.
- Checkpoints, raw per-episode records, exact dependency versions, source commit, runner hashes and adjudication JSON are preserved as workflow artifacts.
- No confirmatory threshold may be changed after the development gate is evaluated.
