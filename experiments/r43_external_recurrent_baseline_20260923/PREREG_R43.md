# R43 — external recurrent baseline qualification on VelocityOnlyCartPole

Status: prospective before training.
Date: 23 September 2026 America/Lima.
Parent evidence: R39/R41 recurrent baselines failed their own validity gates; R42 strengthened same-program D+C+R transport but left a strong external recurrent comparator OPEN.

## Question

Can a standard third-party recurrent RL implementation demonstrate competent, memory-dependent control on an external partially observable benchmark before any POLAR comparison is attempted?

R43 is a baseline-qualification experiment. It does not compare POLAR and cannot establish superiority.

## External benchmark

Unmodified POPGym 1.0.7:
- training: VelocityOnlyCartPoleEasy (max episode 200)
- development evaluation: VelocityOnlyCartPoleMedium (max episode 400)
- confirmatory held-out evaluation: VelocityOnlyCartPoleHard (max episode 600)

The environment exposes only cart and pole velocities, hiding positions. It is third-party code from proroklab/popgym.

## Models

For each training seed:
- RecurrentPPO / MlpLstmPolicy from sb3-contrib 2.7.1;
- PPO / MlpPolicy from stable-baselines3 2.7.1 as a matched feed-forward descriptive comparator.

Training seeds: 2146001, 2146002, 2146003.

Fixed budget/model:
- 1,048,576 environment transitions;
- 8 vectorized envs;
- n_steps=256;
- batch_size=2048;
- n_epochs=5;
- learning_rate=3e-4;
- gamma=.99;
- gae_lambda=.95;
- ent_coef=.0;
- clip_range=.2;
- recurrent LSTM hidden size 64;
- fixed final checkpoint only; no checkpoint selection.

No hyperparameter sweep or curriculum is permitted.

## Evaluation

Development episode seeds: 2146101–2146120.
Confirmatory episode seeds: 2147001–2147040.

Per episode:
score = steps_completed / max_episode_length.

Each recurrent checkpoint is evaluated:
1. intact hidden state;
2. RESET control: LSTM state forcibly reset before every decision.

Feed-forward PPO is evaluated as a descriptive matched-capacity comparison.

## Development eligibility

At least 2/3 recurrent checkpoints must simultaneously satisfy on Medium:
- mean intact score >= .80;
- mean intact-minus-RESET >= .15.

Additionally:
- no NaN/non-finite metric;
- every evaluated episode length must be <= environment maximum.

If development is ineligible, confirmation remains unopened and reserved seeds are not consumed.

## Confirmatory gate

Using the same fixed checkpoints on Hard:
At least 2/3 recurrent checkpoints must satisfy:
- mean intact score >= .70;
- mean intact-minus-RESET >= .15.

Frozen resolution:
- PASS: R43_EXTERNAL_RECURRENT_BASELINE_QUALIFIED
- otherwise: R43_EXTERNAL_RECURRENT_BASELINE_NOT_QUALIFIED

Feed-forward performance is reported but does not determine eligibility.

## Boundaries

A PASS validates this recurrent baseline on this external benchmark only. It does not establish:
- POLAR superiority;
- D or R;
- E6b independent replication;
- E7;
- universal recurrence benefit;
- consciousness.

A later experiment would be required to compare a frozen POLAR-class controller against the now-qualified external baseline.
