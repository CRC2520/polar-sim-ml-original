# R43 result — external recurrent baseline qualification remains unresolved

Frozen resolution:

`R43_DEVELOPMENT_FAIL_NO_CONFIRM`

R43 tests whether a standard third-party recurrent RL implementation can first establish its own competence and memory dependence on an unmodified external POMDP before any POLAR comparison is attempted.

## External benchmark

POPGym 1.0.7 VelocityOnlyCartPole:
- training: Easy, max episode 200;
- development: Medium, max episode 400;
- reserved confirmation: Hard, max episode 600.

The benchmark exposes only cart and pole velocities and hides positions.

## Fixed training

Three independent RecurrentPPO/MlpLstmPolicy checkpoints and three matched PPO/MlpPolicy checkpoints were trained with 1,048,576 transitions per model, fixed final checkpoint only.

Training seeds:
- 2146001
- 2146002
- 2146003

No hyperparameter sweep, checkpoint selection or post-result tuning was performed.

## Development results on Medium

| Training seed | Recurrent score | Step-reset score | History benefit | Feed-forward PPO | Eligible |
|---|---:|---:|---:|---:|---|
| 2146001 | 0.66925 | 0.39350 | +0.27575 | 0.540625 | No |
| 2146002 | 0.52125 | 0.211625 | +0.309625 | 0.87075 | No |
| 2146003 | 1.00000 | 0.091125 | +0.908875 | 0.99250 | Yes |

Development eligibility required at least 2/3 recurrent checkpoints with:
- recurrent score >=0.80;
- intact-minus-step-reset history benefit >=0.15.

Only 1/3 is eligible.

The failure is therefore an absolute competence/robustness failure, not a generic absence of recurrent-state utility: all three recurrent checkpoints show positive history benefit, and one solves the external development benchmark perfectly.

The feed-forward comparator is also heterogeneous: it substantially exceeds recurrent seed 2146002 while seeds 2146001 and 2146003 show different relationships. No architecture-wide ranking is inferred.

## Scientific decision

Development is not eligible, so confirmatory Hard episode seeds 2147001–2147040 remain unopened.

R43 does NOT establish:
- a qualified strong external recurrent baseline;
- POLAR superiority;
- D or R on this task;
- E6b;
- E7;
- universal recurrence benefit;
- consciousness.

The external recurrent-baseline question therefore remains OPEN.

This result also argues against simply increasing training on this same opened panel. The next clean baseline attempt, if pursued, should use a prevalidated external recipe or a separately supplied qualified checkpoint rather than another in-house tuning pass on VelocityOnlyCartPole.

## Provenance

Workflow run: 35828621896.
Execution head: bdf5523fbab7c32d96830740af2c1e30d7edf73f.
Frozen source blob: 1d6330846973ee901bd13998a8b408013223db20.
Frozen protocol blob: a61be491df343656e2a81f84c5e10ebf7bd032c9.

Evidence artifact: 10735824368,
sha256:43c8956910865736053cb7e8093a1a308347bbae489b94fbfaaa2c75b8ea56c0.

Model artifacts:
- 2146001: 10736748180, sha256:02671fe048f14efd55518f9ae5be3a952b15003142774bc41e376b3f5ccbe516
- 2146002: 10736498933, sha256:0f120e38eb3f3a3be7c88beafd68d54832c55e0573c4b8e18b125169260e8ba5
- 2146003: 10735709373, sha256:68dadaa693698f0085daafcccc52887c0fd097d22442bbddaffd48892f218533
