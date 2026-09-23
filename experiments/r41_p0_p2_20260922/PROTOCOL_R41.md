# R41 — redesigned P0–P2 campaign after R40

Status: prospective protocol before scientific runs.
Date: 22 September 2026 America/Lima (workflow may execute after 00:00 UTC).
Parent evidence: R40 merged as a bounded negative/partial result. Historical panels R36-R2/R37/R38/R39/R40 are not retuned.

## Epistemic separation

P0 asks whether a strong recurrent comparator can be made demonstrably competent on a fresh third-party memory task. P1 asks whether D+C+R transports under scale-normalized causal metrics in a fresh control generator. P2 asks whether an explicitly integrated episodic-memory extension becomes causally relevant and whether history/profile interventions have bounded identity-like effects. A PASS in one scope is not a PASS in another. E6b and E7 cannot be self-certified here.

## P0 — two-tier external memory baseline

Fresh third-party task: unmodified POPGym RepeatFirstEasy, not used in R40. Three RecurrentPPO/MlpLstmPolicy models use seeds 2106001–2106003, 1,048,576 transitions/model, 16 envs, n_steps 256, batch 4096, 10 epochs, lr 5e-5, gamma .99, gae_lambda 1.0, clip .3, entropy 0, LSTM64. These choices are fixed before training and are paper-informed; no checkpoint selection or hyperparameter sweep is allowed.

Development evaluation seeds: 2106101–2106116. Eligibility requires at least 2/3 models with accuracy >=.85 and intact-minus-step-reset >=.25. If eligible, confirmation uses seeds 2107001–2107064 on the same fixed checkpoints. Confirmatory gate: at least 2/3 models accuracy >=.85 and history benefit >=.25.

A separate standard torch LSTM classifier is trained on sequences generated from RepeatFirstEasy only as a task-solvability instrument. Its PASS cannot substitute for recurrent PPO validity and cannot be called independent replication.

No comparison to POLAR is made in P0.

## P1 — transport-invariant core evaluation

Fresh seeds 2116001–2116004 for instrument checks and 2117001–2117032 for the frozen evaluation. The controller is the exact R36-R1 runner implementation. No new cue filter or tuned architecture is evaluated.

A fresh 3D random stable-dynamics generator uses independent RNG streams for matrices, regime schedule, body profile, process noise, cue noise, shocks and block targets. Targets are generated independently from hidden regimes. Two observation conditions are evaluated: standard and stress.

For each seed/condition, capability is normalized:
  score = (cost_RANDOM - cost_AGENT) / (cost_RANDOM - cost_ORACLE)
where RANDOM and two-step true-dynamics ORACLE are evaluator controls on matched noise streams. Score is not clipped.

D, C and R use within-trajectory paired lesion differences. Each effect is standardized as dz = mean(delta)/sd(delta), avoiding transport of raw MSE magnitudes from R36. Prespecified pass per seed:
- normalized score >= .50
- D dz >= .20
- C dz >= .20
- R dz >= .20
- stable fraction >= .99

Core transport support requires >=24/32 seeds passing in BOTH standard and stress. Current-only closed-loop control is reported separately and implementation-isomorphic equivalence remains a boundary, not a superiority claim.

Higher-order memory/metacognition/planning diagnostics are reported separately and are not multiplied into the core PASS. Explicit timestamps are telemetry and are removed from the scientific core conjunction.

## P2 — causal episodic memory and bounded identity interventions

P2 introduces a versioned extension, EPI_REPLAY, without changing Core v1.1 membership. The task uses a fresh generator with persistent context-specific additive dynamics biases not representable by the frozen A/B-only linear model. EPI_REPLAY estimates a context-specific residual bias from its own stored transitions and uses that retrieved residual in myopic and two-step predictions. BASE is the unmodified R36-R1 agent.

Fresh development seeds 2126001–2126004 verify instruments only. Frozen evaluation seeds: 2127001–2127032.

At the midpoint (after the first four blocks), clone the same EPI_REPLAY checkpoint and same environment state into:
- intact archive
- archive deleted
- archive replaced by a same-body donor with independent experience
- archive replaced by a different-body donor with independent experience

All clones retain identical active A/B models; only episodic archive differs. Primary causal-memory endpoint is paired future tracking-cost damage from archive deletion. Support requires a positive mean with bootstrap 95% lower bound >0 and positive damage in >=24/32 seeds.

Profile-specific episodic-history support requires different-body donor damage minus same-body donor damage >=0 in >=24/32 seeds and bootstrap 95% lower bound >0.

History-fork continuity: exact clones receive either identical or different post-fork experience streams under the same body and dynamics, then are frozen and evaluated on an identical common probe sequence. Different-history action divergence must exceed identical-history divergence by >=.05 in >=24/32 seeds for the bounded computational-continuity claim.

Archive removal is allowed to fail: if EPI_REPLAY is not useful, the negative result is retained. Any PASS means only causal episodic use in this extension, not phenomenal self, universal identity, or core necessity.

## E6b / E7

E6b remains OPEN because all runs are performed by this program. E7 remains OPEN because no held-out biological dataset is supplied. R41 does not replace either requirement with simulations.

## Stop rule

No thresholds, seeds or metric definitions change after the workflow is opened. Infrastructure-only fixes must be documented and cannot depend on scientific outcomes. Full raw per-seed records and provenance are retained.
