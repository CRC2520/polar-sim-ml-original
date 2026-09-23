# R40: post-R39 P0-P2 design repair and evaluation

Status: prospective protocol committed before this campaign's training/evaluation. This is a new internal campaign, not a re-run or optimization of the consumed R36-R2/R37/R38/R39 panels. No historical source or adjudication is edited.

Reference simulator commit: d28958279eae052cffcc319146a200614575c030.
Reference manuscript commit: 737c67333f85dd5fa5bd0d7d465b0298fe9f1b54.
Frozen R36-R1 reference runner blob: 99c6ac712e26abbbe7a1202709701e5af94c6ab4.

## P0-A: competent externally implemented recurrent baseline on a NEW external task

Use unmodified POPGym RepeatPreviousEasy (four-suit delayed recall, not the consumed CartPole panel), with sb3-contrib RecurrentPPO/MlpLstmPolicy. Use three independently trained models: seeds 2066001, 2066002, 2066003. Fixed training budget per model: 262144 transitions; 16 environments; n_steps=128; batch_size=256; n_epochs=4; learning_rate=0.0003; gamma=0.95; gae_lambda=0.95; ent_coef=0.01; LSTM64; no hyperparameter sweep, checkpoint selection, curriculum or early stopping.

Twelve development evaluation episodes per model, seeds 2066101..2066112. The model carries LSTM states and resets them only at episode boundaries. The paired NO_HISTORY control resets the LSTM at each decision. A history-only queue controller is an evaluator ceiling, not POLAR and not a fair learning comparator.

Baseline eligibility: at least 2/3 independently trained models must achieve mean accuracy >=0.80 and paired intact-minus-reset accuracy >=0.10. If ineligible, do NOT open confirmation or infer POLAR superiority. If eligible, evaluate all three fixed checkpoints, including any unsuccessful model, on 32 held-out episodes per model (2067001..2067032) at observation corruption 0, 0.10 and 0.20. Corruption replaces the observed suit uniformly, independently; true rewards and unmodified dynamics stay in the environment. Confirmatory baseline gate: at least 2/3 models clean accuracy >=0.80, corruption-0.10 >=0.65 and history benefit >=0.10. This only validates the baseline on the NEW memory task, not the old noisy-control panel.

## P0-B: reproducibility versus independent replication

Export exact code, package versions, full model checkpoints, raw episode metrics and hashes. Recompute adjudication from raw records. A same-program run of third-party software is NOT E6b independent replication. E6b stays OPEN until a separate team submits a separately implemented run. Include a machine-readable handoff and acceptance checklist. No imaginary independent reviewer or team is credited.

## P1-A/B: noisy control and unified-agent evaluation on a new task generator

Generate random stable 3D transition matrices and random regime schedules; control targets are generated independently of hidden regime so the goal cannot disclose regime identity. Dynamics, disturbances and observation noise come from independent RNG streams. Preserve 1760 steps and the frozen R36-R1 D/C/R, memory, confidence and planner implementations. The only proposed architecture change is one predeclared Bayesian cue filter, a conventional 4-state HMM with fixed transition persistence 0.985 and Student-t emission likelihood (df=4, scale=0.72). It has no access to true regime, future observations or source labels.

Compare FROZEN and BAYES_CUE agents on paired environments under standard observations (state noise .005, cue noise .72) and stress (state noise .025, cue noise 1.10, 10% cue outliers). Truth is kept in evaluator records, not active agent inputs. Terminal observations retain the actual final state instead of an artificial zero target.

Use four development seeds 2076001..2076004 for instrument checks ONLY. The source/configuration is fixed before this run, and there is no selection among hyperparameters/variants based on outcomes. If instruments pass (finite outputs, exact lifetime, no leakage and vectorized-planner equivalence), run 32 confirmatory seeds 2077001..2077032. A negative development outcome is retained and does not trigger tuning.

Report BOTH unconditional all-candidate planning gain (the original R36 quantity) and invoked-only gain. The primary legacy-comparable conjunction uses unconditional planning and the original R36 numeric component thresholds, all in the SAME seed. Report the R36-R1 invoked-only conjunction separately. Require >=24/32 full strict conjunctions in EACH observation condition for unified-robustness support. Do not aggregate components across agents, seeds, domains or noise levels to create a PASS. The proposed Bayesian filter is beneficial only if the paired mean tracking-cost improvement has a positive 95% bootstrap lower bound; one observation condition does not establish universal benefit. State estimates, causal ablations and effects remain local to this task family.

## P1-C: causal-state reduction

Delete the archival episode list and last-output bookkeeping from a copied checkpoint, retaining active dynamics buffers, context state, visit counts and calibration state. Compare online actions against the unpruned checkpoint under identical future observations/feedback. Require action/prediction identity to numerical tolerance 1e-10 for an ACTION-STATE reduction result. Separately test old-event query availability; missing archival reports prohibit a FULL-CAPACITY reduction claim. No universal nonlinear minimality claim is possible from this finite intervention class.

## P2-A: identity-continuity interventions

Fresh seeds 2086001..2086004 (instruments) and 2087001..2087032 (confirmation). One common trained checkpoint per recipient. Test exact cloning, archive deletion, metadata redaction, recurrent-state reset, learned actuator-map transplantation, graded actuator-map replacement (0/.25/.5/.75/1) and a declared arithmetic merge. Use BOTH a same-actuator-profile donor with independent experience and a different-profile donor with the same environmental matrices; otherwise generic estimation damage is confounded with identity.

Primary own-profile effect: different-profile transplant prediction damage must exceed the matched same-profile transplant damage by >=1e-4 in at least 24/32 seeds. Report shared-input action and prediction changes for each intervention, including zeros and sign reversals. Fork identity is only computational reproducibility; an arithmetic merge is only an engineering intervention. This campaign cannot establish a phenomenal self or an invariant that no conventional adaptive controller possesses.

## P2-B: prospective biological correspondence E7

No biological measurements have been supplied to this campaign. Do not generate or label synthetic signals as biological evidence. Run only an analysis-pipeline negative-control/power simulation (1000 independent simulated studies, 64 paired units, four outcomes, Bonferroni family alpha .05) and an input-provenance validator. Require null familywise rejection <=.08 and report power for a declared d=.5 alternative. This is software/statistical calibration, NOT E7 validation. Biological adjudication remains BLOCKED_REAL_DATA_PENDING and E7 remains OPEN.

## General adjudication and stopping

All source/configuration hashes are committed before training/evaluation. Save complete per-seed records; no threshold changes after opening. Instrument/infrastructure fixes may occur only when they cannot depend on scientific outcomes, must be recorded, and cannot erase first failures. P0, P1 and P2 are distinct scopes: competent delayed-recall baseline does not validate control; computed timestamps do not demonstrate an emergent time concept; residual source classification does not demonstrate autobiographical awareness. Core v1.1 remains a candidate D+C+R core with A conditional unless a separately justified versioned revision is made.
