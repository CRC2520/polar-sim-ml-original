# B1 Development / Calibration — implementation specification

**Scope:** B1-D = **B1 Development / Calibration**. This implementation prepares a future B1-E = **B1 Confirmatory Evaluation** but supplies no final-seed generator or B1-E execution command. The letters D/E denote these stages, not prospective manuscript experiments D/E. All resulting task records carry `development_only=true`, `confirmatory=false`, and `reusable_as_final=false`.

This is a new, isolated realization of the operational contracts. It does not alter Study 1–3, P0–P2, `integrated_polar/`, the historical result files, or the immutable B0 content. A successful instrument test establishes the tested implementation property; it does not establish a natural polarity, consciousness, architectural superiority, or O4 generalization. P5/P6/P7 entered this implementation as O2 specifications. No automatic evidence-level promotion follows from executing this code.

## 1. Immutable sources and handoff

| Source | Immutable identity |
|---|---|
| Code source | `CRC2520/polar-sim-ml-original` at `c958049743f989b5c1f8866018ea48b60bae9ed7` |
| Historical P0–P2 realization | `05caf7abef6fdcb769604539a22cdcd1cfd6b397` |
| B0 publication | `CRC2520/POLAR_MODEL_CRC` at `c6ba1e836b6c96636ca29be7913c9c5ab9facc62` |
| B0 frozen content | `b4796d881ee2581367fc8dd7396521dac30d827a` |
| B0 narrative source | `a824ff83a18e0690a6290e9091ac58b0e6b73611` |
| B0 freeze SHA-256 | `7d9abe376401ca53c8e8f9f519b75c6bb227dfc49749059bb72eb24050978cc1` |

`contracts/frozen_b0/` contains byte-preserved copies of the five canonical source artifacts and the B0 manifest: `PILOT_CONTRACTS_v1.json`, `CONSTRUCT_DICTIONARY_v1.yaml`, `B1_PROTOCOL_DRAFT_v1.md`, `B1_DECISION_RULES_v1.yaml`, `PD_OPERATIONAL_FOUNDATIONS_v3.0.md`, and `B0_FREEZE.json`. `contracts/loader.py` rejects duplicate keys, malformed/nonfinite JSON, a changed B0 manifest hash, missing or changed canonical files, inconsistent source commits, and inconsistent pilot registries. Controller imports verify once and access copied contract properties; stage gates separately verify frozen implementation/configuration hashes.

The imported B0 state must remain `B0_COMPLETE`, version `3.0.0`, with `ready_for_b1_design=true` and `ready_for_b1_confirmatory_run=false`. Its `historical_evidence_modified=false`, `scientific_code_modified=false`, and `new_experiments_run=false` describe the B0 freeze. New B1 implementation and development execution are recorded separately, never written back into that manifest. The B0 pre-freeze count of 9,630 checks and post-freeze count of 9,752 checks describe different verification moments; neither number is changed to make them equal.

The historical decisions retained in every new B1 interpretation are:

| Historical study | Exact decision/status |
|---|---|
| P1 | `engineering_verified_with_partial_empirical_support` |
| P2 | `bounded_specific_capabilities` |
| S2 | `suspend_exclusive_polar_advantage_claim` |
| S3 main | `no_confirmed_network_advantage` |
| S3 active | `no_confirmed_network_advantage` |

The P1 decision must not be assigned to P2. `manifests/HISTORICAL_BASELINE.json` inventories the complete original source tree. `validate_b1.py` distinguishes full Git ancestry/working-tree verification from a partial sparse-workspace check. A partial local check is not a claim that the remote repository was verified.

## 2. Namespace and artifact mapping

| Concept | Implementation responsibility |
|---|---|
| `contracts/` | Immutable B0 copies, strict parsing, hash checks, canonical constants/rules |
| `tasks/` | Independent P5/P6/P7 state machines, common admission, physical gates and event channels |
| `controllers/` | C0–C5 decision families, exact C6 conjugacy, explicit resource limits, six-axis audit |
| `interventions/` | Deterministic source/sham diagnostics and their prescribed contexts |
| `metrics/` | Frozen primary denominators, paired differences, guardrail accounting |
| `evaluation/` | Development episodes, trace archives, precision plan and fail-closed decision interpreter |
| `seeds/` | Development-only derivation/usage ledger, historical seed exclusions and calibration access gate |
| `tests/` | Deterministic instrument, corruption, invariant, preservation and decision-rule fixtures |
| `development_results/` | Execution records, complete retained development traces and stage reports |
| `manifests/` | Historical inventory, implementation/configuration freezes and execution provenance |
| `run.py` | Staged development CLI; no B1-E command |
| `validate_b1.py` | Read-only structure, hashes, provenance and historical-preservation validation |

The published result, audit and manifest files describe what was actually executed. This specification describes the implementation and interpretation limits; it does not substitute expected outputs for observed results. Development artifacts and their source hashes are retained even if a configuration fails or a stage stops.

## 3. Common task and observation API

Each task is a separate cell. Construct `P5Task(...)`, `P6Task(...)`, or `P7Task(...)`, call `observe()` at a decision boundary, and pass an action dictionary to `step(action)`. `observe()` is idempotent within a boundary: it matures due operations once and exposes only the current authorized public observation. `step()` starts admitted operations, records current service and effects, advances one epoch, and retains the event. A bundle contains the canonical three cells and 32 epochs, with two demanded service jobs per cell per epoch. The primary denominator is therefore the frozen 192 demanded jobs, regardless of eligibility or success.

Common action fields are independent requested intensities `A` and `B`. They are not constrained to sum to one. A finite request in `[0,1]` is internally admitted as `floor(D_i * request)/D_i`, subject to the internal permission. The common external adapter then applies physical feasibility and authorization. Full admitted intensity means the nominal budget was admitted; it does not imply execution or success. Zero means no new start by that channel; retained memory and delayed work can persist.

Every event separates:

1. `requested_activation`;
2. `admitted_activation`;
3. `external_requested_dose`;
4. `external_admitted_dose`;
5. `executed_operation`;
6. `continuing_operation`;
7. `environmental_effect`.

The proposal, public state before/after, internal memory, integrated route content, feedback, attempted requests, admitted operations and physical effects are separately traceable. The controller never estimates activation by dividing realized success by a favorable outcome. Failed service stays in the demand denominator.

All matched controllers receive the same complete public observation for all three cells, current context and prior public records. P6 latent live mappings and future flips, and P7 future drift/version schedules, remain task-internal evaluator state. The generator samples exogenous events independently of the chosen controller action. External `c_t` appears in the task context; internal `C_t` is derived only from available observations and the declared route/decision state. They are not interchanged.

## 4. Pilots, operations and limits of their evidence

| Pilot | Operational realization | Scientific role in B1-v1 |
|---|---|---|
| P5 | Productive consumption / reserve replenishment | `engineering_ceiling_negative_control` |
| P6 | Isolated alternative trials / routine execution | `information_redundancy_and_expected_ceiling_negative_control` |
| P7 | Drift repair / configuration deployment | `operational_coordination_candidate` |

Historical philosophical labels are aliases, not executable definitions. All physical capacities, delays, catalogue values, instance counts and primary metric constants are imported from B0.

### P5

Service consumes available reserve. Replenishment occupies its own lane and arrives one epoch later, subject to the reserve cap; overflow and spent replenishment tokens remain visible. The diagnostic holds productive service off initially, applies recharge versus an equal-token sham, then measures the fixed next-epoch service attempt. Empty-stock and full-stock contexts are separate, declared fixtures. Their predicted diagnostic effects are imposed by the synthetic law and measure faithful instrumentation, not an independently discovered ontology.

The two B0 C4 descriptions disagree. The JSON requests the minimum refill necessary for the next demand and zero refill in the final epoch; the protocol fixes an always-service/always-refill policy. They can attain the same primary physical ceiling while differing in actions, resource use, overflow and traces. This is `B1-CONFLICT-P5-C4-REFILL-001`. The affected C4 constructor raises `ContractConflict`; no interpretation is silently selected and no frozen B0 file is edited. Named `json_minimal_refill_witness` and `protocol_full_refill_witness` may inspect the two laws as unresolved ceiling fixtures. They are not a selected C4 architecture comparison.

P5 positive utility superiority and positive pairing specificity are ineligible. A result apparently exceeding the physical bound requires an implementation, feasibility, leakage or measurement investigation. P5's nominated route records a next-epoch inventory forecast; the current realization does not establish a behavioral dependence of the current request on that route. P5 architecture interpretation remains blocked with the unresolved C4 contract.

### P6

The assay and routine service are separate positive operations. Assays produce a delayed, source-indexed report with candidate, value, transferability and age. Routine choice uses a common authorized selector over already installed immutable routines; it is not a P7 deployment. Full live correctness feedback identifies the preceding epoch's mapping. Report transferability is public; the hidden realized mapping and future flip are not.

C4 uses the lowest-job-ID consistent live feedback; conflicting deterministic feedback is an implementation failure. With no live feedback it uses the newest eligible positive report, or retains the current routine. It requests full routine service and does not introduce a weaker tuned replacement for the frozen conventional policy. In the exact utility law, the initial uniform mapping and independent possible flip at epoch 16 give an expected minimum loss of `1/32`. This is an expected bound, not a per-bundle bound. An individual difference favoring C1 is compatible with it.

The forced-routine-off diagnostic tests report delivery and the selector mediator. A same-epoch assay is redundant given complete contemporaneous live feedback in the utility task. Positive expected utility superiority, pairing specificity and incremental assay-information superiority are therefore ineligible in B1-v1.

### P7

Repair positively clears drift while preserving the current target version. Deployment requests a different approved target version and follows the fixed mode-specific source condition. Repair holds the write lock throughout the initiation epoch and completes at its end; deployment completes at the next boundary. Same-instance concurrent requests receive the common repair-priority/deployment-rejection rule. Compatible different-instance repair and deployment can run concurrently.

The conventional scheduler handles ordinary drift even when the current version already matches demand, leaves a clean correct version unchanged, checks approval, and uses mode-aware prerequisite repair for a version change. It selects the lowest eligible instance ID per operation. Replacement migration and state-preserving migration remain distinct contexts. Typed repair evidence cannot override a later observed drift, target mismatch, lock or final gate.

Version, target, object identity, drift, repair, deployment, conflict, authorization, snapshots and completions remain logged. There is no random deployment failure or automatic rollback in the frozen law. Zero rollbacks in this task do not establish rollback robustness. P7 is eligible prospectively for mechanism, utility, pairing and context questions; eligibility supplies no expectation of success.

## 5. Controller family and adaptation protocol

`Controller(pilot, comparator, config_id, cycle, lesion, limits).act(observations)` returns one action for each of the three cells. The per-decision report records routes, source IDs, route consumption, bypass status, operation usage, memory use and declared limits. Actual C6 construction is `C6Controller(Controller(pilot, "C1", ...))`.

| Arm | Implementation and permitted interpretation |
|---|---|
| C0 | No nominated receiver input. It retains all public source records and can reconstruct an ordinary useful relation from raw observations. |
| C1 | Same-cell typed route plus the same raw observations and conventional action interface. No useful dependency is presumed. |
| C2 | Both fixed degree-preserving three-cell cycles, with original source IDs and raw records retained. Evaluation averages both cycle losses and never selects the worse cycle. |
| C3 | Three generic typed edges; the frozen eight-slot family includes identity connectivity and other permutations. Recovery of the nominated graph does not prove its exclusive necessity. |
| C4 | Fixed conventional policy, not an eight-slot search. P5 is blocked by the canonical conflict; P6 and P7 implement their complete stated domain policies. |
| C5 | Generic controller with six message slots, two raw snapshots and exactly twice C3's declared memory and useful-operation allowance. Same physical actions, observations and horizon. Descriptive capacity frontier only. |
| C6 | Exact image of the specified C1 realization. It has no independent tuning procedure and is not a competitor expected to lose. |

The tunable family has eight predeclared configuration IDs. The grid crosses receiver policy (`domain_aware`/`reactive`), source schedule (`on_demand`/`proactive`), and tie direction (`lowest`/`highest`); C3/C5 connectivity choices are part of the same eight slots. Domain-aware C0 can reconstruct conventional relations. P7 never repairs clean objects or performs gratuitous version changes merely to force every configuration to produce a distinct action: some slots may realize identical behavior on a given task.

These are explicit finite decision-rule configurations. Competitive adaptation means selection using the frozen training/tuning opportunity; it does not mean neural training or eight fabricated fits of an analytical controller. Every tunable slot receives the same declared histories and selection opportunity. The search family and implementation hashes must be frozen before comparator scores. Selection uses mean primary tuning loss subject to guardrails, then useful-operation count, then lexicographic configuration ID. Failed configurations remain in the ledger. An implementation change after score access must be recorded and cannot silently replace the frozen search family.

Acute lesions are a different intervention. They retain the selected configuration and all raw records, compute the message, and mask its nominated use while permitting charged raw reconstruction. P6/P7 can consequently produce identical actions under intact and bypassed routes. A null lesion is legitimate. A lesion difference identifies dependence of that particular realization; it is not a competitive retraining comparison or a proof of architectural superiority.

## 6. Exact C6 and numerical specification

The transformation is `d=A-B`, `q=A+B`, with inverse `A=(q+d)/2`, `B=(q-d)/2`. The continuous domain is the diamond `abs(d) <= q <= 2-abs(d)`. Admitted activations must additionally map to the original capacity-specific discrete dose grids. The enclosing diamond does not relax the actuator grid.

`encode_tree`/`decode_tree` apply a tagged invertible transformation to numeric A/B pairs throughout the realization. They also transform the paired requested/admitted activation fields in nested task events. Unpaired environmental quantities, categorical permissions, identities, timestamps and physical resources use the identity block of the transformation. Parameters, memory, raw observations, route state, actions and constraints are preserved under this complete mapping rather than being selectively dropped.

`C6Controller` stores only the transformed realization state between decisions. Its update is the exact conjugate of the original update: decode the input and state, execute the specified original transition, then encode the state and action. `ConjugatedTask` similarly stores the complete transformed task state and conjugates boundary observation and environment transition. Hidden schedules stay private task state; they are not added to the observation.

The implementation uses exact `Fraction` arithmetic for the coordinate additions, subtraction and division by two. Finite binary floating inputs are represented by their exact rational values. The declared absolute and relative coordinate tolerances are both zero; IDs, counts, admission and categorical effects require exact equality. Primitive coordinate operations are explicitly these additions/subtractions/divisions, so no empirically enlarged envelope is needed for this exact representation. The discrete task intensities and primary count fractions are retained exactly internally. Standard logarithm/square-root precision calculations are separate statistical computations, not C6 fidelity tolerances.

Recursive encoding and rational arithmetic have real implementation overhead. They are reported separately from the original useful-operation accounting. Equality under conjugacy establishes algebraic fidelity, not a CPU-speed, memory-efficiency or physical superiority claim. A discrepancy is an implementation error and blocks interpretation; it is not favorable evidence for polar coordinates.

## 7. Six invariants, resource accounting and unresolved calibration

Each contrast has machine-readable records for `available_information`, `restrictions_and_feasibility`, `learning_and_training_resources`, `memory`, `response_and_planning_dynamics`, and `decision_compute`. Additional fields explicitly record `action_space`, `communication_width`, `planning_horizon`, `adapter`, `training_data`, `hyperparameter_search`, and `latency`.

The current engineering `Limits` default declares 4,096 memory scalars, 8,192 useful-operation proxy units and one future epoch of planning; C5 doubles the first two allowances and retains the horizon. The operation counter charges declared scalar copies/reads, record inspections, route slots and decision primitives. It is a transparent algorithmic proxy, not machine instruction counting. Unused allowance does not transfer to another arm. Exceeding the declared memory or operation allowance fails the controller before its actions return.

These numeric caps are provisional engineering allocations. They are not a certification that the cap is the smallest common allowance accommodating every valid worst-case input. Runtime profiling does not establish that missing mathematical property. Until the required compute certification, training provenance, configuration freeze and latency calibration are recorded, `matching_passed=false`; equal provisional caps do not make the audit pass. Any future certified cap or numerical deadline must be frozen before the relevant evaluation access. The deadline procedure is the canonical twice-maximum development timing rule, not a limit expanded after unfavorable final outcomes.

Every unequal axis is exposed. C4's analytical fixed-policy exception is explicit rather than disguised as eight searches. C5's capacity differences restrict it to descriptive use. If a purported Γ intervention also changes raw information, feasibility, memory, compute or action repertoire, `joint_manipulation=true` prevents a Γ-specific verdict even when the change was labeled “expected.” A declared source operation versus sham is separately scoped: its intended operational dose and downstream mediator are not confused with a selective internal route intervention.

## 8. Development data, calibration access and failure accounting

Only `PD-B1-D-v1` is implemented. The derivation takes the first eight bytes of the SHA-256 of the canonical joined development fields as an unsigned big-endian integer. The `source_commit` in that derivation is the B0 narrative source prescribed by B0; it is not silently replaced by the new code commit. Task event substreams encode cell, episode, epoch and event identity. A complete per-use ledger records role, bundle, namespace and derivation hash. A collision with a registered historical final or another distinct development stream blocks instead of generating an unregistered replacement.

Bundles 0–31 are the development/tuning block. Bundles 32–63 are effect-blinded technical calibration. `DevelopmentSeeds` refuses calibration derivation until it receives a content-verified tuning-closure token covering implementation, configuration and the completed tuning ledger. Opening calibration is not a label change and does not itself draw a seed. The final namespace, final CSPRNG entropy and final seed generation are unsupported.

The blinded block assesses instrumentation, stability, runtime/resources and the permitted technical tolerance procedure after tuning closes. It does not select a new task, new polarity or more favorable comparator. Consumed calibration is logged. A need to redesign the construct or task requires a new prospective development version; consumed data are not renamed an untouched calibration block.

The development runner retains demanded jobs and failures. A controller exception, malformed output, unauthorized attempt, exhausted decision budget or missed declared deadline receives the frozen controller-failure loss of one for the affected registered episode. A normal physical reduction/rejection receives its actual missed service and rejection trace; it is not automatically treated as a crash. Task-law assertion failures are instrument failures and stop the collection rather than being recoded as weak controller outcomes. Missing or corrupt records do not become favorable complete-case subsets.

Hard execution guardrails separately count unauthorized executions, invalid concurrent writes, executed infeasible actions, physical-state violations and undeclared budget overruns. Rejected invalid attempts and actually executed invalid operations are different records. No utility gain purchases a guardrail violation. The implementation does not automatically retry infrastructure failures or replace seeds; any future supported retry would have to preserve the same seed/code/configuration and both attempts under the frozen policy.

## 9. Metrics, precision and decision semantics

The primary loss uses the registered failure fraction with a fixed demand denominator. Replenishment input/overflow, assay costs/report use, repair/deployment counts, downtime, failed requests and rollbacks are secondary observations. They are not combined into a newly weighted utility after viewing results.

`evaluation/precision.py` expands and verifies the frozen registry of 27 scalar estimates, supports, margins and target half-widths. It calculates the canonical Hoeffding-plus-union-bound fixed-sample requirement. The positive mechanism/context margins, negative-direction utility margins, strict equivalence intervals and boundary behavior are taken from `B1_DECISION_RULES_v1.yaml`. The requirement is a conservative precision calculation, not a statistical-power claim. Its computed `B1E_required_N` is a planning artifact; calculating it does not allocate final seeds or run B1-E.

If the precision requirement cannot be afforded, readiness remains false. The implementation cannot shrink N merely to finish. A more efficient inference method needs an independently justified prospective protocol version and freeze before final outcomes or seeds. There is no outcome-driven sample extension.

The predicate interpreter uses typed comparisons and three-valued logic. Missing, null, malformed or unverified input cannot default to support; `True` is not accepted as numeric `1`. Synthetic rule fixtures test that the frozen decision language can distinguish its allowed outcomes. Actual `B1-D` records never obtain confirmatory verdicts, even if their numeric values happen to satisfy a confirmatory predicate.

Mechanism, utility, pairing, context and generalization stay separate. Positive P5/P6 utility/specificity claims remain ineligible. P7's future utility requirement compares C1 with both C0 and C4; its pairing requirement includes C0/C2/C3/C4 under valid matching. Weak, incomplete, unmatched or imprecise comparisons suspend the applicable claim. `H_generalization=not_established` remains unchanged by these synthetic development tasks. Source-law diagnostics cannot establish consciousness, ASI, AGI, natural validity of eight pairs, universal stability or global safety.

## 10. Stage and publication interpretation

Use the staged CLI and the generated artifact hashes in the accompanying README. Instrument checks precede architecture benchmarking. A failed instrument gate yields `B1D_BLOCKED_IMPLEMENTATION`; the P5 canonical conflict independently blocks the affected architecture comparison. The absence of a certified invariant is a real readiness limitation, not permission to manufacture a passing audit.

The final exported `B1D_RESULTS.json`, `B1D_AUDIT.json` and `B1D_MANIFEST.json` are authoritative for executed stages, consumed bundles, observed failures, actual counts and readiness. This specification intentionally does not hard-code a test count or assert completion of a stage merely because its code exists. The documentary repository supplies the handoff report, development report and prospective confirmatory package. Any eventual B1-E execution still requires resolution of applicable conflicts, verified instrumentation/matching, viable resources, complete preregistration and a new authorized final-seed lifecycle.
