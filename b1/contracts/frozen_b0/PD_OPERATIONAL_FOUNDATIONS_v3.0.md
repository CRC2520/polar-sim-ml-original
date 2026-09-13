# Polar Dynamics of Consciousness

## Operational Foundations v3.0 — B0

**Scope:** original research, operational specification. **Source:** `Polar_Dynamics_Investigacion_Original_Borrador.md` at commit `a824ff83a18e0690a6290e9091ac58b0e6b73611`. The source remains the narrative reference. These foundations close a prospective operational baseline; they do not report a new experiment or rewrite historical evidence. The final completion decision and artifact hashes belong to [B0_FREEZE.json](B0_FREEZE.json).

### 1. Scientific object and statement classes

The object of study is whether explicitly related functional processes offer an identifiable, useful organization of bounded adaptive control. A pole is an observable, intervenable process with a declared operation and operational dose. A candidate polarity is $P_k=(b_i,b_j,\Gamma_{ij})$: two processes and a testable relationship contract. Neither the label nor the inclusion of two channels establishes that the relationship matters.

Statements in this baseline have five classes: **historical** (reported source evidence, unchanged); **documentarily derived** (a definition or analytical consequence of specified premises); **new prospective specification** (a task, unit, interface or procedure introduced here); **hypothesis** (a prediction that a future eligible test could reject); and **methodological rule** (an inference or design constraint). The ontology and notation below consolidate the source. Task capacities, deterministic dynamics and interfaces in sections 5–7 are new prospective engineering specifications. Their arithmetic consequences are documentary derivations, never measurements.

O1 means a described candidate. O2 means an operationally specified candidate ready to implement and intervene on. O3 requires appropriately obtained local causal evidence. O4 requires independent-domain generalization evidence. P5, P6 and P7 reach **O2 as specifications**. Their implementation status remains `not_implemented` and no O3/O4 evidence is created here. P1–P4 and P8 remain in the global catalogue with their explicit source gaps; the unexplained distinctive relationship of P8 may remain `no_determinado` without blocking this selected scope.

The full 50 numbered fields for each selected pilot, including the 33 required operational fields and the 17 comparator/inference fields, are canonical in [PILOT_CONTRACTS_v1.json](PILOT_CONTRACTS_v1.json). The original 33-field catalogue remains separately traceable in [POLARITY_OPERATIONALIZATION_MATRIX_v1.csv](POLARITY_OPERATIONALIZATION_MATRIX_v1.csv); its field order is not silently replaced with the prompt's different numbering.

### 2. Ontology, symbols and units

| Object | Operational meaning | Unit or type | Essential separation |
|---|---|---|---|
| $b_i$ | Process with an initiation, execution and effect contract | Typed process | A philosophical name is not an actuator |
| $\widetilde a_i$ | Requested new-operation intensity | Dimensionless request in [0,1] | Precedes admission |
| $a_i$ | Internally admitted normalized operational dose | Dimensionless actuator dose grid | Not completed work or success |
| $d_i^{req},d_i^{adm}$ | Externally requested and physically admitted operation doses | Actuator slots per epoch | May differ despite $a_i=1$ |
| $u_i,w_i$ | New executed operations and continuing operations | Actuator-native units | Zero new initiation does not cancel continuing work |
| $s_t,o_t$ | Environment state and available observation | Task schema | Hidden environment truth is evaluator-only |
| $c_t$ | Externally available context | Typed observable fields | Not integrated internal content |
| $C_t$ | Internally integrated available information | Typed retained representation | Cannot contain unobserved future state |
| $\kappa_t=(c_t,C_t)$ | Ordered context/content tuple | Product type | Does not erase either component |
| $U_t$ | Retained memory | Records, precision and capacity declared | Not learning and not the action set |
| $\theta_t$ | Configured or learned parameters | Component-specific | Memory retention does not imply parameter learning |
| $\Gamma$ | Operational source–recipient relationship contract | Typed record | Not an edge weight or environmental law by name |
| $\mathbf R_t$ | Internal typed relation/routing representation | Routes, payloads and implementation map | Not automatically $W$, $M$ or a reserve stock |
| $E_t$ | Authorization, precedence and admissibility rules | Rules with recorded reasons | Not a performance objective |
| $g_t$ | Externally specified objectives and priorities | Native objective units | Not an inferred intrinsic motivation |
| $\mathcal U_t$ | Feasible external action set | Set of actuator actions | Not memory $U_t$ |
| $\mathcal P_t$ | Concrete restriction/admission operator | Typed map with rejection reasons | Not necessarily a Euclidean projection |
| $\mathcal P_{adm}$ | Admissible internal configurations | Subset of activation space | Not the external action set |
| $\boldsymbol\tau_t$ | Family of tension descriptors | Heterogeneous units | No universal scalar or success criterion |
| $L_k$ | External service failure fraction | Dimensionless [0,1] | Not activation, tension or consciousness |

The construct dictionary supplies aliases, dependencies, observability, manipulability and exclusions in [CONSTRUCT_DICTIONARY_v1.yaml](CONSTRUCT_DICTIONARY_v1.yaml). Reserve in P5 is $r_t$, never $\mathbf R_t$. Historical $W$, content modulation $M(C_t)$, learned action model $\widehat B$, relationship-audit notation $\Pi$, routing representation $\mathbf R_t$ and environmental transition $f_{env}$ retain distinct types. Historical $K_{effective}$ and $W_{effective}$ are derived sensitivities, not additional independent regulators. Historical HGI and INC remain internal descriptors, not revised success criteria. RLS and MPC remain conventional mechanisms.

### 3. Dose, inactivity, time, memory and restrictions

All three prospective task families use a synthetic epoch as the declared time unit. This is an engineering scheduling unit, not a claim about psychological or physical time. Matched utility episodes have 32 epochs, three replicated cells, one decision per epoch and a planning horizon of one future epoch. Each cell has two demanded service jobs per epoch; the fixed primary denominator is therefore $3\times2\times32=192$.

For nominal integer capacity $D_i$, valid requested activation is quantized by $q_i=\lfloor D_i\widetilde a_i\rfloor$, after which internal permission admission produces $a_i=q_i/D_i$. Internally denied requests receive zero and a reason. The external adapter requests $d_i^{req}=a_iD_i$, then separately applies physical feasibility to obtain admitted and executed operations. P5 uses capacities 2 and 2; P6 uses 1 and 2; P7 uses 1 and 1. These denominators are fixed prospective capacities. They are not maxima fitted to observed outcomes. Intermediate requests round down; invalid or nonfinite requests are rejected and logged, never silently clipped or used to renormalize the scale.

Zero means no new initiation by that channel during the epoch. It does not clear memory, erase process capability, cancel prior operations or remove delayed effects. One means admission of the full nominal initiation budget; it does not guarantee environmental execution. Requested activation, internally admitted activation, external requested/admitted dose, new execution, continuing execution, completion and environmental effect are separate log fields.

| Internal regime | Meaning | External qualification |
|---|---|---|
| (0,0) | Neither channel initiates new work | Continuing work and delayed effects may remain |
| (1,0) | A admits its full nominal dose | Demand, stock, permission or target can still block action |
| (0,1) | B admits its full nominal dose | Completion may be delayed or infeasible |
| (1,1) | Both admit their nominal doses | Joint execution depends on the same external constraints as every other arm |

No conservation rule $a_A+a_B=1$ is imposed. Independence means separate accessible operations, not statistical independence or absence of shared resources. Coactivation is not automatically synergy; a net-zero reserve change can coexist with nonzero production and refill flows. Dynamic balance has no universal 50/50 target.

At each epoch, due operations complete first; exogenous events with declared timing then occur; the controller receives current observations, updates retained memory and permitted parameters, and forms current content and proposals. Admission and action follow. New observations cannot influence an earlier decision. Repair completion, deployment completion and assay report delays are explicitly fixed in their task contracts. The state includes every pending job, lock, report queue and persistent cache needed to reproduce behavior.

History is the full event sequence. Memory $U_t$ is a particular retained encoding of that history. Parameter learning changes $\theta_t$ through a separate permitted update rule; it may be disabled. Acute interventions freeze parameters. Scratch-trained comparisons allow compensation using the matched training envelope. Neither losing a message nor changing a pole's activation silently resets memory.

Goals and restrictions are also separate. Minimizing missed service is an objective; stock conservation, authorization, write locks and maximum start capacity are feasibility rules. A pairing lesion never grants a permission, removes a lock or changes the environment's conservation law. If an intervention changes relation, raw information, feasibility or compute together, its estimand is explicitly a `joint_manipulation` rather than a uniquely attributed relationship effect.

### 4. Operational relationships and tension

Each $\Gamma$ declares elements, direction, context, intervention, permitted mediators, recipient observable, prediction, application conditions and falsification conditions. The JSON adds a **routing realization**: the optional internal message used to represent or exploit the relationship. A physical or informational source effect is not the same scientific object as the usefulness of representing it through a nominated internal route.

P5's reserve-mediated enabling relation continues to exist if an internal pending-delivery message is masked. P6's assay produces a report even if a nominated selector route ignores it. P7's clean-source migration precondition remains in the common eligibility adapter even if a repair-completion route is bypassed. A generic controller may reconstruct every useful deterministic feature from equally available raw information. Such reconstruction is allowed, because preventing it would confound pairing with information deprivation.

The substrate diagnostics below have deterministic predicted effects because their environment and fixed mediators are explicitly constructed that way. Checking those consequences is an implementation/instrument check and, at most, a local intervention result in that constructed task. It is not independent confirmation of a natural polarity, a semantic label or organizational advantage. A future causal route lesion asks a separate question: what does using this representation contribute in this frozen controller? Its sign is not guaranteed merely by specifying $\Gamma$.

Tension remains a family: (i) functional mismatch in task units, (ii) requested resource excess in the relevant resource's units, (iii) incompatible action pairs on an identified object, (iv) incompatible objectives in their native units, (v) coactivation together with a declared relationship, and (vi) rejected requests or constraint conflicts with reasons. The components are not added without an explicit dimensionally defensible model. Resource excess is computed from requests, not already feasible admitted doses. Lower tension is not automatically better: unproductive inactivity can eliminate conflict, and a conflict report can improve a decision.

### 5. P5 — Productive consumption / reserve replenishment

The historical name is “Deseo / Límite”; the scientific IDs are P5-A, P5-B and Gamma-P5. A starts up to two service operations per epoch, each consuming one reserve unit. B starts up to two costly replenishment units, each delivered at the next epoch boundary. Reserve capacity is eight. Production and recharge have separate physical lanes; two recharge tokens are available per epoch. Deliveries beyond capacity are recorded as overflow. A full stock does not prohibit requesting recharge, and waste is not silently removed from the trace.

At an epoch boundary, due recharge is added and capped; current production then consumes available stock; current recharge is queued for the next boundary. A has no access to future stock before delivery. The source diagnostic compares two actual recharge units with an equal-token dissipative sham, holding current production off and next-epoch attempted service and demand at two. Under $\kappa_1$, initial reserve is zero: the declared treatment–sham difference in next completed service is two. Under $\kappa_2$, initial reserve is eight: that difference is zero. The normalized recipient is completed service divided by two. Timing or stock observations that contradict these imposed consequences identify an implementation/contract failure, not an unfavorable theory result to relabel.

The nominated internal route carries source-cell ID, scheduled delivery epoch and forecast delivered units to a planning adapter. The ordinary reference calculation plans service from current stock, withdrawals and pending deliveries. Raw delivery records remain accessible to C0, C2 and C3. Source-operation manipulation and route masking are different contrasts.

For utility episodes, each cell's initial reserve is independently uniform on {0,8}; demand is two every epoch. This simple family is deliberately retained as a **ceiling negative control**. The maximum service total is 64 from initial reserve eight and 62 from zero. A conventional serve-and-replenish policy attains these bounds by construction. These are analytical limits of the task, not measured findings. Consequently, positive primary-utility superiority over that reference is unavailable in v1. Equivalence, worsening, conservation checks and correct rejection of spurious pairing superiority remain meaningful. A future more demanding inventory task requires a new prospective version; difficulty is not changed after results to create an advantage.

### 6. P6 — Isolated alternative assay / routine execution

The historical name is “Libertad / Orden”; scientific IDs are P6-A, P6-B and Gamma-P6. A starts one assay of an already authorized immutable routine q0 or q1 in an isolated sandbox. A report arrives next epoch. B starts up to two live executions of the currently selected routine. An assay cannot write production state or deploy a configuration.

A separate, conventional **authorized routine selector** mediates any effect on later B. Choosing between installed routines is a decision within a common permitted catalogue, not a third concealed pole and not P7's version deployment. The fixed diagnostic selector changes to the tested candidate only upon a transfer-eligible positive report; otherwise it retains q0. The organizational comparators may use the same selector API and all equally available live correctness feedback.

In the diagnostic, the live correct routine is q1, the current routine is q0, current production is held off, and q1 is tested versus an equal-cost noninformative sham. Under $\kappa_1$ the sandbox matches the live mapping and its report is transferable; under $\kappa_2$ it returns zero and is declared nontransferable. With the prescribed report mediator and two next-epoch service attempts, the imposed treatment–sham correct-service effect is two versus zero. The normalized recipient is correct service divided by two. A report cannot affect earlier service or bypass authorization.

Utility cells independently draw initial hidden correct routine from {q0,q1}, a change schedule from {none, flip at epoch16}, and sandbox context from {matched, nontransferable}, each uniformly. A sampled hidden mapping and future change realization are not controller inputs. This is distinct from the fixed-q1 diagnostic. The source JSON fixes initialization, feedback timing and all generator support; no distribution is selected by which controller wins.

This v1 full-feedback family is an **information-redundancy negative control**. With two mutually exclusive correct routines, a known executed routine and complete binary correctness, live execution already reveals the current mapping. A contemporaneous sandbox report is a deterministic function of that same mapping and adds no information conditional on that feedback when both become available together. This is an analytical property of the stipulated task. Holding production off in the forced diagnostic can exhibit report mediation, but cannot demonstrate incremental assay-information value in ordinary full-feedback episodes. A conventional selector that uses live correctness is mandatory. A future informative-assay task needs a separately versioned design; the present task must be allowed to reject an unnecessary nominated pair.

The fixed uniform initial mapping and independent probability-1/2 flip at epoch16 also give the last-feedback C4 an optimal **expected** failed-service fraction of $1/32$: one expected wrong job at epoch0 and one at epoch16 per cell, out of64 demanded jobs. All intervening epochs are predictable from equally available feedback. A contemporaneous assay cannot reveal either new independent uncertainty earlier. Consequently, positive expected primary-utility or pairing superiority is unavailable under this particular distribution. This is an expectation bound, unlike P5's per-trajectory ceiling; an individual sampled bundle may favor another policy by chance. Equivalence/worsening and accurate uncertainty accounting remain meaningful. The task is not redesigned to create a positive advantage.

### 7. P7 — Current-version drift repair / authorized configuration deployment

The historical name is “Preservación / Transformación”; scientific IDs are P7-A, P7-B and Gamma-P7. Each cell has two service instances. A starts one repair per epoch: it clears observed drift while retaining the current target version. B starts one deployment per epoch: it changes the target to an approved different version. Each instance supplies one demanded service job per epoch if it is unlocked, nondrifted and at the currently required version.

A repair locks its target for the epoch and completes at its end. A deployment locks its target for an epoch and completes at the next boundary. Same-instance writes are exclusive in every arm. Concurrent requests for repair and deployment on the same instance receive the same repair-priority arbitration and recorded deployment rejection. Operations on different eligible instances may coexist. Thus (1,1) can be internally admitted without both external operations being executable on one object.

Under $\kappa_1$, state-preserving migration requires a clean source; under $\kappa_2$, replacement migration clears drift itself. The diagnostic starts a version0, drift1 instance with candidate1 approved: actual repair versus equal-time locked sham at t0, identical deployment attempt at t1, completion measured at start t2. The imposed difference is one successful deployment under $\kappa_1$ and zero under $\kappa_2$. Both arms retain the same mode-specific eligibility rule. The relation's source manipulation changes the source state; turning off an internal route never changes the gate.

The optional internal message carries source-cell and instance IDs, target version, post-repair drift and completion epoch. The ordinary reference manager repairs a dirty source first when required, then deploys a needed approved version. A generic event manager can use the same state directly. P7 therefore proposes neither a novel repair primitive nor an indispensable historical pairing.

Utility cells independently draw initial drift bits, migration mode, required-version schedule and drift-event choices from the exact finite uniform supports in the JSON. Current required version and observed drift are available; future sampled changes are hidden until observation. No random migration failure or rollback mechanism is introduced in v1. Snapshot records do not establish resilience, safety or general rollback capability.

### 8. Invariants, alternatives and inference

| Separate comparison axis | What remains matched in C0–C4 where a matched claim is made |
|---|---|
| Information | Raw observations, context, source IDs, reports, feedback, forecasts and action repertoire visibility |
| Restrictions and feasibility | Final adapters, permissions, conservation, locks, physical resources, latency and continuing work |
| Learning and training resources | Allowed development data, update schedule, search allowance and training envelope; frozen parameters for acute lesions |
| Memory | Raw event retention, state slots, precision, initialization and reset policy |
| Response and planning dynamics | Episode32, one future-epoch planning horizon, one decision/epoch, same lookahead and interruption permissions |
| Decision compute | Budget, adapter calls, message width and decision deadline; bypass computes the message before masking its nominated use |

Resource ledgers, action sets, communication width, seeds, dataset exposure and adapter implementation are separately recorded; parameter count alone cannot establish a fair comparison. C0 has independent channels with no required nominated route. C1 uses the nominated route. C2 cyclically permutes message recipients across the three cells, preserving count, width and degree while retaining source IDs and raw records. C3 uses a generic matched graph. C4 is the declared conventional domain controller. C5 is descriptive, with exactly doubled memory and compute but unchanged planning horizon and physics; it is not a matched pairing-specificity contrast.

C6 is the exact invertible coordinate control $d=a_A-a_B$, $q=a_A+a_B$, with inverse $a_A=(q+d)/2$, $a_B=(q-d)/2$ and enclosing domain $|d|\le q\le2-|d|$. Both inverse activations must additionally satisfy their pilot-specific discrete dose-grid constraints; the enclosing diamond does not grant a continuous relaxation. It preserves the same action sequence, feasible set and randomness. It must not be treated as a rival representation that needs to lose.

The questions remain separate: **H_mechanism** asks whether the specified intervention changes the declared recipient within its scope; **H_utility** asks about externally measured service under guardrails; **H_pairing** asks whether nominated organization adds value beyond independent, shuffled, generic and conventional alternatives; **H_context** asks about the predeclared moderator interaction; **H_generalization** requires a separate implementation/domain. P5's ceiling and P6's full-feedback redundancy make specified positive superiority claims ineligible in v1; their negative-control role is itself explicit and useful.

The primary metric is failed service fraction in [0,1]. Resource costs, recovery, drift, report use, version changes and other native outcomes remain secondary; they are not silently collapsed into a weighted primary. Practical margins, equivalence, sample size, multiplicity, seed namespaces, stopping, failure accounting and sensitivity are governed by the closed prospective procedures in [B1_PROTOCOL_DRAFT_v1.md](B1_PROTOCOL_DRAFT_v1.md) and [B1_DECISION_RULES_v1.yaml](B1_DECISION_RULES_v1.yaml). Final outcomes may not choose margins, extra observations or inclusion rules. No significance threshold alone establishes mechanism, useful effect or equality.

Independent functions, usefulness of both processes, usefulness of a relationship, necessity of specific pairing, contextual interaction and advantage over a generic graph are six different propositions. Useful components can survive the abandonment of their nominated pair. A useful relationship can be implemented by a conventional controller. A statistical interaction is not by itself emergence. The detailed retain/merge/reformulate/abandon conditions are in [POLARITY_REDUNDANCY_AUDIT_v1.md](POLARITY_REDUNDANCY_AUDIT_v1.md).

### 9. Architecture, historical limits and entry to B1

The source layered architecture remains the reference organization: bounded environmental interaction and conventional regulators; observable operational channels; relation-aware coordination; retained memory and adaptive updates; integrated content and bounded monitoring. A realization must map each proposed layer and route to concrete inputs, outputs, state, timing and interventions. Naming a component does not establish its implementation. None of the present task contracts claims to realize the complete architecture.

| Source architectural role | B0 operational correspondence and scope |
|---|---|
| Environmental adapters and polar channels | Typed dose admission and action requests with declared units; no independent psychological meaning |
| Action and feedback interface | Execute feasible operations, retain continuing work and observe completed service/state effects for the next decision |
| Conscious Integration / integrated content $C_t$ | Assemble only available reserve, report, drift, version and context records; no consciousness attribution |
| Ethical Self-Regulation / rule record $E_t$ | Common stock, authorization, precedence and write-lock adapters; bounded feasibility, no general moral competence |
| Polar Unconscious / retained memory $U_t$ | Event, report, pending-delivery, version and lock queues; no claim of unconscious experience |
| Goals and priorities $g_t$ | Externally supplied service objective and explicit native resource measures; no intrinsic goals |
| Planning and relation coordination | Conventional inventory calculation, authorized routine selector and maintenance manager, with optional typed routes |
| Tension Engine | Separate logged mismatch, resource, incompatibility and constraint descriptors; no universal tension optimizer is implemented |
| IACL / inter-agent coordination | Outside this task baseline; three replicated task cells are not three negotiating agents |

These mappings are prospective correspondences, not claims that the complete named layers have been executed in a new implementation.

Program A concerns bounded control and functional organization. Program B remains **B-design**, requiring a separate theory-specific protocol. Integrated content is not automatically validated global workspace access; confidence is not a validated higher-order representation; a relation graph is not IIT; complexity is not consciousness. No present artifact authorizes claims of phenomenal experience, ASI, AGI, intrinsic motivation, general moral competence, universal polar advantage, natural necessity of eight pairs, global safety or global stability.

Historical findings and identifiers remain unchanged: S2 retains `suspend_exclusive_polar_advantage_claim`; S3 retains `no_confirmed_network_advantage`; integrated P1 retains `engineering_verified_with_partial_empirical_support`; P2's archived status remains `bounded_specific_capabilities`, with calibration not guaranteeing a better policy. The new P5/P6/P7 catalogue IDs are not historical experiment IDs. No historical result upgrades these new contracts to O3 or O4.

B0 can be complete when the selected O2 contracts, contexts, interventions, comparator/invariant inventories, external metrics and closed prospective calibration rules are internally consistent, traced and statically validated. Lack of new empirical evidence is expected, not a blocker. A contradiction in essential operational meaning is a blocker. Tasks that correctly offer no superiority opportunity are reported as such rather than altered to favor the proposal.

`ready_for_b1_design` may therefore be true while `ready_for_b1_confirmatory_run` remains false. Future B1-D implementation/calibration must obey the protocol, pass independent implementation checks, instantiate permitted controller envelopes and freeze code, seeds, margins, sample size, exclusions, hashes and preregistration before any confirmatory B1-E outcomes. This B0 assignment executes none of those experiments. The existing LaTeX/PDF snapshot is not automatically synchronized with this operational baseline.
