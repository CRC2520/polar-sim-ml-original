# R36 preregistration — unified persistent agent

Status: development protocol before confirmatory freeze  
Date: 22 September 2026  
Parent theory: POLAR Core v1.1 = D+C+R, with A conditional.

## Primary question

Can the currently supported reduced core D+C+R coexist, in one persistent agent and one continuous lifetime, with capabilities that earlier campaigns demonstrated only in separate realizations?

R36 prohibits editorial accumulation. Per seed there is exactly one PersistentAgent object, one continuous 1,760-step trajectory and one persistent memory/model state. Lesions are counterfactual evaluations of the same checkpoint and do not substitute separate agents for the main conjunction.

## Environment

Four recurring continuous dynamical regimes are visited in the schedule:

0,1,2,3,0,2,1,3

Each regime has:
- a distinct stable cross-variable transition matrix;
- a noisy context cue, never a regime ID;
- a control target.

The agent also has a persistent hidden actuator profile. A complementary donor profile is used only for the own-history transplant comparator.

Rare exogenous shocks create a source-attribution problem.

## Same-agent capacities

### D — differentiated causal state
A state-collapsing lesion replaces differentiated transition columns with a shared averaged representation. Prediction damage is measured against the intact model.

### C — recurrent causal state
The agent identifies context through a recurrent cue state. A current-cue-only comparator tests whether recurrent context improves prediction.

### R — learned relational organization
The intact learned transition matrix is compared with an off-diagonal relation lesion.

### Persistent memory / recurrence across revisits
On the second visits to regimes, intact persistent models are compared with block-local cold models over the early reentry window.

### Own-history specificity
A donor agent with a complementary persistent actuator profile acquires its own history. Donor-model transplantation is evaluated on the main agent's current trajectory. The donor is a lesion/comparator only; it does not contribute capabilities to the main agent.

### Source attribution
The agent infers whether a transition included an external perturbation from its prediction residual. Balanced delayed source accuracy is evaluated after at least 80 steps.

### Time attribution
Episodic entries carry the agent's persistent internal clock. Delayed pair-order queries test temporal attribution.

### Metacognition
Confidence is computed from sample support and recurrent prediction error. Calibration requires high-confidence transitions to have lower factual error than low-confidence transitions. Confidence also gates myopic versus two-step planning; a reversed-gate counterfactual tests functional use.

### Counterfactual planning
The learned model evaluates two-step action sequences. The selected first action is compared with a myopic action under the true two-step environment cost.

## Development / confirmatory separation

Development seeds: 2006001–2006004.

Confirmatory seeds, unopened until freeze:
2007001–2007012.

The development panel is used only to establish feasible frozen thresholds. No confirmatory criterion may be relaxed after opening confirmatory seeds.

## Pass interpretation

A confirmatory PASS requires the mandatory D+C+R core and the declared persistent capacities to pass jointly in the same seed at the frozen guard level, with a global seed-count requirement.

A failure of any component remains visible; R36 must not report the union of capacities across different seeds as same-agent coexistence.

## Boundaries

R36 tests coexistence of computational functions in one bounded synthetic lifetime. It does not establish phenomenal consciousness, an intrinsic self, open-ended autonomy, universal minimality, independent replication or biological correspondence.
