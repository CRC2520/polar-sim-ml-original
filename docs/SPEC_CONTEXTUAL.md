# Contextual computational specification v2.1

This is a new, NumPy-only regulator evaluated separately from the historical
`TensionEngine`. It is a functional engineering prototype, not an implementation
or measurement of phenomenal consciousness, psychological polarities, ethics,
life, or ASI. No experimental success here establishes those broader claims.

## State, dimensions and external contract

The default state has shape `(K, P, 2) = (3, 8, 2)`: three computational agents,
eight **types of task-channel pair**, and two independently nonnegative activity
channels per type. There are 48 scalar activity channels, not eight neurons.
Agent and type counts are configurable; only the default eight types have names
in the operational catalogue below. Scalars are dimensionless normalized units;
one cost unit is a benchmark-defined unit, not a physical energy measurement.

`ModelConfig` records dimensions, seed, representation, step size, the three
mechanism enable flags, and memory/capability learning rates.

| Variable | Range and meaning | Required or default | Code |
|---|---|---|---|
| `q[k,p,r]` | `[0,1]`, persistent activity/last executed action | Initially zero | `ContextualPolarModel.q`, `act` |
| `s=q[...,0]-q[...,1]` | `[-1,1]`, signed net orientation | Derived | `polarities.to_signed_intensity` |
| `i=q[...,0]+q[...,1]` | `[0,2]`, total activity, **not conflict/tension** | Derived | Same |
| `target` | `[0,1]`, desired external effects by channel | Required, exact full shape | `act` |
| `observed` | Boolean visibility mask | True; `observed_mask` alias accepted | `act` |
| `cue` | Symbolic episodic retrieval key | None | `EpisodicMemory` |
| `weights` | Nonnegative importance of each effect channel | 1 | `ResourceWorkspace.allocate` |
| `costs` | Strictly positive resource cost per unit action | 1 | `act` resource projection |
| `budget` | Finite nonnegative total resource budget per step | Number of scalar channels | Same |
| `horizon` | Nonnegative response-timescale multiplier | 1 | `propose_action` |
| `allowed` | Boolean admissible action channels | True | `act` |
| `effect` | `[0,1]`, measured effect of the preceding own action | Required feedback | `learn` |
| `gain_hat` | `[0.1,4]`, estimated effect/action ratio per channel | 1 | `CapabilityModel` |
| `counts` | Nonnegative integers, observed own-action samples | 0 | Same |
| `error_ema` | Nonnegative mean absolute prediction error estimate | 0 | Same |

Weights, costs, and horizon can broadcast to the full shape; a vector of `K`
entries means one value per agent. Boolean masks follow NumPy broadcasting.
There is no implicit zero substitution for a missing `target`, missing
`effect`, invalid shape, NaN, or infinity. Unknown **channels** must have an
explicit false visibility mask. Their supplied target values are ignored.
The evaluator's hidden answer key is never used to update memory or planning.

## Eight operational labels

These are declared **synthetic task interpretations**, not empirically validated
maps from psychological terms to computational state. The current environment
provides desired numerical channel effects; the controller does not itself
perform real-world delegation, exploration, preservation, or reward experience.
Every pole is observed as its action effort and measured channel effect.

| Pair | Channel 0 proxy | Channel 1 proxy | Cooperation and conflict condition |
|---|---|---|---|
| Poder / Vulnerabilidad | Independent execution effort | Assistance-request effort | Execute one task while seeking help on another; conflict only with shared limited time/communication |
| Placer / Dolor | Reward-acquisition effort | Loss-monitoring effort | Both may be demanded; acquisition and monitoring can compete for resources; no pleasure/pain phenomenology |
| Integración / Fragmentación | Aggregation effort | Decomposition effort | Decompose and aggregate distinct outputs; shared computation may limit both |
| Control / Rendición | Direct-actuation effort | Delegated-actuation effort | Different channels may use both; contradictory commands to one actuator require an external restriction |
| Deseo / Límite | Throughput effort | Reserve-maintenance effort | Both may meet external demands; conflict depends on resource budget |
| Libertad / Orden | Alternative-exploration effort | Routine-execution effort | Concurrent across different tasks; shared time creates competition |
| Preservación / Transformación | Existing-configuration maintenance | Configuration-change effort | Maintain one subsystem and change another; one value cannot both change and remain fixed |
| Reconocimiento / Autenticidad | Externally specified target effort | Predeclared local-target effort | Targets may agree; disagreement is supplied by the task, not inferred from names |

`POLARITIES` in `polar/polarities.py` is the machine-readable catalogue. A pair's
name does not impose mutual inhibition or automatic consensus. Conflict is an
external action incompatibility (`allowed`) or competition under `costs` and
`budget`. More expressive pairwise action restrictions and domain validation
remain future work; the current Boolean mask cannot represent every real-world
conflict in the table.

Inactivity `(0,0)`, first-pole predominance `(.8,0)`, second-pole predominance
`(0,.8)`, and coactivation `(.8,.8)` remain distinguishable. The last and first
both have net orientation zero, but intensity 1.6 and 0 respectively. Nonzero
intensity alone establishes neither conflict nor consciousness.

## Update equations and code mapping

One step is `act(observation)` followed by `learn({"effect": ...})`. Calling
`act` again with an unconsumed action fails explicitly. When feedback is
unavailable, `skip_feedback(reason)` records that absence and permits the next
step; it does not invent an effect. `reset(seed)` resets all learned state and
random generators. Deterministic observations/feedback and a fixed seed replay
the same sequence, including a shuffled workspace intervention.

1. **Internal cue memory.** Let each cue store value `v`, sample count `n`, and
   channel-wise last-update time `u`. At step `t`, retrieval confidence is
   `c = n/(n+1) * retention**(t-u)`. Effective target is
   `d = where(observed, target, c*v)`; unknown cues return zero confidence.
   This is an explicit ignorance policy, not an observed zero target.
   On visible channels only, a first observation sets `v=target`, subsequent
   observations set `v <- v + memory_learning_rate*(target-v)`; `n <- n+1`
   and `u <- t`. Hidden channels are not written. Raw record values persist;
   the retention factor decays **retrieval influence**, not the stored value.
   `EpisodicMemory.learn/recall/tick` implement these equations. `erase`, `edit`,
   and `shuffle` are logged causal interventions. Shuffling reassigns complete
   records among cue labels and can be an identity permutation; traces retain
   the realized permutation so the evaluator can identify ineffective shuffles.

2. **Learned self-model for action effects.** For the preceding own action `a`
   and measured effect `y`, valid channels have `a > 1e-6`. Compute
   `z=clip(y/a, .1, 4)` and update
   `gain_hat <- gain_hat + capability_learning_rate*(z-gain_hat)`.
   Before that update, prediction error is `abs(y-a*gain_hat)` and updates
   `error_ema` with the same rate; valid counts increment by one.
   `CapabilityModel.learn` implements this. No learning occurs from a zero
   action, and sensor noise/output clipping can bias the ratio estimator.
   Uncertainty is `U=1/sqrt(counts+1)+error_ema`. Planning trust is `T=1/(1+U)`
   and effective gain is `G=1+T*(gain_hat-1)`. Thus estimated capability and its
   uncertainty change decisions, not only reported labels. With self-model
   disabled, `G=1` and capability learning is disabled. `lesion(gain)` replaces
   capabilities and resets uncertainty statistics; it is distinct from disabling.

3. **Contextual recurrence.** With `eta=clip(step_size*horizon,0,1)`, propose
   `z = q + eta*(d/G-q)`. Goals can favor either pole or both. `propose_action`
   is an explicit hook for a matched alternative update rule; all observation,
   memory, self-model, feedback and resource operations stay in `act/learn`.
   `horizon` is a response-timescale multiplier, not a multistep predictive
   planning horizon. There is no attraction to a population mean or HGI/INC.

4. **Separate numerical bounds and action admissibility.** Numerically bound
   `b=clip(z,0,1)` and apply admissibility `p=where(allowed,b,0)`. Invalid/NaN
   proposals raise an error rather than being repaired. Finite clipping is
   logged separately from blocked channels. These operations do not evaluate
   ethics or psychological coherence.

5. **Functional global workspace / resource coupling.** Broadcast the agents'
   weighted demands `D[k]=sum(weights[k]*costs[k]*p[k])`. The enabled workspace
   allocates `B[k]=budget*D[k]/sum(D)`, using equal shares when all demands are
   zero. This is coordination through scarce resources, not agreement between
   states. With workspace disabled `B[k]=budget/K`. The shuffle intervention
   permutes the demands assigned to agents before computing budget shares.
   The trace broadcasts demands, priorities, allocations and mean uncertainty;
   uncertainty affects planning via `G`, not this allocator's formula.
   Each agent uses `a[k]=p[k]*min(1, B[k]/sum(costs[k]*p[k]))`, with scale one
   for zero local demand. Final persistent `q <- a`. The total action cost is
   at most budget, and blocked channels stay zero. Unused agent allocations
   are **not redistributed**, an intentional baseline limitation documented
   in resource efficiency metrics; no optimal allocation claim is made.

6. **Feedback and trace.** `learn` consumes measured effects associated with
   that own action. Any evaluator-supplied `feedback['target']` is deliberately
   ignored. `last_trace` and `traces` contain observations, masks, target
   estimates, before/after state, actions, predictions, full memory and
   self-model states, budgets, intervention events, finite clipping, blocked
   actions, resource scaling, and subsequent effect/capability update. Unreceived
   feedback is `None`; explicitly missing feedback includes a reason. These are
   not zero-valued successful measurements. The task runner must additionally
   log the external answer key/environment state separately from observations.

## Representation and causal controls

`representation="dual_pole"` stores the two channels. `"signed_intensity"`
stores `(s,i)` and decodes `q0=(i+s)/2`, `q1=(i-s)/2`, subject to
`0<=i+s<=2` and `0<=i-s<=2`. The recurrence is identical in decoded coordinates.
The two representations are mathematically equivalent; equality within floating
point tolerance is the expected **null result**, not failure or evidence of a
special polar advantage. State representation and update rule are separate:
`ModelConfig` accepts only these two representations. The recurrent comparator
is a separate subclass using the same dual-channel state with a different rule.

The evaluation supplies a distinct projected-gradient recurrence through the
proposal hook. Any difference versus that comparator identifies differences
between update rules and experimental settings; by itself it cannot establish
an advantage specific to polar organization. All mechanisms here can also be
used by nonpolar controllers.

`reverse_convention(types)` swaps stored q, memory values/counts/timestamps,
and capability estimates/statistics for selected pairs. Future targets, weights,
masks, costs, gains, effects and semantic labels must also swap those same
channels. Resource sums remain unchanged. `tests/test_contextual_model.py`
tests equivalence with learned state, not only initially empty arrays.

Memory interventions act after the original target is removed while equalizing
the immediate controller state across controls. Capability lesions affect
predicted gain and thus planning. Workspace ablation affects resource shares
only when demands/budgets make shares consequential. The test with abundant
resources ensures no artificial difference is claimed there. Hidden-target
perturbation tests ensure that unobserved answer-key values cannot affect action
or memory. Individual unit tests establish functional consequences, not broad
task superiority; paired multiseed held-out evaluation is separate.

## API example

```python
import numpy as np
from polar import ContextualPolarModel, ModelConfig

model = ContextualPolarModel(ModelConfig(seed=7))
target = np.full((3, 8, 2), 0.3)
action = model.act({"target": target, "cue": "context-a", "budget": 18.0})
# Supplied by the environment, not fabricated by the controller:
effect = np.clip(action * 0.8, 0, 1)
model.learn({"effect": effect})

model.memory.erase("context-a")  # logged causal intervention
model.workspace.mode = "disable"
model.self_model.lesion(gain=1.0)
```

Verification: `python -m unittest discover -s tests -p test_contextual_model.py -v`.
No HGI, INC, consciousness score, or ethics score enters the controller. Historical
HGI/INC remain separately named descriptions of spatial/temporal similarity in
the evaluation specification; external performance, changes, memory effects,
prediction error and resource costs must be assessed independently.
