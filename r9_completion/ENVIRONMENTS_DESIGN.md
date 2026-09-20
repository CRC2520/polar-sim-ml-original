# R9 scalar environments: prospective operational design

These are three new scalar physical realizations, not byte-equivalent versions
of the R8 collective simulation. Ecological depletion, stored vitality and
delayed logistic regeneration continue the R8 mechanisms conceptually. There
are no groups, transmission, inherited Q tables or demographic replacement.
Numerical results must not be presented as direct R8 replications.

## Public interface and fixed domains

```python
from r9_completion.environments import ScalarEnvironment
env = ScalarEnvironment(domain="ecology_train")
obs = env.reset(seed=950081)          # optional domain=... override
obs, reward, terminated, info_eval = env.step(action)  # integer 0..3
```

`obs` is a float64 vector of length three: reserve, health and current demand
signal. Reserve and health use physical unit capacities, with larger values
better. Demand is divided by the fixed maximum actuator flow 0.12. Independent
sensor noise has standard deviation 0.012 on each normalized coordinate;
sensors saturate to [0,1]. This saturation does not alter latent physics.

The controller receives only observations, its own past actions and realized
rewards. It must not receive the environment object, domain label, `info_eval`,
true demand, climate regime, temperature, queues, resource history, seeds,
tapes or future observations. Sixteen learned state features, if used, belong
to the separate agent implementation and are not engineered environment inputs.

| Harness domain | Physical family | Hidden memory | Permitted use |
|---|---|---|---|
| ecology_train | Reserve and stored energy | Resource lag 5 | Training and development |
| ecology_delay9 | Same ecology, only lag differs | Resource lag 9 | Frozen delay shift |
| inventory_transfer | Fulfillment, procurement and liquidity | Pending orders, lag 7; backlog | Zero-shot transfer |
| thermal_transfer | Battery dispatch and equipment health | Thermal inertia 0.95 | Zero-shot transfer |

Transfer families must not provide fitting episodes, normalization statistics,
architecture choices or tuning targets before the controller freeze. Physics
fixtures and an explicitly privileged feasibility witness are permitted; they
do not rank candidate controllers or establish comparative performance.

All four actions request throughput `u = [0, .04, .08, .12]`. Executed flow is
limited by physically available reserve and, for inventory, current demand plus
backlog. Request and executed flow are recorded separately. These common action
semantics reduce an arbitrary transfer confound, but do not make the different
physical processes identical.

## Temporal contract and random tapes

At decision time t, `obs_t` measures current reserve, current health and current
demand, before choosing action t. `step` applies action t to the same demand and
the time-t exogenous tape, then returns factual reward t and `obs_(t+1)`.
Initial state is t=0; termination is exactly after step 319. The final returned
observation at t=320 is available but does not generate an additional reward.

All randomness is allocated at reset from NumPy SeedSequence `[seed, 9901]`.
Arrays have length 321 and are immutable during ordinary execution. For an identical environment seed, base tapes have the same generation
law across domains; deterministic domain transformations produce different
physical inputs. The runner pairs the two ecological domains with the same
environment seed. Each transfer family has its own domain-derived seed. All
variants within one domain share that domain's tape. Actions, death and branching never
consume random numbers. `tape_digest()` identifies the complete tape. Paired
policies therefore experience the same exogenous innovations even when their
states diverge.

The latent binary regime switches with probability .035 per step. Storms start
with probability .018 and last 6–16 steps. Two independently randomized phases
govern periods 83 and 47. With bounded uniforms e in [-1,1], base demand is
`d0 = .027 + .008 regime + .004 sin(2πt/47 + phase) + .003 e`.
Ecology adds `.004 storm`, inventory `.008 + .009 storm`, and thermal
`.011 + .018 storm`. The policy observes a noisy measurement of current demand,
not the regime or future demand. Regeneration, charge, ambient heat and order
fulfillment use their own fixed tape entries; sensor noise is separate.

## Physical equations and event order

Physical stocks have capacity one. Overflow is explicitly recorded as spill;
withdrawal never exceeds stock. Capacity limits represent discarded material
or energy, not a guarantee of viability. There is no positive stock or health
floor. Negative raw health, excessive inventory backlog or dangerous temperature
can cause irreversible failure. Numerical nonfiniteness raises an exception
rather than silently repairing the state.

### Ecology

Initial reserve R=.70 and energy H=.60. Requested extraction is u and actual
extraction x=min(R,u). Logistic growth uses the oldest saved stock L, before
appending the new stock:

`g = .27 + .045 sin(2πt/83 + phase) - .045 storm`

`R_raw_next = R - x + g L (1-L)`

The upper capacity spill is removed. Energy converts extraction at efficiency
.90, pays current metabolic demand d and actuator cost `.002 (action/3)^2`:
`H_raw_next = H + .90 x - d - cost`. Energy above one spills; H_raw_next≤0
causes failure. The resource history is initialized to the initial stock.
An extraction impulse at t changes its delayed regeneration contribution first
at t+5 in training and t+9 in the delay shift. Metabolic service is the fraction
of current d payable from available energy, capped at one. A constraint is
recorded for failure or reserve below .10. Reward severity is the realized
energy deficit divided by current demand, capped at one.

### Inventory

Initial stock R=.55, liquidity H=.80, zero backlog and an empty seven-slot queue.
Each step first receives the queue's oldest shipment and discards capacity
overflow. It then ships `x=min(available_stock,u,current_demand+backlog)` and
loses `.002` of the remaining stock to spoilage. Action also places a new order
of volume u, paid immediately. The accepted volume is `u f`, where fulfillment
f is fixed by the time-t tape in [.96,1]. That accepted order arrives at t+7;
the procurement shortfall is separately recorded. No future tape is consulted
at order time.

`backlog_next = max(0, backlog + d - x)`

`H_raw_next = H + .12 x - .08 u - .0015 - .015 max(0,d-x)/d`

This couples shipment income, procurement cost, operating overhead and current
missed service. Ordering can have a real delayed benefit and an immediate
liquidity cost. Service is `min(x/d,1)`; reward severity is backlog/.30 capped at
one. Backlog above .10 is a constraint; above .60 causes failure, as does
nonpositive raw liquidity. The policy cannot observe the pending orders or
backlog directly, although its history can contain information about them.

### Thermal microgrid

Initial battery R=.60, equipment health H=.80 and latent temperature T=.20.
Actual dispatch is x=min(R,u). Delivered demand is min(x,d); excess dispatch is
explicit wasted output. Remaining battery loses .001 to leakage and then
receives exogenous charge

`c = .047 + .015 sin(2πt/83 + phase) + .006 regime - .018 storm + .002 e`.

Capacity overflow spills. Charge is strictly positive under this tape support,
but it does not guarantee that demand or thermal constraints can be satisfied.
Temperature has memory and responds to current dispatch:

`T_next = .95 T + 4.5 x² + ambient`, with ambient in [.002,.0045].

`overload = max(0,T_next-.65)`

`H_raw_next = H + .004 - .012 (1-service) - .08 overload - .001(action/3)²`.

Service is delivered/current demand. A constraint occurs when unmet demand
exceeds .20 or temperature exceeds .65; temperature above 1.20 or nonpositive
raw health causes failure. Temperature is not clipped at its safety threshold.
Reward severity is the greater of unmet-demand fraction and overload/.35,
capped at one. The temperature may respond immediately while cumulative health
damage and the persistence of earlier heat create delayed consequences.

## Reward and evaluation

For a living post-action system,

`reward = service × [1-.25(action/3)²] × [1-.5 severity]`.

All three factors have fixed analytical bounds, so reward belongs to [0,1].
There is no min/max fit from observed episodes and no future-outcome reward
label. Service and severity are realized outcomes of the current transition.
The reward is a specified task utility, not a claim that all domains share an
identical natural unit or that this is the uniquely appropriate utility.

Failure is absorbing: subsequent actions execute zero flow, health and reward
remain zero, and alive remains false. Physical supply may continue passively;
the episode still runs to 320 steps. A failed transition earns zero reward even
if some flow occurred before failure. Summaries divide accumulated return,
alive time, resource, service and constraint counts by 320. No surviving subset
or early-stop denominator is used. `evaluation_summary()` explicitly marks
unfinished episodes and preserves this fixed denominator.

Evaluation-only info includes `alive`, `reserve`, `constraint_violation`,
`service`, `health`, `state_snapshot`, physical flows and conservation residual.
Convenience aliases are `resource`, `viability`, `task_reward`, `constraint`.
The snapshot contains latent state for audit, not observations for the actor.

## Prospective physics acceptance, not controller selection

Fixtures reserve seeds 950081–950096 exclusively for physics development; none
may be used as untouched final seeds. Tests check exact replay, action- and
domain-independent tapes, no future suffix leakage, exact action-to-lag timing,
observational aliasing with different latent futures, mass conservation,
requested/executed flow limits, thermal inertia, irreversible failure and the
fixed horizon. They do not claim that recurrence will outperform feedforward
policies merely because aliasing exists.

The hand-fixed privileged witness uses true energy for ecological extraction,
true backlog for inventory throughput, and current true demand and temperature
for thermal dispatch. It consults no future tape and receives no learning or
tuning. Its exact rules precede fixture execution in `test_environments.py`.
The acceptance criterion is alive fraction≥.80 for eight fixed fixtures per
domain. This establishes existence for those fixtures, not universal
feasibility, optimality, or performance of a policy restricted to public inputs.
No environment parameter was altered in response to those witness outcomes.

Run the contracts with:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=POLAR_reconciled python -m unittest r9_completion.test_environments -v
```
