# Corrected legacy specification — engine 2.0.1

This is the normative specification for `engine_v2_corrected.py`. It corrects the
legacy simulation; it does not implement, measure, or establish consciousness,
subjective experience, ASI, ethical judgment, or biological trauma. The consensus
and self-EMA forces below are retained for controlled comparison with the original,
not endorsed as the definition of dynamic balance. Contextual models and external
evaluations must be identified separately from this corrected reference engine.

`engine_v1_locked.py` is preserved unchanged. Historical artifacts belong to the
original version and must never be relabeled as corrected results. No original
CSV is evidence for the corrected engine without a new run.

## Units, polarity types, and public state

`N` is the number of numerical units, not the number of polarity types. There are
8 default labels, cyclically assigned to `N` units: Poder/Vulnerabilidad,
Placer/Dolor, Integración/Fragmentación, Control/Rendición, Deseo/Límite,
Libertad/Orden, Preservación/Transformación and Reconocimiento/Autenticidad.
For instance, `N=120` means 120 units carrying eight label types. Custom `nodes`
must provide **exactly N nonempty labels**; implicit expansion is not allowed.
Labels alone do not establish an operational meaning for a unit.

`engine.current_activation` and its compatibility alias `engine.A` return a
**detached clone** of the current N-vector, including before the first step. No
warm-up is required. A malformed/nonfinite state fails explicitly. Multi-agent
code must not substitute a missing state by zeros or infer an eight-unit state
from the number of labels.

## Variables and ranges

| Symbol / code | Meaning | Range / initialization |
|---|---|---|
| `N`, `K` | Numerical units; row top-k proposals for the graph | Integers, N >= 2, 1 <= K < N |
| t / `self.t` | Number of completed transitions | Integer >= 0; initially 0 |
| dt / `cfg.dt` | Integration time increment | Finite positive; default 1 |
| B / `self.B` | Fixed baseline vector, not an external goal | N real finite values, sampled Normal(0, .05²) |
| x_t / `self.tensions` | Internal preactivation tension | [-c,c]^N, c=`tension_clip`=1.2 |
| a_t / `current_activation` | Signed activation tanh(x_t) | [-tanh(c),tanh(c)]^N; default ±0.8336546 |
| a_(t-1) / `A_prev` | Previous activation for temporal alignment | Same bound; initialized to a_0 |
| μ_t / `mu_t` | Running EMA of the system's own activations | Same bound; initialized to a_0 |
| U_t / `homeostasis(a_t)` | Return-to-baseline drive | h(B-a_t), or zero if disabled; finite N-vector |
| Q / `mask` | Fixed unweighted undirected adjacency | Symmetric binary N×N, diagonal zero |
| W0 | Random row-softmax matrix used only to select Q | Positive rows summing to 1; not a learned influence matrix |
| M_t / `M` | Optional learned linear modulation | N×N; starts at 0; bounded by `max_abs_M` after updates |
| s_t / `stimulus` | Explicit external input | Finite tensor shape (N,), values [-1,1] |
| α_t / `alpha` | Continuous dynamical modulation | [`ethics_alpha_min`,1]; 1 when disabled |
| R_t / `risk` | Weighted descriptor deficit for modulation | Nonnegative; **not** ethical consequence risk |

Initial x is clip(.2 Uniform(-1,1)+B, -c,c). Graph construction uses the **union**
of row top-k proposals and their transpose, then removes the diagonal. Historical
function name `mutual_topk_mask` does not mean intersection. Self proposals can
be selected and subsequently removed, so the final degree is not K. Isolated
vertices are possible. The degree denominator is max(1,d_i), as in the original.

A omitted input (`step()` or `stimulus=None`) explicitly means no external
stimulus and logs an N-vector of zeros. A supplied missing, malformed,
nonfinite, wrongly shaped, or out-of-range input raises an error rather than
being silently repaired. Experiment-specific missing data must be rejected by
its caller, which has the context to distinguish it from an intentional no-input
step.

## Descriptors

For d_i=max(1,Σ_j Q_ij), define n_i(a)=Σ_j Q_ij a_j/d_i.

- **HGI = graph activation homogeneity**:
  H(a)=1 − [Σ_i Σ_j Q_ij(a_i−a_j)²/d_i]/(4N).
  Implemented in `HGI_graph`. The denominator 4 uses the theoretical tanh range
  [-1,1], not the narrower clipping range. With c=1.2 its possible lower bound is
  1−tanh²(c), approximately .305; an edgeless graph gives 1 by convention.
- **INC = temporal/self-EMA alignment**:
  I(a,b,μ)=½[1+λ_T cos₀(a,b)+(1−λ_T)cos₀(a,μ)].
  Implemented in `INC_weighted`, with λ_T=`lam_T`=.30. `cos_t` is ordinary cosine
  clipped to [-1,1] for rounding, and **cos₀=0 if either vector has zero norm**.
  Therefore I(0,0,0)=.5. This is an explicit convention, not measured absence or
  presence of narrative, purpose, integration, or consciousness.
- **SAT = saturation occupancy**: mean_i[|a_i|>θ], implemented in `SAT`.
  θ=`saturation_threshold`=.80 is validated as 0<θ<tanh(c). The original .85 was
  unreachable at c=1.2. This is activation occupancy near a bound, not the
  frequency of numerical clipping.
- `lapE` = mean_i(a_i−n_i(a))², implemented in `Laplacian_energy`; `varA` is
  population variance (`unbiased=False`). The original logged sample variance.

Uniform, motionless, or disconnected systems may have HGI=1 and INC≈1. These
scores cannot serve as external success criteria. All-zero states give HGI=1,
INC=.5. No threshold on these descriptors proves a functional capacity.

## One transition and exact timing

Every `step` begins with (x_t,a_(t−1),μ_t,M_t), with a_t=tanh(x_t).
`mu_before` is a clone of μ_t. All right-hand sides below use pre-transition
values unless specified. Initialization uses a_(−1)=a_0 and μ_0=a_0.

1. Compute pre-transition H_t=H(a_t), I_t=I(a_t,a_(t−1),μ_t), S_t=SAT(a_t).
   They are logged as `HGI_before`, `INC_before`, `SAT_before`.
2. `modulation_alpha` computes
   R_t=w_H(1−H_t)+w_C(1−I_t)+w_S S_t,
   α_t=clip(1/(1+κR_t),α_min,1), and O_t=[R_t>τ].
   Weights are finite nonnegative (they need not sum to one); defaults .45,.45,.10.
   κ=1.08, τ=.68, α_min=.60. When `disable_modulation` **or** `no_ethics` is true,
   α=1 and O=false, but the raw R is still measured. No component evaluates
   actions, consequences, rights, welfare, or ethical standards.
3. γ_t=γ_start+(γ_end−γ_start)min(t/max(1,steps−1),1).
   Endpoints are .18,.34 by default. The schedule stays at its final value when
   additional transitions are run; it does not extrapolate indefinitely.
4. U_t=h(B−a_t) with h=`homeostasis_gain`=.05, or zero for `no_homeostasis`.
   D_t=γ_t[n(a_t)−a_t]+β(μ_t−a_t)+U_t+[M_t a_t if learn_M].
   β=`beta_mu`=.06. `homeostasis` is logged per unit, and `U` stores the last
   computed vector; it is not a second independent hidden dynamic variable.
5. Proposed integrator change Δ*_t=dt α_t(D_t+s_t), logged as `proposed_delta`.
   y=clip(x_t+Δ*_t,−c,c), a*=tanh(y). Any ordinary clipping at this step or either
   guard below sets `numeric_clip=true`. Overflow/nonfinite proposals fail.
6. Compute m_H=max(0,h_floor−H(a*)) and m_I=max(0,i_floor−I(a*,a_t,μ_t)).
   If H(a*)<h_floor and g_H>0, apply
   y←clip(y+dt g_H(1+m_H)[n(a_t)−a_t],−c,c), then recompute a*=tanh(y).
   If I(a*,a_t,μ_t)<i_floor and g_I>0, apply
   y←clip(y+dt g_I(1+2m_I)(μ_t−a*),−c,c), then recompute a*.
   The m_I gain is measured before the HGI correction; its activation condition
   is checked after that correction. Defaults h_floor=.95, g_H=.10,
   i_floor=.96, g_I=.18. Gains of zero disable the respective guard. These are
   descriptor-restoring feedback terms, **not** proof of numerical stability or
   objective fulfillment.
7. If O_t, apply y←clip(y,−c_ovr,c_ovr), where `override_clip`=1.0<=c, then
   a_(t+1)=tanh(y). `override` records the trigger; `override_effect` records
   whether this additional clamp actually changes any coordinate. Ordinary
   numerical clipping and override are separate events. The override bound
   has activation maximum tanh(1)≈.7616, below SAT's .8 threshold by design.
8. If learning is enabled, evaluate the loss below on this final a_(t+1), with
   **μ_t**, and update M. M_(t+1) affects the **next** transition, not this one.
9. Measure H_post=H(a_(t+1)), I_post=I(a_(t+1),a_t,**μ_t**) and SAT_post. The
   logged `INC` therefore uses `mu_before`, never the already-updated memory.
10. Commit μ_(t+1)=ρ μ_t+(1−ρ)a_(t+1), with ρ=`memory_retention`=.90;
    x_(t+1)=y; `A_prev`=a_t. Log `mu_after`, `A`, `tension`, and realized
    `delta`=x_(t+1)−x_t. Then increment t.

ρ is explicitly the **retained old-memory fraction per transition**, not the
new-input weight. It lies in [0,1]; ρ=1 freezes the EMA and ρ=0 replaces it.
For 0<ρ<1 its exponential decay time in simulation units is −dt/log(ρ). This
EMA is a self-state statistic, not episodic memory or an external goal model.
Changing dt while holding ρ fixed changes the effective memory time constant.

Continuous modulation, override trigger/effect, HGI correction, INC correction,
and numerical clipping have separate Boolean traces. `modulation_delta_norm`
quantifies the norm of dt(α−1)(D+s); α<1 alone does not imply a nonzero realized
change. No combined unnamed intervention count should replace these fields.

## Learned modulation matrix

`learn_M=False` fixes M=0. If true, a one-step differentiable computation uses
M_t and the loss

L_t = 1.2[1−H(a_(t+1))] + .6[1−I(a_(t+1),a_t,μ_t)]
      + .02 mean_i[a_(t+1),i−n_i(a_(t+1))]²
      + reg_M mean_ij[M_t,ij²].

`step` implements this expression immediately before committing state. The
optimizer is Adam (`lr_M`=.0035, `reg_M`=.0005; other Adam settings are PyTorch
library defaults recorded with its version). Gradients are one-transition only:
state is detached between calls. Branch decisions and the error-scaled guard
gain factors use detached scalars; there is no gradient through branch selection
or those factors. Clamps/tanh retain their usual piecewise derivatives.

Missing/nonfinite loss or gradient raises an error. Gradients are norm-clipped
with `max_grad_norm`=1; `gradient_norm` records the **pre-clipping** norm.
After the optimizer step, M is checked for finiteness and projected to
[−`max_abs_M`,+`max_abs_M`] (default ±1). `M_clipped` records projection.
`learning_loss` and `gradient_norm` are JSON null when learning is disabled,
not fabricated zeros. This loss intentionally rewards old homogeneity and
self-similarity; it is a reproducible comparison control, not the new external
objective and not evidence that learned M offers a functional advantage.

## Configuration map and reproducibility

All numeric parameters above are fields of `EngineConfig`, validated before
construction and copied into the engine. `steps` determines run length and the
consensus schedule. `stim_step` and `stim_scale` are metadata available to
experiment runners; **the engine does not invent an input schedule**. `run()`
runs explicit no-stimulus transitions. Each P7 arm must receive the same saved
stimulus schedule when comparing learning or controller ablations.

Historical `ethics_*`, `ethics_alpha`, and `no_ethics` spellings are compatibility
aliases for **dynamical modulation**, not ethical functionality. Canonical new
code should call `modulation_alpha` and `disable_modulation`. The legacy
`no_ethics` alias disables modulation and its risk-triggered clamp together,
while leaving HGI/INC guards and homeostasis independently controlled.

`export_payload()` emits metadata plus full timeseries. Metadata includes:

- all `asdict(EngineConfig)` values, seed, version, engine source SHA256;
- Python, NumPy, PyTorch versions/build configuration, device/dtype, thread count,
  and whether deterministic algorithms are enabled;
- exact initial B, tension, activation, μ, adjacency Q, M, unit labels and types;
- descriptor meanings and the legacy alias warning.

Every transition retains input, pre/post activation and tension, pre/post memory,
proposed/realized deltas, homeostasis, descriptor values, control flags, and
learning diagnostics. Scalar/vector trace lengths must equal completed steps.
An external schedule is reproducible directly from the logged `stimulus` vectors.
PyTorch Adam state can be regenerated by replaying the full initial run with the
same environment; arbitrary mid-run optimizer checkpoint restoration is not
implemented. Reproducibility is tested on the same CPU/software environment;
bit-identical results across devices or library builds are not promised.

## Differences to original results that must be reported

1. Memory coefficient .90 is exposed and consistently denotes old-state retention.
2. Post-transition INC uses the prior EMA instead of silently rewarding an EMA
   that already incorporates the output being measured.
3. SAT default changes .85→.80, because the former is unreachable under the clamp.
4. Exact zero-norm cosine conventions replace epsilon perturbation of all norms.
5. Configured dt multiplies integrator/guard increments; defaults preserve dt=1.
6. Graph topology preserves legacy construction, with its union/self-selection
   behavior documented; K and eight polarity labels are not confused with N.
7. α modulation, override effect/trigger, guards, numeric clipping, and missing
   learning diagnostics are distinguishable; trajectories retain full states.
8. Population variance replaces sample variance; long runs stop extrapolating γ.
9. Learned M has finite-gradient validation and explicit gradient/value bounds.
10. Repaired experiment schedules/state reads are evaluated through new runs;
    they cannot be interpreted as reproduced published results by assumption.

The corresponding executable acceptance checks are in
`tests/test_engine_corrected.py` (saturation/error handling, memory intervention
and timing, controller isolation, learning/replay, N/type distinction, and
uniform/zero/disconnected high-descriptor counterexamples).
