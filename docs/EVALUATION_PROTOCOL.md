# Frozen external evaluation protocol

The machine-readable authority is `evaluation_protocol.json`. `run_benchmarks.py`
copies its exact bytes and records SHA-256 hashes of the protocol and implementation
**before** running the first seed. This is an internal prospective freeze, not an
independent public preregistration. Five development seeds check execution; there
is no parameter search or tuning. A mechanical smoke run also executes the first
two seeds of each split, including heldout; these two seeds therefore have diagnostic
exposure before the full run. Performance values are not used for model selection
or hyperparameter adjustment. Thirty heldout seeds use unseen patterns, gains,
and priorities in the same two generator families, not independently designed new
tasks. The freeze does not establish broad generalization.

## External tasks and observation boundary

Both tasks have 3 agents, 8 operational channel pairs, 48 independent nonnegative
action components, and 96 steps in six phases of 16 steps. Environment effects are
`clip(gains * action + exogenous_noise, 0, 1)`. Exogenous noise is drawn once per
seed--task and replayed identically for every comparator. All model initial states
are zero, with paired mechanism random seeds. Goals, masks, weights, costs,
permitted-action masks, budget, horizon and cue are the same for every controller.
Unknown gains and hidden target values are never given to a model. Feedback only
contains the effect of its own action. Masked values are zeroed before `act`.

* `switching_memory`: cue A exposure, reverse-pole cue B, distinct distractor,
  masked A recall after distraction, A reacquisition, and B reversal. Independent
  channel pairs explicitly include inactivity, opposite predominance and coactivity.
  The task tests retrieval of experienced cues; it does not require recalling an
  unseen cue or claim autobiographical memory.
* `gain_resource_shift`: calibration, unknown action-gain change, scarce global
  budget with unequal priorities, new gain/role change, an explicit prohibited
  action, and resource release. Workspace contribution is resource allocation,
  not a claimed biological global workspace or arbitrary cross-agent inference.

The constrained oracle minimizes weighted squared effect error with the known
nominal gain, action bounds, costs, permission mask and budget. It is computed by
Lagrange-multiplier bisection. This is an *offline evaluator*, never a controller
input. Realized regret is controller MSE minus this oracle's MSE under the same
noise. Because the oracle is optimal for the nominal noiseless plant, tiny negative
realized regrets are valid and are not clipped. Resource constraints can make the
original target infeasible, so raw RMSE alone is insufficient.

## Comparators and causal scope

`dual_pole` uses inverse-gain planning with recurrent interpolation. Its
`signed_intensity` counterpart stores the exact invertible coordinates
`s=q0-q1, i=q0+q1`. These must produce equivalent trajectories up to floating-point
roundoff: any purported advantage over them would be an implementation problem.

The nonpolar recurrent comparator uses one projected-gradient update:

`q_next = clip(q + eta * (weights / max(weights)) * estimated_gain *
              (effective_target - estimated_gain * q), 0, 1)`.

It shares the 48-element state, cue-memory structure and parameters, gain estimator,
uncertainty, resource allocator, hard constraints, observation access and feedback
timing. Both perform one action update per environment step with the same eta.
They do **not** have identical arithmetic operations/FLOPs; per-step elapsed
microseconds are saved as descriptive compute measurements (includes observation
construction, model trace creation and environment feedback; excludes gzip writing).
Such timing is machine-dependent and is not a controlled performance benchmark.
The difference identifies an update-rule comparison, **not uniquely polar causality**.

Mechanism ablations: `no_memory`, `no_workspace`, `no_self_model`. Targeted
interventions: erase or shuffle memory just before masked recall, shuffle resource
allocation, reset capability estimates after adaptation. Erase and shuffle test
different questions; shuffle may by chance leave a cue unchanged. All interventions
retain numerical bounds and action/resource restrictions. `zero`, `uniform`, and
`disconnected` are explicit noncompetitive negative controls, not the only baselines.

## Outcomes, missingness, inference

Primary endpoint: equal-weight mean of the two task-level mean regrets per seed.
Primary contrast: `dual_pole - recurrent`. Thirty **paired seeds**, not individual
time points, are the independent resampling units. Report mean raw difference,
paired standardized dz, and a 2,000-draw percentile bootstrap 95% interval. A zero
variance difference has undefined dz (`null`/NA), not fabricated zero or infinity.
Other contrasts are exploratory unadjusted intervals; no multiple-comparison
significance claims are made. The primary comparison was specified before execution.

Secondary measurements: RMSE, resource cost, action constraints, target-response
discrimination, cue-recall regret, reacquisition regret, and recovery fraction.
Node homogeneity and temporal action cosine are explicitly descriptive analogues
of the legacy HGI/INC constructs; they do not measure consciousness or goal success.
A temporal cosine with zero denominator is undefined and omitted with null if no
valid pairs exist, never imputed as maximal coherence or zero.
Recovery requires three consecutive steps with regret <= 0.015, observed before
the next phase or episode end. Failure to recover is right censored (`null` in
event semantics), never represented by zero; reported recovery means condition on
observed recoveries and must be read with censor counts and recovery fraction.
Nonapplicable recall scores are NA. A high homogeneity descriptor cannot count as
success: the frozen conjunctive pass rule additionally requires low external regret,
preserved effect discrimination, sufficient recovery, successful recall when
applicable and no hard action violations. This pass rule is a task convention, not
a validated consciousness threshold or benchmark of general intelligence.

## Reproduction and complete trace join

```
python -m unittest discover -s tests -p 'test_evaluation.py'
python scripts/regenerate_reports.py results_corrected/contextual
python scripts/run_benchmarks.py --output results_corrected/contextual_reproduction
```

Use a new empty output directory for a fresh execution. `--smoke` produces a
separately labelled two-seed artifact and is never evidence for the paper. The
runner preserves existing output and refuses overwrite. It requires NumPy and
Matplotlib; there is no external dataset, LLM service, PyTorch or training run.

`environment.jsonl.gz` contains every target, observed mask, gain, disturbance,
weight, action restriction, oracle action, resource budget, cue and phase.
`controllers/<name>.jsonl.gz` contains every actual action/effect, intervention,
resource/numerical-control trace and timing. Join by `(split, task, seed, t)`.
The recurrent state q equals the saved action; complete latent memory/capability
snapshots are saved at each phase boundary and intervention. Interior latent state
is exactly reconstructible from the saved observations and own action effects using
the hashed implementation. Shared environmental arrays are normalized to avoid
duplicating evaluator truth in every controller log.

The exporter checks raw hashes, completeness, order and pairing before regenerating
per-run CSV/JSON, aggregate CSV, paired-effects CSV, Markdown, LaTeX and PDF/PNG.
No manually entered summary supplies the scientific tables. The manifest preserves
versions and full model configuration. Missing values serialize as null or blank/NA,
never NaN or zero. Original v2.0 outputs remain historical and are not overwritten.

## Conclusions permitted by this experiment

The task can demonstrate context-sensitive control and causal uses of specific
engineered memory, gain-estimation and allocation mechanisms. A null or adverse
comparison is retained. Equivalence under coordinate reversal is a sanity check,
not proof of an ontological polarity principle. Neither superior control nor a
functional self-model establishes subjective experience, consciousness, ASI, life,
or a universal physical/metaphysical principle.
