# Internal implementation review and prospective record

This is an internal review by separate collaborating implementation/review
agents. It is not external peer review, independent laboratory replication,
or a public study registry's endorsement.

## Before final outcome generation

- The root ran `python -m unittest discover -s tests -v`: **63 tests passed**
  in 7.965 seconds, including the unchanged 38 earlier tests and 25 new tests.
  Recorded runtime: Python 3.12.13, NumPy 2.3.5; CPU PyTorch supplied through
  the session's existing dependency installation. Numerical thread counts were
  set to one through OMP, MKL and OpenBLAS environment variables.
- The separate reviewer independently ran all **18 Study 2 tests** and reviewed
  the controller, environmental transitions, trace reader, sample-size rule,
  primary seed-paired analysis, guardrails and prospective execution gate.
- Review identified that the proposed inventory task was initially another
  affine drift system. Before any scored pilot, it was changed to bounded
  inventory with stockouts/overflow. The controller's shared affine planning
  approximation remains a stated model limitation.
- A mechanical check used development seed **51999**, checking coordinate
  equivalence rather than selecting performance parameters. The scored pilot
  used **52001–52006**. All development use is disclosed; these are not final
  evaluation seeds.
- The pilot contained **288 trials**. Its standard deviation of paired,
  equal-cell seed means was **0.0007284646726591249**. The previously specified
  variance-only precision rule selected the minimum **40 final seeds**. This
  is precision planning with an uncertain small-pilot SD, not a power analysis.
- The controller reviewer replayed **12 pilot trials / 1,536 transitions**
  from saved observations and feedback, without hidden plant truth. Actions
  and all recorded mechanistic fields matched exactly, maximum difference
  **0.0**. The full RLS covariance is reconstructed from its updates, while
  the recorded covariance diagonal and subsequent predictions are checked.
  Run `python scripts/audit_study2_pilot_replay.py` to repeat this audit.

## Public source registration

The root created and fetched the public Git registration commit
[`fde8d0ac9b56b035a0a53c40c8cd5b79b2bc6b32`](https://github.com/CRC2520/polar-sim-ml-original/commit/fde8d0ac9b56b035a0a53c40c8cd5b79b2bc6b32)
at **2026-09-06 02:20:56 UTC**, before authorizing any final environment generation. The remote Git blob hashes
of all 13 registered files matched local bytes: 11 frozen sources/protocol/tests,
the freeze, and the pilot decision report. The freeze also records the pilot
manifest's SHA256, allowing the later complete raw-data deposit to be checked.

The freeze was created at **2026-09-06 02:19:55 UTC**. The final run began at
**2026-09-06 02:21:51 UTC**, records that registration SHA, and contains exactly
the fixed seeds **62001–62040**. No final outcomes informed this gate.

## Interpretation boundaries checked before execution

- The primary lesion keeps the estimator architecture and disables learned
  off-diagonal effects only in planning. Closed-loop actions and subsequent
  fitted estimates can diverge; the comparison is a complete policy intervention.
- Targets vary with the plant across regimes. Regime comparisons therefore
  compare condition packages, not an isolated topology manipulation.
- Study 2 uses two labelled groups of control channels within one centralized
  eight-action planner, not two autonomous decision-making agents. There is
  constrained resource coordination, not emergent negotiation.
- Both task families are designed inside this project. The two control indices
  are operational proxies and do not implement all eight philosophical labels
  from the earlier draft.
- A return-phase loss change is a limited reacquisition measure. It does not
  establish general lifelong capability retention. Settling is relative to a
  task threshold, not a proof of universal dynamical stability.
- RLS and predictive constrained control are established methods. A benefit
  from learned cross-effects would not itself establish a novel universal
  polarity principle. Signed-plus-intensity equivalence is an explicit control.
- The separate functional probes test memory and action-effect estimation.
  Resource allocation is not content broadcasting; source attribution and
  metacognitive calibration are not implemented. No consciousness score is used.

Later result interpretation belongs in `RESEARCH_PROGRESS.md`; the frozen
experimental sources and decision criteria are not changed to accommodate it.

## After final completion

The independent internal reviewer regenerated all 1,920 final trials from their
checked archive records. Decision fields and the primary paired estimate with
5,000 bootstrap resamples matched exactly. The 11 frozen source hashes remained
unchanged. Numerical/Markdown/TeX/PNG exports matched byte for byte; PDF metadata
can change with rebuild time.

The reviewer then found a narrow decision-label discrepancy missed before the
run: the protocol says **only** aligned regimes should meet practical benefit
for the conditional-pivot branch, while the frozen code checks aligned benefit
alone. Both aligned and misaligned regimes met that threshold. The raw code label
is retained; the literal protocol decision is
`suspend_exclusive_polar_advantage_claim`. This does not alter effects, intervals,
guardrails or controller trajectories.

`scripts/adjudicate_study2.py` separately reconstructs the literal decision from
hashed inputs, recording its post-run status in `results_study2/adjudication.json`.
The independent reviewer approved its logic and **five additional tests**.
Thus validation comprises **63 tests before final execution and five tests of the
post-run interpretation correction**, not 68 prospective tests. This discrepancy
is disclosed in the single current manuscript and the research progress report.
