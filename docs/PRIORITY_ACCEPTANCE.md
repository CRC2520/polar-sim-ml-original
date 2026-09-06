# Priority and acceptance checklist

Execution order is P0 → external evaluation protocol → contextual mechanisms →
causal comparisons → manuscript conclusions. Independent implementation work
may proceed in parallel; interpretation follows these gates.

| Order | Priority | Deliverable | Required evidence |
|---|---|---|---|
| 1 | P0 | `SPECIFICATION.md` and versioned equation-to-code maps | Defined states, ranges, memory coefficients, activation limits, metrics and unit counts; historical hashes preserved |
| 2 | P0 | Corrected P6, P7 and Mini-IACL | Opposite signed stimuli; nonzero coupling from actual states; explicit missing-state error; recorded P7 pulse; reachable SAT |
| 3 | P0 | Complete traces and regenerated reports | Config/seed/environment/source hashes; separate modulation/override/guards/clips; missing/censored recovery; legacy deltas |
| 4 | P1 | Frozen external evaluation protocol | Uniform, zero and disconnected controls fail external acceptance regardless of HGI/INC |
| 5 | P1 | Independent dual poles and explicit operational proxies | Inactivity/coactivation distinct; sign-swap equivariance; dual/signed-intensity equivalence |
| 6 | P1 | Contextual goals, budgets, weights and horizon | Response to changing and opposed demands; reacquisition of both orientations; no mandatory consensus |
| 7 | P1 | Numerical limits, external goals and action constraints separated | Distinct trace fields and tests; homogeneity never substitutes for an ethical consequence evaluation |
| 8 | P2 | Episodic internal memory with observable update/recall/forget/edit | Influence after removal of input; matched memory lesion changes subsequent decisions |
| 9 | P2 | Global resource workspace and learned capability model | Predicted effects/uncertainty influence action; targeted lesions and controls produce measured consequences |
| 10 | P2 | Paired multiseed comparisons and revised article | Held-out task variants, effect sizes, confidence intervals, honest results including equality/no advantage |

Automated tests assess implementational acceptance. Benchmark tables assess
functional hypotheses; a hypothesis with a null or adverse result is reported as
such. Neither kind of result establishes consciousness. Test commands and actual
run artifacts are linked in the README and release report.

## Executed evidence — 6 September 2026

The ten implementation/evaluation deliverables above are completed for the scoped
synthetic prototype. The [revision report](REVISION_REPORT.md) maps each row to
its implementation, acceptance evidence and remaining scientific limit.

- [38 automated tests passed](validation_tests.txt), including full-state IACL,
  opposite-pole routing, P7 stimulus equality, causal memory/capability/workspace
  interventions, coordinate equivalence, hidden-target isolation and report replay.
- [40 corrected P0 scenarios](../results_corrected/p0/index.json): eight settings
  by five seeds, with full traces and paired controls where applicable.
- [910 contextual runs](../results_corrected/contextual/manifest.json): two task
  families by 35 seeds by 13 controllers; the inferential contrast uses 30 paired
  heldout seeds with equal-weight task means, not 910 independent observations.
- [Heldout outcomes](../results_corrected/contextual/generated/results.md): the
  dual-pole controller has a small primary mean-regret advantage versus the
  recurrent update rule, but worse recall and 58/60 external passes versus 60/60.
  Signed/intensity coordinates are equivalent; every negative control passes 0/60.

Completion of an experiment does not require a favorable hypothesis result.
A unique polar advantage, broad out-of-domain generalization and consciousness
remain **unestablished**. Operational proxies, symbolic cue memory, heuristic
gain estimation and same-family heldout tasks retain the limits documented in
[SPEC_CONTEXTUAL.md](SPEC_CONTEXTUAL.md) and
[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md).
