# R50 result — fresh higher-order integration

Frozen development resolution:

`R50_DEVELOPMENT_FAIL_NO_CONFIRM`

## Canonical execution

- run: `36094560746`
- scientific head SHA: `aa63053788b9ea171c3abc6075c3539347ef6c69`
- development seeds: 2206001–2206006
- reserved confirmatory seeds: 2207001–2207016

R50 uses a fresh five-regime 1,800-step persistent-agent task, a
two-timescale recurrent context state, separated dynamics/reliability memories
and confidence-dependent three-step planning.

## Development component counts

Required for authorization: at least 4/6 for every component and 4/6 full
same-agent conjunction.

| Component | Pass count |
|---|---:|
| D | 6/6 |
| C | 6/6 |
| R | 6/6 |
| persistent reentry memory | 6/6 |
| own-history specificity | 6/6 |
| source attribution | 6/6 |
| time attribution | 6/6 |
| metacognitive calibration | 6/6 |
| functional metacognitive gate | 5/6 |
| planning | **3/6** |
| planning invocation guard | 6/6 |
| stability | 6/6 |
| same 1,800-step lifetime | 6/6 |
| five learned context models | 6/6 |
| full conjunction | **3/6** |

Because planning and the joint conjunction remain below 4/6, confirmation is
not authorized.

## Development medians

- D prediction damage: +0.008849
- C recurrent-context gain: +0.001672
- R relation-lesion damage: +0.001077
- memory reentry gain: +0.000772
- own-history transplant damage: +0.002258
- source balanced accuracy: 0.9842
- time-order accuracy: 1.000
- metacognitive calibration gap: +0.000680
- metacognitive gate gain: +0.000329
- planning gain on invoked states: +0.000336
- planning invocation rate: 0.0423
- stable fraction: 1.000
- learned models: 5

The median planning gain is positive, but only three seeds meet the frozen
per-seed planning threshold.

## Retained adverse seeds

- 2206001: metacognitive gate FAIL, planning strongly adverse
  (-0.006083).
- 2206002: planning adverse (-0.000291).
- 2206004: planning adverse (-0.001201).

Seeds 2206003, 2206005 and 2206006 pass the complete conjunction.

## Interpretation

The fresh partitioned architecture removes most of the heterogeneous
interference observed in R36-R2:

- D/C/R are robust 6/6;
- memory reentry is robust 6/6;
- metacognitive calibration is robust 6/6;
- the functional metacognitive gate improves to 5/6.

However, three-step planning remains seed-sensitive. The diagnostic
`planning_candidate_gain_all` has a negative median (-0.000270), indicating
that the learned model often ranks multi-step candidates poorly even when the
confidence gate sometimes filters them successfully.

This is therefore no longer a broad integration failure. The remaining
same-agent bottleneck is localized primarily to **model-based planning
fidelity under learned dynamics**, with a secondary single-seed metacognitive
gate failure.

## Scientific decision

No R50 confirmatory run is authorized.

Reserved seeds 2207001–2207016 remain unopened.

No R50 tuning loop is authorized on this panel.

POLAR Core v1.1 remains unchanged.

The next independent roadmap experiment is R51 external robustness on fresh
third-party tasks. A future planning-specific campaign, if justified, must use
fresh tasks/seeds rather than tuning R50.
