# Functional mechanism audit

Exploratory synthetic diagnostics, not a consciousness assessment.

10 paired seeds; all intervals are descriptive 95% seed bootstrap intervals, unadjusted.

| Metric | Mean | Interval |
|---|---:|---:|
| Memory: intact recall MSE | 0.052050 | [0.051010, 0.053073] |
| Memory: erased recall MSE | 0.288013 | [0.282263, 0.293677] |
| Memory: erasure MSE increase | 0.235963 | [0.231253, 0.240604] |
| Prediction: intact MSE | 0.000208 | [0.000203, 0.000212] |
| Prediction: reset MSE | 0.007719 | [0.007550, 0.007879] |
| Prediction: foreign feedback MSE | 0.026022 | [0.025451, 0.026558] |
| Prediction: foreign feedback MSE increase | 0.025814 | [0.025248, 0.026346] |
| Allocator: content-permutation allocation difference | 0.000000 | [0.000000, 0.000000] |

Memory edits reverse recalled channel orientation; erasure and an unknown cue isolate cue-addressed storage from residual action state. The capability estimator learns an action/effect gain and can improve prediction; deliberately supplying another source's effects corrupts it. The model does not infer effect provenance. The resource allocator responds to aggregate weighted demand, not to broadcast content identity.

The uncertainty variable is a count/residual heuristic, not calibrated probability. No global-content access test, phenomenal-experience test, neural validation, or self/other attribution mechanism is claimed.

Regenerate: `python scripts/run_functional_probes.py --regenerate`. Full inputs, outputs, replay schedules, model states and interventions are in `traces.json.gz`; software and source hashes are in `manifest.json`.
