# POLAR P0-D — Latent Unconscious / Reconsolidation

Execution date: 2026-09-21
Development seeds excluded: 1101001–1101004
Confirmatory seeds: 1102001–1102012

## Global verdict

**FUNCTIONAL_LATENT_UNCONSCIOUS_PASS**

All four preregistered experiments pass, with 12/12 seed guardrails in D1–D4.

Exact replay is byte-identical to the confirmatory JSON.
Independent post-run audit: **30/30 PASS**.

## D1 — Latent influence without workspace broadcast

Medians:
- symmetric workspace-leakage AUC: **0.560156**
- implicit policy effect: **0.176380**
- near-trigger broadcast: **0.000000**
- latent persistence: **0.795163**
- seed guardrail: **12/12**

Matched workspace observations remain below the frozen leakage bound while ablating the latent trace materially changes policy.

## D2 — Selective trigger / intrusion

Medians:
- exact-trigger broadcast: **1.000000**
- near-trigger broadcast: **0.000000**
- unrelated broadcast: **0.000000**
- near-trigger implicit bias: **0.181573**
- unrelated implicit bias: **0.002365**
- latent-lesion implicit bias: **0.000000**
- seed guardrail: **12/12**

A near cue biases behavior without broadcast; an exact trigger promotes the trace to workspace access. Unrelated cues and latent lesions remove the effect.

## D3 — Reactivation-dependent reconsolidation

Medians:
- reconsolidation advantage: **0.501172**
- control spread: **0.000000**
- reactivation gate fraction: **1.000000**
- reconsolidated trace value after 300 neutral steps: **+0.890497**
- control trace value mean: **-0.995610**
- seed guardrail: **12/12**

Contradictory factual evidence changes the persistent trace only when the trace is reactivated and reconsolidation is enabled.

## D4 — Learned cross-relational propagation

Medians:
- true relation recovered: **1.000000**
- propagated related-module effect: **0.155694**
- unrelated-module effect: **0.019462**
- relation-lesion related effect: **0.000000**
- local trace effect preserved: **0.345535**
- related minus equal-size shuffled: **0.135961**
- seed guardrail: **12/12**

A latent trace in module A propagates to B through a relation learned from factual outcomes, not to unrelated modules. Relation lesion abolishes the cross effect while preserving the local trace.

## Scientific boundary

P0-D closes the architectural/functional gap left by removal of the original explicit Polar Unconscious layer:
- non-broadcast latent information can remain causally active;
- triggers can promote it selectively to workspace access;
- reactivated traces can be persistently updated by contradictory evidence;
- latent effects can propagate through learned contextual relations.

Open questions remain:
- spontaneous emergence of the dual-route architecture in a less constrained neural learner;
- biological trauma/reconsolidation correspondence;
- human-like repression;
- phenomenal unconsciousness/consciousness;
- independent external replication.

Hashes:
- executable: `08c81d9e5020af05aee73dcd1dcfa35e55fdf6cbf0c5cbcf0886880127bb5c49`
- preregistration: `ccb703926d55ebb9ce407fea17038af87ed7173b1978fdd7dc8600493844606b`
- confirmatory result: `370dfe90375f803fb1e89796005192f181534d8873bcff46050edf6ea5dba0f5`
- exact replay: `370dfe90375f803fb1e89796005192f181534d8873bcff46050edf6ea5dba0f5`
