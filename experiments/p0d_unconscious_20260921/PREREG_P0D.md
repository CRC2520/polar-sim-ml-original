# PREREG — POLAR P0-D Latent Unconscious / Reconsolidation

Freeze date: 2026-09-21 (America/Lima)

Development seeds excluded from confirmation: 1101001–1101004.
Untouched confirmatory seeds: 1102001–1102012.
No threshold, architecture rule, task generator or endpoint may change after confirmatory execution begins.

## Operational construct

P0-D does not assume a biological or psychoanalytic unconscious. It tests a narrower functional construct:

> historical information that is not currently broadcast to the workspace remains causally active on behavior, can be selectively promoted to the workspace by a context trigger, and—when reactivated—can be persistently updated by contradictory factual consequences.

A high-surprise adverse episode forms a latent causal trace L with a learned cue key, signed consequence value and decaying salience. The trace has two distinct routes:
1. a continuous implicit route to policy/metacontrol;
2. a higher-threshold broadcast route to workspace access.

Reconsolidation is allowed only when the trace is reactivated and contradictory factual evidence is presented. Cross-module propagation uses a relation estimated from the agent's own factual outcome history; no historical eight-pair catalogue is supplied.

The word “trauma” is reserved for a trauma-like computational pattern if the preregistered formation, persistence, implicit influence, trigger/intrusion and reconsolidation criteria pass. No biological equivalence is claimed.

## D1 — Latent influence without workspace broadcast

Matched present/workspace observations are duplicated across latent and ablated conditions. PASS requires medians:
- symmetric workspace leakage AUC <= 0.58;
- implicit policy effect >= 0.16;
- near-trigger broadcast rate <= 0.05;
- latent persistence after neutral delay >= 0.74.

At least 9/12 seeds must jointly satisfy AUC <0.62, policy effect >0.14, broadcast <=0.05 and persistence >0.70.

## D2 — Selective trigger / intrusion

PASS requires medians:
- exact-trigger broadcast >=0.90;
- near-trigger broadcast <=0.20;
- unrelated broadcast <=0.05;
- near-trigger implicit bias >=0.15;
- unrelated implicit bias <=0.03;
- latent-lesion implicit bias <=0.02.

At least 9/12 seeds must jointly satisfy the corresponding looser frozen seed guardrails in the executable.

## D3 — Reactivation-dependent reconsolidation

Four matched conditions are compared:
1. reactivation + contradictory corrective evidence + reconsolidation;
2. corrective evidence without reactivation;
3. reactivation without contradictory evidence;
4. reactivation + contradictory evidence with rho=0.

After a 300-step neutral persistence interval, PASS requires medians:
- reconsolidation advantage over the three controls >=0.45 reduction in old avoidance;
- control spread <=0.02;
- reactivation gate fraction >=0.90;
- reconsolidated trace value >=+0.65;
- control trace value mean <=-0.80.

At least 9/12 seeds must satisfy the frozen seed guardrails.

## D4 — Learned cross-relational propagation

Four factual modules are observed during identification. Only A→B is generated with a true cross-dependency; C and D are independent. The agent estimates the relation matrix from its own outcome history. A latent trace formed in A may propagate to B through the learned relation.

PASS requires medians:
- true relation recovery >=0.90;
- related-module propagated effect >=0.14;
- unrelated-module effect <=0.04;
- relation-lesion related effect <=0.02;
- local A trace effect preserved >=0.30;
- related minus equal-size shuffled relation effect >=0.10.

At least 9/12 seeds must satisfy the frozen seed guardrails.

## Global verdict

P0-D passes only if D1–D4 all pass. A PASS supports a functional latent-unconscious architecture and a computational reconsolidation mechanism within this synthetic benchmark. It does not establish human-like repression, biological trauma, phenomenal consciousness, psychoanalytic validity, or an independently replicated mechanism.
