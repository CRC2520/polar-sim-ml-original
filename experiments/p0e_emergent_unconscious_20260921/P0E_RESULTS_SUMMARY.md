# POLAR P0-E — Emergent Latent Unconscious

Execution date: 2026-09-21  
Development seeds excluded: 1201001–1201004  
Stress seeds excluded: 1201101–1201102  
Confirmatory seeds: 1202001–1202012  
Training: 500 steps per independent agent

## Verdict

**EMERGENT_FUNCTIONAL_SEPARATION_PASS**

All frozen median criteria pass and the joint per-seed guardrail passes **12/12**.

## Confirmatory medians

- hidden-memory AUC: **1.000000**
- current-observation-only AUC: **0.509127**
- pre-trigger workspace AUC: **0.667976**
- hidden minus workspace AUC: **+0.332024**
- workspace query AUC: **1.000000**
- workspace promotion gain: **+0.332024**
- near-history-dependent action accuracy: **1.000000**
- query report accuracy: **1.000000**
- action drop after post-hoc memory-direction lesion: **0.475833**
- action accuracy after memory lesion: **0.524167**
- random-direction lesion drop: **0.000000**
- current-evidence accuracy: **1.000000**
- current-evidence drop under memory lesion: **0.000000**

Seed 1202010 retained the weakest causal-lesion effect (drop ≈0.27) but remained above the frozen per-seed threshold 0.25. All 12 seeds satisfy the complete guardrail.

## Secondary online-update metrics

These were frozen as descriptive and do not determine the verdict:

- post-action accuracy: **0.8325**
- post-query accuracy: **0.8300**
- recon-like update accuracy: **0.720745**
- no-reactivation preservation: **0.790302**
- reactivation-without-contradiction preservation: **1.000000**
- post-update hidden-memory AUC: **0.939933**

Therefore P0-E closes the emergent-separation question, **not** emergent reconsolidation.

## Reproducibility

- executable SHA-256: `296acf3a37002dc885eb76c1996ca08164f3645a01b8613b1f4f0ce097e65175`
- preregistration SHA-256: `9d946ee10d600fd44d1919a52d884b37fa8915e61d1035a462b623f74091ebbf`
- architecture SHA-256: `29ab58bf49dae4d05754a3261a0f9b62a34ccca6adb35df3a1a32b7f14048198`
- confirmatory aggregate SHA-256: `ece8854da71cbb9f226516c40ecf0191de8253e056b8704fbce5f7ffef179506`
- audit SHA-256: `04d7b302d0547ab6eb9c5a034c5716cccdfe80c42e4257d1819f4fe31027fdd4`
- independent audit: **25/25 PASS**
- selected byte-identical exact replays: **1202001, 1202010, 1202012**

A full sequential 12-seed replay was attempted but the session runner timed out after the first replay; it is therefore not represented as 12/12 exact replay.

## Scientific interpretation

P0-E removes the explicit P0-D latent store and memory-specific gate. A single GRU learns historical information that is causally necessary for implicit action, while a generic noisy low-capacity workspace carries materially less of that information before a query and gains access when the task demands a report.

This supports emergence of a **functional distinction between recurrent causal memory and workspace access**. It does not show that an unconstrained network invents its own workspace, nor does it establish biological unconsciousness, repression, trauma, phenomenal consciousness or external replication.
