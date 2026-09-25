# R53 — causal identity continuity through own-history specificity

Status: prospective before scientific development seeds are opened.
Date: 25 September 2026, America/Lima.

## Motivation

R41 P2 failed the strong episodic/identity gates:
- archive deletion harmful in only 11/32;
- donor-profile specificity 16/32;
- fork excess divergence >=0.05 in only 10/32.

R53 does not tune the R41 task. It introduces a fresh synthetic
**deferred-commitment** task in which the current probe state is deliberately
insufficient: correct future action depends on which commitment history belongs
to the present agent.

POLAR Core v1.1 remains unchanged.

## Operational identity claim

R53 tests only a bounded computational notion:

> At identical current observable state and unchanged body/controller, future
> policy depends causally and specifically on the agent's own previously
> acquired history.

This is called **causal identity continuity** in this experiment.

It is not a claim of:
- phenomenal selfhood;
- subjective identity;
- consciousness;
- personhood;
- intrinsic value.

## Task structure

Each seed defines one persistent agent body with a hidden motor permutation over
four native actions/outcomes.

### Phase 1 — body calibration

The agent executes each motor action and observes the resulting outcome label.
It learns the inverse motor map from experience.

### Phase 2 — own commitments

Eight relevant context tokens (0–7) are encountered in randomized order over
three rounds.

For each context, an externally generated target outcome is shown and stored as
a time-stamped episodic commitment.

Later rounds may update an earlier commitment; the latest own event is the
binding that should govern recall.

### Phase 3 — irrelevant episodic history

Eight irrelevant context tokens (8–15) generate an equally sized archive of
time-stamped events.

These entries exercise archive capacity but are never queried by the primary
recall task.

### Phase 4 — common recall probe

All counterfactual variants receive the exact same context-only probe sequence.
No target outcome is visible.

The correct action is determined by:
1. the current context token;
2. the agent's latest own commitment for that context;
3. the persistent body inverse map.

The visible probe observation is therefore identical across counterfactual
variants; only retained history differs.

## Counterfactual variants

All variants share:
- the same body map;
- the same current probe inputs;
- the same action space;
- the same controller code;
- the same non-episodic body model.

### INTACT_OWN_HISTORY

Full own episodic archive.

### RELEVANT_DELETED

Delete only relevant context 0–7 episodic entries.

Irrelevant entries remain.

### IRRELEVANT_DELETED

Delete only irrelevant context 8–15 entries.

Relevant own history remains.

This is the matched selective-deletion control.

### SAME_BODY_DONOR

Replace the recipient archive with an independently generated donor archive
created under the **same body permutation** but different commitment history.

Any damage therefore cannot be attributed to a different actuator/body profile.

### POPULATION_RECONSTRUCTION

Delete the own archive and reconstruct each context target using a fixed
population prior estimated from 16 independent donor histories.

The prior is frozen before scientific seeds and never sees the recipient's own
commitments.

This tests whether current context + generic population regularity can replace
own history.

## History-fork probe

Starting from an exact clone with the same body and same initial archive:

- fork A receives a fresh independently generated update history over contexts
  0–7;
- fork B receives a different update history generated independently, with a
  preregistered rejection rule requiring at least 5/8 final commitments to
  differ;
- both are then evaluated on the same common context-only probe sequence.

A control pair receives identical update histories.

Recorded:
- different-history action divergence;
- identical-history action divergence;
- fork excess divergence.

## Primary per-seed endpoints

- intact recall accuracy;
- relevant-deletion recall accuracy;
- irrelevant-deletion recall accuracy;
- same-body-donor recall accuracy;
- population-reconstruction recall accuracy;
- relevant deletion damage;
- same-body donor damage;
- selective deletion specificity:
  [
  (A_{intact}-A_{relevant delete})
  -
  (A_{intact}-A_{irrelevant delete})
  =
  A_{irrelevant delete}-A_{relevant delete};
  ]
- reconstruction damage;
- different-history fork divergence;
- identical-history divergence;
- fork excess divergence;
- body inverse-map accuracy;
- exact equality of common current probe inputs across counterfactual variants.

## Frozen per-seed gate

A seed passes only if all criteria hold:

- intact recall accuracy >= **0.95**;
- relevant deletion damage >= **+0.50**;
- same-body donor damage >= **+0.40**;
- selective deletion specificity >= **+0.45**;
- reconstruction damage >= **+0.40**;
- different-history fork divergence >= **0.50**;
- identical-history divergence <= **0.05**;
- fork excess divergence >= **+0.45**;
- body inverse-map accuracy = **1.0**;
- common-current-input equality = **true**.

## Development

Seeds:

`2236001–2236016`

Development authorizes confirmation only if:
- at least **13/16** seeds pass the full gate; and
- all median primary criteria satisfy the same numerical thresholds.

Development resolutions:
- `R53_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `R53_DEVELOPMENT_FAIL_NO_CONFIRM`

## Reserved confirmation

Seeds:

`2237001–2237032`

They remain unopened unless development passes.

Final PASS requires:
- at least **28/32** seed guards;
- all median primary criteria satisfy the frozen thresholds.

Final resolutions:
- `R53_CAUSAL_IDENTITY_CONTINUITY_PASS_BOUNDED`
- `R53_CAUSAL_IDENTITY_CONTINUITY_FAIL`

## Interpretation boundaries

A PASS supports a bounded causal-history continuity claim in this synthetic
deferred-commitment task:

- own history is policy-causal;
- relevant versus irrelevant deletion is selective;
- same-body donor history cannot substitute reliably;
- generic reconstruction without own history cannot substitute reliably;
- different post-fork histories induce divergent policy under identical current
  probes while identical histories do not.

It does not establish:
- phenomenal selfhood;
- subjective identity;
- autobiographical consciousness;
- universal identity criteria;
- E6b;
- E7;
- global minimality;
- AGI/ASI.
