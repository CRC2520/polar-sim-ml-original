# P0-D architecture contract — Functional Latent Unconscious

## Position in current POLAR

P0-D adds a latent-memory sublayer between recurrent integration and workspace access:

Perception/Input
→ Recurrent state / world model
→ Latent Causal Trace
  ↘ implicit route → Policy / Meta / Routing
  ↘ gated route → Workspace / Broadcast
→ Action / factual consequence
→ trace formation / reconsolidation
→ learned relational propagation

It is not an eighth consciousness layer. It is a memory/access mechanism inside the current integration/self-memory stack.

## Latent trace

For trace j:

`L_j(t) = (k_j, v_j, s_j)`

- `k_j`: contextual key learned from the episode.
- `v_j`: signed factual consequence / prediction-error content.
- `s_j`: salience/persistence.

A high-surprise event forms a trace from factual prediction error. Neutral dynamics decay salience without necessarily erasing its signed content.

## Dual access routes

Similarity activation:

`a_j(t) = exp(kappa * (cos(q_t,k_j)-1)) * s_j(t)`

Implicit causal route:

`b_t = beta * sum_j a_j(t) v_j(t)`

This route may modify policy/metacognition even when no trace content is broadcast.

Broadcast gate:

`g_j(t) = 1[a_j(t) >= theta_b]`

Only gated traces enter workspace-visible content. Causal influence and broadcast access are therefore experimentally separable.

## Reactivation-dependent reconsolidation

A trace can be updated only when it is reactivated:

`v_j <- (1-rho) v_j + rho y_t`

where `y_t` is contradictory factual consequence evidence. Controls are correction without reactivation, reactivation without contradiction, and `rho=0`.

This is computational reconsolidation only; no biological equivalence is assumed.

## Learned cross-relational propagation

A relation matrix `R` is estimated from the agent's own factual outcome history:

`b_m^cross = gamma R_mj a_j v_j`

No fixed historical polarity catalogue is supplied. Lesioning `R` must remove propagated effects while preserving the local latent-trace effect.

## Falsifiers

The construct fails if:
- latent vs ablated behavior is indistinguishable under matched present/workspace state;
- the latent condition is readily decodable from workspace content while claimed non-broadcast;
- exact and unrelated triggers broadcast equally;
- corrective evidence changes a trace without reactivation;
- reactivation without contradictory evidence produces the same persistent update;
- relation lesions fail to remove cross-module propagation;
- shuffled/unrelated relations reproduce the propagation.

A PASS establishes a functional separation between non-broadcast causal memory and workspace access plus a reactivation-dependent update mechanism. It does not establish repression, human trauma, unconscious phenomenology, biological reconsolidation or consciousness.
