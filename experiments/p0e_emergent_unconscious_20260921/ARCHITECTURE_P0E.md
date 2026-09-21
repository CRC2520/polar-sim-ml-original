# P0-E — Emergent Latent Unconscious in a Shared Recurrent Agent

## Architectural restriction

P0-E deliberately removes the explicit P0-D latent-memory subsystem.

The agent has only:

1. one shared GRU recurrent state `h_t`;
2. an action head reading `h_t` + current observation;
3. a small noisy workspace bottleneck `w_t=f(h_t,o_t)+eps`;
4. a generic report head reading `w_t`;
5. a generic factual-outcome prediction head.

There is **no** `LatentTrace`, no unconscious-memory vector, no memory-specific gate, no salience variable, no trauma threshold, no reconsolidation rule and no target labelled `conscious/unconscious`.

The workspace bottleneck is generic: it is capacity-limited and noisy, but no loss tells it which historical variable must be hidden. Report supervision occurs only when the environment poses a generic outcome-query about the currently presented cue.

## Training task

Each episode contains two independently generated cue identities with independently sampled factual outcome signs. The agent sees the outcomes during acquisition. Later:

- a noisy near-match cue requires an action that depends on the earlier factual history;
- an exact cue can be accompanied by a generic report/query demand;
- a later factual event may contradict the original target association, update a distractor association, or repeat the original evidence;
- a final near-match decision tests online updating.

No phase ID or latent-memory label is supplied.

## Post-hoc operational definition of emergent separation

A functional separation is inferred only after training if all of the following hold on untouched confirmatory agents:

- historical outcome is strongly decodable from the recurrent state before trigger;
- the same information is substantially less decodable from the accessible workspace before trigger;
- current observation alone does not reveal the historical outcome;
- the action nevertheless depends on that historical outcome;
- workspace decodability increases when an exact-cue report/query demand occurs;
- projecting out a post-hoc memory direction from the recurrent state damages the implicit action;
- an equal-dimensional random lesion does not;
- the same memory-direction lesion leaves decisions supported by explicit current evidence intact.

This is a functional/causal separation between internal recurrent memory and workspace access. It does not establish a biological unconscious, phenomenal unconsciousness, repression or consciousness.

## Why this is stronger than P0-D

P0-D explicitly supplied two access routes. P0-E does not. The only persistent substrate is the GRU state. Any distinction between causal memory and workspace availability must arise from learned recurrent organization plus the generic workspace bottleneck/task demands.
