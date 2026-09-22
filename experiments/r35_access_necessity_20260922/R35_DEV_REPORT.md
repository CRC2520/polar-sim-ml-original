# R35 final development report before confirmatory freeze

## DEV1 retained failure

The first R35 development intervention treated contextual access as history enabled only for q=1 and blocked for q=0. Development showed that recurrent learners also used historical state to improve the q=0 current-only target, so this intervention removed C as well as A. Under that invalid design ALWAYS could outperform CONTEXTUAL.

DEV1 is retained as a design failure:
- run: 35722584906
- head: c22018827e3cff65b3bae215a33a1f8880c3ceed
- artifact: 10692417034
- digest: sha256:31a7e342aef500301c775e5bbe2a0de1c9c605a40a7df7dc935503e3a5e8aa83

Before any confirmatory seed or held-out architecture was opened, the intervention was corrected to cue-conditioned historical contributions:

- h0 = p_hist(q=0) - p_reset(q=0)
- h1 = p_hist(q=1) - p_reset(q=1)
- CONTEXTUAL uses h_q
- ALWAYS uses pooled 0.5*(h0+h1), independent of cue
- RANDOM uses h0 or h1 with matched 0.5 probability
- BLOCKED uses no historical contribution

This keeps C/R computed upstream and isolates A as selection of which learned historical contribution reaches the readout.

## Corrected formal development

Formal corrected run:
- run: 35722883093
- head: d02346a3f97d3ace06348964a03e6cfcdaa80c16
- artifact: 10692042896
- digest: sha256:e38bb5abf0b65cfc35a9d9913db0e4fb76bc5d21c5280cdb2da41e9944bf4666

Development medians on directed-ring:

RNN:
- SELECTIVE contextual vs always: +0.00002222
- SELECTIVE contextual vs random: +0.00002643
- SELECTIVE contextual vs blocked: +0.00260668
- interaction: +0.00002391
- history energy: 0.00246828
- SELECTIVE contextual gain vs CURRENT_MLP: -0.21252

GRU:
- SELECTIVE contextual vs always: +0.00003939
- SELECTIVE contextual vs random: +0.00004156
- SELECTIVE contextual vs blocked: +0.00321865
- interaction: +0.00003204
- history energy: 0.00283693
- SELECTIVE contextual gain vs CURRENT_MLP: -0.00558

Across the eight architecture-seed development cells, the final per-seed guard would pass 7/8 (RNN 4/4; GRU 3/4).

The effect separating contextual from pooled/random access is intentionally much smaller than the effect of blocking history altogether. Confirmatory thresholds therefore test a small selective-access effect while separately requiring material blocked-history damage.

## Frozen confirmatory logic

Support for causal necessity of A requires:
- at least 3 memory architectures passing the median pattern;
- at least 1 strict-heldout architecture among them;
- at least 8/12 seed-level passes per counted architecture;
- CURRENT_MLP null control.

If at least 3 architectures are capable but no more than 1 passes the A pattern, the core-necessity claim is weakened.

Two passing architectures yields an explicitly heterogeneous/inconclusive result.

Confirmatory seeds 1907001–1907012, random-DAG/skew topologies and WINDOW_MLP/LSTM remained unopened when this freeze was created.

Source blob: 8664e40269d477cd0bbad1bca1032632be7c0dcb
Adjudicator blob: a3309973d3544374b38f55219c8e65d28d362080
