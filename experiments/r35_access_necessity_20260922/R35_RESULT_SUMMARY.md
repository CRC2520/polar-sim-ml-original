# R35 result summary — representation-independent contextual-access necessity

Frozen resolution:

`R35_A_CORE_NECESSITY_WEAKENED`

R35 intervened on access after recurrent/relational history had already been computed. The learned cue-conditioned historical contributions were:

- h0 = p_hist(q=0) - p_reset(q=0)
- h1 = p_hist(q=1) - p_reset(q=1)

Access modes:
- CONTEXTUAL: matching h_q
- ALWAYS: cue-independent pooled 0.5*(h0+h1)
- RANDOM: h0 or h1 with matched probability
- BLOCKED: no historical contribution

This makes the experiment a functional access intervention rather than a test for a privileged neural gate.

## Confirmatory panel

Seeds: 1907001–1907012  
Topologies: random-DAG + skew  
Architectures: CURRENT_MLP, RNN, GRU, WINDOW_MLP, LSTM  
Strict held-outs: WINDOW_MLP + LSTM

The CURRENT_MLP null control passes.

Three memory architectures satisfy the frozen capability gate:
- RNN
- GRU
- LSTM

WINDOW_MLP is excluded from the necessity decision because its SELECTIVE historical-contribution energy is 0.000319, below the frozen 0.0005 capability threshold.

## RNN

RNN passes the full frozen necessity pattern:
- SELECTIVE contextual vs pooled ALWAYS: +0.00002131
- SELECTIVE contextual vs RANDOM: +0.00002657
- SELECTIVE contextual vs BLOCKED: +0.00336351
- specificity interaction: +0.00002119
- UNIFORM contextual-vs-always difference is effectively zero
- seed guard: 11/12

RNN therefore provides positive internal evidence that selective access can be causally useful.

## GRU

GRU is capable and strongly history-dependent:
- SELECTIVE contextual vs ALWAYS: +0.00003364
- vs RANDOM: +0.00004301
- vs BLOCKED: +0.00200849
- seed guard: 8/12

However its median UNIFORM contextual advantage over pooled access is about +0.00001059 in the same direction. That exceeds the preregistered specificity tolerance: the cue-conditioned advantage is not restricted enough to the SELECTIVE task. GRU therefore does not pass the full necessity pattern.

## LSTM — strict held-out

LSTM is capable and shows the largest SELECTIVE access effects:
- contextual vs ALWAYS: +0.00012153
- vs RANDOM: +0.00017065
- vs BLOCKED: +0.00297766
- seed guard: 8/12

But it also retains a median cue-conditioned advantage in UNIFORM of about +0.00001771, beyond the frozen specificity tolerance. LSTM therefore fails the full necessity pattern despite strong contextual-access capacity.

## WINDOW_MLP — strict held-out

WINDOW_MLP shows SELECTIVE contextual-vs-pooled/random advantages, but its historical-contribution energy is below the frozen capability threshold and its blocked-history damage is only 0.000291. It is therefore not counted as a capable architecture for the necessity adjudication. Seed guard: 3/12.

## Frozen decision

Capable architectures: 3  
Architectures passing full necessity pattern: 1 (RNN)  
Held-out architectures passing full pattern: 0

The preregistered weakening rule was:
- at least 3 capable memory architectures;
- no more than 1 passing the necessity pattern;
- valid CURRENT_MLP null.

All three conditions are met.

Therefore:

`R35_A_CORE_NECESSITY_WEAKENED`

## Interpretation

R35 does not show that contextual access is useless. RNN provides positive causal evidence, and GRU/LSTM show strong contextual-access capacity and material damage when history is blocked.

What fails is the stronger proposition that context-selective access is a universal membership requirement across capable architectures. GRU and LSTM show cue-conditioned benefits that are not sufficiently specific to the SELECTIVE task under the frozen control, and no strict-heldout architecture passes the full necessity pattern.

This result supports a versioned theoretical revision: D+C+R remains the strongest common reduced organizational core, while A should be treated as a conditional/context-sensitive extension unless later independent evidence re-establishes universal necessity.

The original POLAR Core v1.0 freeze must remain preserved as historical preregistered theory; it must not be silently overwritten.

## Development provenance

DEV1 incorrectly blocked all history for q=0, confounding C with A. It is retained as an invalid intervention design. Before confirmatory seeds were opened, R35 was amended to cue-conditioned h0/h1 contributions. No source, threshold, seed, topology, architecture hold-out or adjudication rule changed after the final freeze.

## Boundaries

Same-program synthetic evidence only. R35 does not establish global mathematical minimality, independent replication, biological correspondence or phenomenal consciousness.
