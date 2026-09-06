# Polar dynamics: versioned specification

This index is the single entry point for the implemented specification. An
experiment must name its engine version and save its complete configuration.
Equations from different versions must not be combined into one purported model.

| Version | Role | Normative specification | Implementation |
|---|---|---|---|
| Historical v2.0 manuscript / v1.0 engine | Immutable reference, including known defects | Original source and artifacts at commit `3c0be12a874e7ad46f4da0e77af2329c19cc61e8` | `engine_v1_locked.py`, `legacy/v2_0/` |
| 2.0.1 corrected reference | Repair and measure the original consensus dynamics | [Corrected reference specification](docs/SPEC_LEGACY_CORRECTED.md) | `engine_v2_corrected.py`, the root experiment entry points |
| 2.1 contextual prototype | Explicit external goals, independent poles, internal memory, workspace and capability estimation | [Contextual specification](docs/SPEC_CONTEXTUAL.md) | `polar/model.py`, `polar/memory.py`, `polar/workspace.py`, `polar/polarities.py` |

The reference engine has **N numerical units with eight repeated label types**.
The contextual prototype has **agents × eight types × two independent channels**.
These are different state spaces. A label is not a validated psychological or
neuroscientific measurement. The operational proxies are defined in the
contextual specification and `polar/polarities.py`.

## Meaning of dynamic balance

Dynamic balance means regulation relative to changing demands, costs, constraints,
and the state of other agents. It does not mean maximizing agreement, suppressing
both poles, enforcing a permanent midpoint, or assuming that both poles must be
active at every time. Inactivity, predominance and coactivation are distinguishable.
Independent channels retain the ability to express either pole when required.

The original consensus terms are preserved only in the corrected reference for
historical comparison. Contextual success is assessed by external task outcomes.
HGI and INC remain explicitly named descriptors of homogeneity and temporal/EMA
alignment. They do not certify adaptation, integration, ethics or consciousness.

## Evaluation contract

[The evaluation protocol](docs/EVALUATION_PROTOCOL.md) specifies environments,
controls, interventions, seed splits, success criteria, effects and uncertainty.
Its machine-readable frozen configuration is hashed before experimental runs.
Held-out results are not used for parameter selection in this revision.

The signed-and-intensity representation is an invertible coordinate change of
the dual-channel model. Equality is an expected control result, not evidence of
superiority. A comparator with a different recurrent update tests that update
rule; it cannot establish that a philosophical principle or polarity labels
caused an advantage. Mechanism ablations support only the bounded functional
claims tested in their specified tasks.

## Evidence boundaries

The philosophical claim that polarity is a universal cause of existence, life or
consciousness is a research motivation, not an established result of this code.
This release implements and tests a simulation model. It does not establish
subjective experience, self-awareness, clinical trauma, ethical judgment, AGI or
ASI. A functional capability estimator is called a self-model only in that
operational sense. Missing results are represented as missing or censored rather
than fabricated zeros.

## Preservation and manuscript

The original source, data and results remain available at the original commit.
Legacy scripts are copied separately before correction. Original output
directories and `engine_v1_locked.py` are not overwritten by revised runs.
The original article source is pinned at
`CRC2520/POLAR_MODEL_CRC@9e04b04594414dbbe58359aae732320bdac61b1b`.
The article is updated in that repository with generated tables sourced from the
same recorded runs. See `PROVENANCE.json` and the acceptance checklist.
