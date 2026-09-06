# Polar dynamics: versioned specification

This index is the single entry point for the implemented specification. An
experiment must name its engine version and save its complete configuration.
Equations from different versions must not be combined into one purported model.

| Version | Role | Normative specification | Implementation |
|---|---|---|---|
| Early draft / v1.0 engine | Immutable implementation reference, including known defects | Original source and artifacts at commit `3c0be12a874e7ad46f4da0e77af2329c19cc61e8` | `engine_v1_locked.py`, `legacy/v2_0/` |
| 2.0.1 corrected reference | Repair and measure the original consensus dynamics | [Corrected reference specification](docs/SPEC_LEGACY_CORRECTED.md) | `engine_v2_corrected.py`, the root experiment entry points |
| 2.1 contextual prototype | Explicit external goals, independent poles, internal memory, workspace and capability estimation | [Contextual specification](docs/SPEC_CONTEXTUAL.md) | `polar/model.py`, `polar/memory.py`, `polar/workspace.py`, `polar/polarities.py` |
| 2.2 coupling experiment | Identify learned off-diagonal effects used in finite-horizon planning | [Coupling specification](docs/STUDY2_CONTROLLER_SPEC.md), [prospective protocol](docs/STUDY2_PROTOCOL.md) | `study2/controllers.py`, `study2/environments.py`, `study2/evaluation.py` |

The reference engine has **N numerical units with eight repeated label types**.
The first contextual experiment has **agents × eight types × two independent channels**.
The coupling experiment uses **two agents × two numerical pairs × two channels = eight
action channels**, with a separate eight-component environmental state. These
two pairs do not claim to instantiate all eight philosophical labels. Its learned
action-effect matrix is specified separately from the original consensus graph.
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
The first study's final results were not used for parameter selection during its
original execution; its data are now explicitly exploratory for diagnosis. The
[coupling protocol](docs/STUDY2_PROTOCOL.md) governs a fresh prospective evaluation,
with a source/protocol GitHub commit recorded before the first final execution.
Its finite prediction horizon is not the response-rate multiplier used by the
first contextual controller.

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
The original unpublished draft source is pinned at
`CRC2520/POLAR_MODEL_CRC@9e04b04594414dbbe58359aae732320bdac61b1b`.
There is one current unpublished article in that repository (`main.tex` including
`body.tex`, compiled as `main.pdf`). Earlier manuscript files are kept through Git
history, while trace data remain supplemental evidence. The current draft
integrates the engineering specification, exploratory first study, prospective
coupling study and separate functional-consciousness protocol. See
`PROVENANCE.json` and the acceptance checklist.


## Inter-polar network extension 0.1

The original layered architecture remains the research object. See [the explicit network tension specification](docs/NETWORK_TENSION_SPEC.md) for the new directed W/K pathways, engineering tests, legacy phase audit and remaining empirical requirements. This extension is not a new confirmatory study and does not change the frozen Study 2 conclusion.


## Integrated P0-P2 gap realization

The original layered architecture now has a bounded integrated implementation in `integrated_polar/`, with separate internal state/intention/action, learned directed effects, priority-sensitive planning, content broadcast, gated memory and institutional constraints. See [the Spanish gap-closure report](docs/P0P2_CIERRE_GAPS.md). The 128 tests passed. The registered P1 decision is `engineering_verified_with_partial_empirical_support`: priority use and return-cue memory pass the practical criteria; coupled planning and the longer horizon do not. The P2 probability forecast is calibrated on the tested domain, but its experimental review policy harms tracking and is not enabled by default. General ethics, intrinsic motivation, philosophical semantics and consciousness are not established. Prior Studies 1–3 remain unchanged.

The new experiment entry point is `python -m gap_resolution.run regenerate`; final seeds are already observed and their replays are not new confirmation. `docs/P0P2_ADDITIONAL_AUDIT.json` records post-run diagnostic/capability verification, separate from the original final records.
