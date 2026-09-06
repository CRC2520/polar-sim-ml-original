# Corrected P0 experiments

Version 2.0.1 preserves every original experiment script, reporting module, and
engine byte-for-byte under legacy/v2_0, with their original imports functional.
Historical result directories remain unchanged. Corrected results are written
under results_corrected/p0.

These experiments repair execution and measurement. Their HGI/INC descriptors
do not establish external adaptation, ethics, consciousness, or a polar advantage.
The contextual task evaluation introduced separately tests functional behavior.

## Reproduction

From the repository root, with the pinned reproducibility dependencies installed:

    python -m unittest discover -s tests -p 'test_experiments_corrected.py' -v
    python scripts/run_p0.py --out results_corrected/p0 --seeds 2025 2026 2027 2028 2029
    python scripts/regenerate_p0_reports.py --index results_corrected/p0/index.json

The third command regenerates summaries, figures, and LaTeX tables exclusively
from saved JSON traces, without running an engine. CSV includes all trace
columns; vector/matrix fields are JSON-encoded cells. Scalar metadata,
configuration, versions, initial states, and seeds accompany the JSON. The
manifest maps stable scenario identifiers to files and configurations.

Full JSON/CSV traces are losslessly compressed as deterministic gzip files with
mtime=0; the manifest and summaries stay readable. Use --no-compress to retain
uncompressed trace files instead.

Determinism is scoped to the logged software/hardware configuration; a fixed
seed does not guarantee bitwise agreement between CPU/GPU versions.
The runner sets one PyTorch CPU thread and records that setting.

## Explicit repaired protocols

| Scenario | Corrected protocol | Difference from original |
| --- | --- | --- |
| P2 | 120 units, 8 repeated polarity types, 20 steps, fixed M, no input | Canonical engine equations and complete logging; no invented intervention/recovery zeros. |
| P6 | 160 units, 20 steps; positive Desire pole input +0.5 at zero-index step 8 (ninth update) | Each of 16 pole words has an explicit sign. Both observed words are retained as two evidence flags; their net signed projection cancels. Unknown text raises an error instead of using Python's randomized hash. |
| P7 | 160 units, 24 steps; positive Freedom input +0.6 at zero-index step 8 (ninth update) | Previously the declared pulse was not injected. Identical pulse, seed, and schedule apply to learned-M and ablations. A paired no-pulse control is saved for each condition. |
| Mini-IACL | Three 64-unit engines, 24 synchronous steps, seeds seed+i; group coefficient ramps 0 to 0.28 | Reads the actual full state before each step, without warm-up. Missing/wrong/nonfinite states raise errors. No zero fallback or dimensional tiling. Records each agent's state/input/flags and an uncoupled control. |
| Persistent external perturbation | 160 units, 28 steps; negative pulse 0.8 at index 10 over types 3,4,5; external amplitude decays at rate 0.15; optional attenuation ramps to 0.5 beginning 3 steps later over 6 steps | Filename retained for compatibility. The manipulation is external attenuation, not trauma reconsolidation. No warm-up. Default input persists to the end; optional release_step provides a withdrawal interval. |

All step indices are zero-based, and logs include physical simulation time.
The prior P6 main text and appendix disagreed on timing and targeted pole; the
corrected protocol above is an explicit new convention, not a claimed exact
reproduction of the old narrative. Keyword routing is not a validated NLP model.

The P7 compatibility flag --no-ethics disables legacy dynamical modulation and
override. It does not remove an ethical evaluator: that mechanism never
implemented one. The --no-homeostasis flag removes only the baseline restoring
drive, leaving separately logged guards intact.

## Measurement definitions

- Continuous_modulation_pct: 100 times the fraction of steps where the modulation
  controller is active. If its flag is unavailable, infer from alpha < 1 - 1e-8
  only when a complete finite alpha trace exists.
- Override_pct: percentage of steps with a threshold-triggered override flag.
- Override_effect_pct: percentage of steps where the override changed the
  already proposed state. Triggering and effect are separate observables.
- HGI_guard_pct / INC_guard_pct: respective guard activation percentages.
  These regulate descriptors, not ethical consequences.
- Clipping_pct: percentage of steps with numerical clipping, separately reported
  from control decisions.
- Mini-IACL percentages average per-agent event fractions. Source traces retain
  every individual flag, not only group means.

Dynamical return is measured against the same-config, same-initialization
unstimulated (or uncoupled) control trajectory. At each sample calculate RMS
activation distance across all units (and agents for Mini-IACL). After the last
nonzero external stimulus, return requires three consecutive samples at distance
<= 0.05. Delay is measured from the last nonzero input to the first sample of that
successful window. The earliest possible delay is one step.

This describes neither recovery of task competence nor a clinical measurement.
The manifest records this prespecified rule. Status is:

- recovered: complete sustained return observed; delay is a positive integer.
- censored: no complete qualifying return window before the run ended.
- no_detectable_departure: distance never exceeded tolerance after the event.
- not_applicable: no nonzero external intervention occurred.
- unavailable: required state/input/control traces were not supplied.

Every status except recovered has a null delay. Persisting interventions cannot
be assigned a post-withdrawal return time. Missing flags remain null; unequal
trace lengths, nonfinite numbers, or inconsistent paired schedules raise errors.
