# R36-R2 development preregistration — joint robustness

Status: DEVELOPMENT ONLY — confirmatory seeds unopened
Date: 22 September 2026
Parent: R36-R1_COMPONENTS_PASS_JOINT_FAIL

## Target

R36-R2 does not add a new capability and does not change POLAR Core v1.1.
It targets the residual same-agent conjunction failure after R36-R1:

- full joint conjunction: 7/12 (required 9/12);
- retained planning failures: 2017007, 2017010, 2017012;
- retained memory-reentry failures: 2017004, 2017005;
- all R36-R1 component medians already passed.

## Allowed corrections

Only the following robustness changes may be explored in development:

1. **Context reentry dynamics** — change cue-state EMA to reduce stale-context carryover at regime reentry.
2. **Persistent model horizon** — increase the per-context retained model window.
3. **Robust planning invocation** — require predicted two-step plan advantage to exceed a fixed fraction of locally estimated factual error, while preserving the existing confidence gate and invocation-rate guard.

No D/C/R lesion definition, memory-reentry metric, source/time metric, metacognitive metric,
planning metric, stability metric, threshold, action set, environment, lifetime, schedule,
or confirmatory decision rule may be relaxed.

## Development seeds

2026001–2026008 only.

Confirmatory seeds 2027001–2027012 MUST NOT be executed before a final source and
adjudicator are frozen.

## Development variants

- BASE_R1: cue EMA 0.78, model window 140, planning uncertainty margin 0.
- FAST_CONTEXT: cue EMA 0.70, window 140, margin 0.
- LONG_MEMORY: cue EMA 0.78, window 220, margin 0.
- ROBUST_PLAN: cue EMA 0.78, window 140, margin scale 0.05.
- COMBINED_A: cue EMA 0.70, window 220, margin scale 0.05.
- COMBINED_B: cue EMA 0.72, window 220, margin scale 0.03.

## Fixed selection rule

Use the unchanged R36-R1 per-seed thresholds.

A variant is eligible only if every mandatory component count is >=6/8 and:
- same-agent lifetime = 8/8;
- stability = 8/8;
- four learned models = 8/8.

Among eligible variants:
1. maximize full same-agent joint pass count;
2. tie-break by memory_reentry count + planning count;
3. tie-break by metacog_calibration count + metacog_function count;
4. tie-break by intervention simplicity in this order:
   BASE_R1, FAST_CONTEXT, LONG_MEMORY, ROBUST_PLAN, COMBINED_B, COMBINED_A.

If no variant is eligible, R36-R2 development FAILS and no confirmatory run is authorized.

## Confirmatory criteria

The final confirmatory freeze must reuse the numerical criteria and >=9/12
component/joint requirements from R36-R1. No threshold may be relaxed.

## Boundaries

Same-program synthetic evidence only. E6b, E7, consciousness, global minimality and
open-ended autonomy remain outside this campaign.
