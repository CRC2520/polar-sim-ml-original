# R36-R2 development result — no confirmatory run authorized

**Frozen development resolution:** `R36_R2_DEVELOPMENT_FAIL_NO_CONFIRM`

Run: **35755362797**  
Artifact: **10707691563**  
Artifact digest: `sha256:bc6d85753e0d7f04539b34ff983bc2574977187a877014c7adb0da8131ad023d`  
Development seeds: **2026001–2026008**  
Confirmatory seeds **2027001–2027012 remain unopened**.

## Preregistered selection rule

A variant was eligible only if every mandatory R36-R1 component remained at least 6/8,
with same-agent lifetime, stability and four learned models at 8/8. Among eligible
variants the rule would maximize joint pass count, then memory+planning, then metacognition,
then intervention simplicity.

**No variant was eligible.**

| Variant | Joint | Memory | Planning | Meta cal | Meta gate | C | R | Eligible |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| BASE_R1 | 4/8 | 6/8 | 6/8 | 7/8 | 5/8 | 7/8 | 7/8 | No |
| FAST_CONTEXT | 3/8 | 8/8 | 6/8 | 8/8 | 5/8 | 7/8 | 7/8 | No |
| LONG_MEMORY | 0/8 | 7/8 | 8/8 | 7/8 | 6/8 | 0/8 | 6/8 | No |
| ROBUST_PLAN | 4/8 | 6/8 | 6/8 | 7/8 | 5/8 | 7/8 | 7/8 | No |
| COMBINED_A | 0/8 | 8/8 | 7/8 | 7/8 | 7/8 | 0/8 | 0/8 | No |
| COMBINED_B | 0/8 | 8/8 | 6/8 | 6/8 | 5/8 | 0/8 | 2/8 | No |

## What the negative development result shows

- Faster context adaptation can improve reentry (FAST_CONTEXT reaches 8/8) but does not
  stabilize functional metacognitive gating or the full conjunction.
- Longer model memory can improve planning, but in this implementation it destroys the
  frozen C criterion and sometimes R/own-history.
- Combining the changes amplifies interference rather than producing robust coexistence.
- The BASE_R1 and ROBUST_PLAN variants remain only 4/8 joint on this broader development panel.
- Seed 2026008 is a broad retained development failure under BASE_R1.

The failure is therefore not localized to one remaining scalar threshold. It is an
**integration-robustness problem**: modifications that help one component can impair another.

## Scientific decision

R36-R2 confirmation is **not authorized**. The unopened confirmatory seeds are not consumed.

POLAR Core v1.1 remains **D+C+R with A conditional**. R36-R1 still establishes all declared
components individually at >=9/12, but robust same-agent co-occurrence remains unestablished.

Given the repeated internal correction sequence R36 → R36-R1 → R36-R2 development, the next
scientific priority should not be another same-program tuning loop. The open requirements are
independent/external validation, E6b, E7, and genuinely external persistent-agent evaluation.

No consciousness, intrinsic selfhood, global minimality or open-ended autonomy claim follows.
