# R37 development result — external POPGym validation does not authorize confirmation

Frozen resolution:

`R37_DEVELOPMENT_FAIL_NO_CONFIRM`

Development run: **35768252808**  
Artifact: **10712409469**  
Artifact digest: `sha256:0fc874a018cd4f949fb2a32a3de988dd4b4377b9584c7978fa25afb02537b960`  
Development seeds: **2036001–2036004**  
Confirmatory seeds **2037001–2037012 remain unopened**.

## Frozen adjudication

Medians:
- CORE_C normalized duration: **0.574140625**
- CORE_C − NO_C_CURRENT: **+0.448984375**
- CORE_C − GENERIC_ISO absolute gap: **0.0**
- ORACLE_STATE normalized duration: **1.0**
- minimum per-environment CORE_C: **0.13625**
- seed guard: **0/4** (required 3/4)

The recurrence-effect, coordinate-equivalence and oracle checks pass. The absolute external-control criterion fails: CORE_C is below the frozen 0.70 median threshold and no development seed satisfies the full guardrail.

## Where the failure occurs

Mean development scores:
- PositionOnlyCartPoleMedium: CORE_C **1.0000**, NO_C **0.09984**
- PositionOnlyCartPoleHard: CORE_C **1.0000**, NO_C **0.06849**
- NoisyPositionOnlyCartPoleMedium: CORE_C **0.16214**, NO_C **0.20859**
- NoisyPositionOnlyCartPoleHard: CORE_C **0.14083**, NO_C **0.14427**

Thus the third-party clean POMDPs strongly favor recurrent state, while the naive finite-difference recurrent observer fails under the benchmark's observation noise. In the noisy medium task, current-only even exceeds CORE_C.

## Scientific decision

R37 confirmation is **not authorized**. Seeds 2037001–2037012 remain unopened.

The result supports an external-development observation that recurrence matters on clean partial-observation control, but it does not establish robust external transfer of C under noise. The failure is localized to observer robustness rather than to coordinate privilege: GENERIC_ISO remains exactly equivalent.

A follow-up may evaluate a generic robust observer on the same unmodified POPGym tasks using fresh development/confirmatory seeds and the same external outcome thresholds. This is an external-adapter test, not renewed tuning of the prohibited R36 higher-order conjunction.

POLAR Core v1.1 remains D+C+R with A conditional. E6b and E7 remain OPEN.
