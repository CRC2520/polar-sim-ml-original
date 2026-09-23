# R44 result — published external-reference compatibility

Frozen resolution:

`R44_PUBLISHED_REFERENCE_COMPATIBILITY_PASS`

R44 evaluates the frozen R37-style recurrent observer/controller on the exact historical POPGym commit used by the external ICLR 2023 benchmark. It does not train or tune the external LSTM; instead it anchors interpretation to the published external result.

## External reference

Morad et al. report in Appendix Table 3, using POPGym commit `e397e5e` and three externally executed trials:
- StatelessCartPoleEasy: LSTM 1.000 ± 0.000; MLP 0.722 ± 0.001.
- StatelessCartPoleMedium: LSTM 1.000 ± 0.000; MLP 0.398 ± 0.006.
- StatelessCartPoleHard: LSTM 1.000 ± 0.000; MLP 0.265 ± 0.002.

The paper endpoint is MMER/max-training reward. R44 uses held-out evaluation episodes. Therefore the result is benchmark-ceiling compatibility, not paired statistical noninferiority.

## Development — Easy

Seeds 2156001–2156008, four episodes per seed/controller.

- CORE_C median score: 1.000000.
- CORE_C − NO_C_CURRENT: +0.785625.
- GENERIC_ISO gap: 0.
- ORACLE_STATE: 1.000000.
- gap to published LSTM ceiling: 0.
- full-episode fraction: 1.0.
- seed guard: 8/8 (required 7/8).

Development authorizes confirmation.

## Confirmatory — Medium

Seeds 2157001–2157020, four episodes per seed/controller.

- CORE_C median score: **1.000000**.
- CORE_C − NO_C_CURRENT: **+0.9009375**.
- GENERIC_ISO gap: **0**.
- ORACLE_STATE: **1.000000**.
- gap to published LSTM ceiling 1.000: **0**.
- full-episode fraction: **1.0**.
- seed guard: **20/20** (required 18/20).

Medium PASS.

## Confirmatory — Hard

- CORE_C median score: **1.000000**.
- CORE_C − NO_C_CURRENT: **+0.9339583**.
- GENERIC_ISO gap: **0**.
- ORACLE_STATE: **1.000000**.
- gap to published LSTM ceiling 1.000: **0**.
- full-episode fraction: **1.0**.
- seed guard: **20/20** (required 18/20).

Hard PASS.

## Interpretation

The result closes a narrower external-comparator boundary than R43. A POLAR-class recurrent observer/controller, frozen before the confirmatory panel, reaches the same benchmark ceiling as the externally published LSTM reference on the exact historical task implementation and retains a large recurrence contrast against the matched current-only controller.

However, R44 is not a direct model-vs-model checkpoint comparison. The published LSTM endpoint and the R44 held-out evaluation endpoint differ, and no external LSTM checkpoint is evaluated on the same episode seeds. Therefore R44 does not establish POLAR superiority or statistical noninferiority against the LSTM itself.

R44 tests only the C/persistent-state contract in this benchmark. It does not test D or R.

## Boundaries

- POLAR superiority: NOT ESTABLISHED.
- E6b independent replication: OPEN.
- E7: OPEN.
- D/R on this benchmark: NOT TESTED.
- universal recurrence benefit: NOT ESTABLISHED.
- phenomenal consciousness: NOT ESTABLISHED.
- POLAR Core v1.1 remains D+C+R with conditional A.

## Provenance

Workflow run: 35838427290.
Execution head: 58c40b9c146494dfb801d8295b41d44036ec9554.
Frozen source blob: bc69823e20443a63c9eefd12577de05177bbcf03.
Frozen protocol blob: 454a75d9d8f162870a3e550bfcddf864f5fd3813.

Source SHA-256: 8e219f018857538660492a34927703c65da63e494379da2c0bd49949f3ee5224.
Protocol SHA-256: 79f38934c6eb714574c006b6bf79be0e05c87a24901242b5c657be92e62b2b92.

Development artifact: 10739339691,
sha256:b721a060628b9e9f6d7fe068608d757e2ed1472cb9f873c3775a04a4c59b8144.

Confirmatory artifact: 10739339774,
sha256:c9c821a16c45c7a04f1448386e9039f6ff39f4f6218072590cd46e3f1cac3ed4.
