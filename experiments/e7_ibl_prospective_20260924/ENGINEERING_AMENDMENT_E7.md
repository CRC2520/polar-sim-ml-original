# E7 engineering amendment — OpenAlyx public authentication

Date: 24 September 2026, America/Lima.

Initial metadata-freeze run:

`36090053919`

failed before panel selection because ONE attempted to authenticate public user
`intbrainlab` with no password and OpenAlyx returned HTTP 400.

No scientific/neural outcome was accessed and no development/confirmatory panel
was frozen by that run.

Repair:

- set documented public credentials:
  - username: `intbrainlab`
  - password: `international`

Unchanged:

- IBL release/freeze;
- metadata fields permitted;
- SHA-256 ranking rules;
- minimum 300 trials;
- minimum 20 good units;
- 8 development / 16 confirmatory / 24 reserve;
- all E7 D/C/R endpoints and thresholds;
- prohibition on spike/neural-outcome inspection before panel freeze.

This amendment is infrastructure-only and does not alter the preregistered
scientific design.

## Aggregate-table version repair

Repaired metadata-freeze run `36090200540` authenticated successfully and downloaded metadata, but the upstream helper defaulted to tag `2026_Q2_IBL_et_al_BWM`, whose internal MD5 registry contains `clusters` but not `trials`; the helper therefore raised `KeyError: trials` after downloading `trials.pqt`.

No panel was produced and no spike/neural endpoint was inspected.

Repair:
- pin both aggregate metadata tables to upstream tag `2024_Q2_IBL_et_al_BWM`, the helper version that includes published MD5 hashes for both `clusters` and `trials` and is compatible with the frozen BWM paper query.

Unchanged: all panel hashing, eligibility thresholds, D/C/R endpoints, cross-validation and confirmatory rules.
