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
