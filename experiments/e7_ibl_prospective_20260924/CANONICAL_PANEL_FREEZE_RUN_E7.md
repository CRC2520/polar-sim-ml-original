# E7 metadata-freeze canonical run selection

Date: 24 September 2026, America/Lima.

Two pre-panel engineering attempts are non-adjudicative:

- `36090053919`: failed before metadata selection because the documented public
  OpenAlyx password was omitted.
- `36090200540`: authenticated and downloaded aggregate metadata but failed
  before panel creation because the upstream helper's default 2026 aggregate
  MD5 registry lacks a `trials` key.

Neither run produced an E7 panel or inspected spike/neural endpoint values.

Canonical metadata-only panel-freeze run:

- run: `36090357428`
- head SHA: `d6a8127d3161efbb4d45d56dc879d1977dd9308d`
- event: `push`

This selection is recorded before the run's panel artifact is inspected.

The only repairs are public authentication and pinning matched upstream
aggregate metadata tables with published hashes. Scientific panel hashing,
eligibility, D/C/R endpoints and confirmatory thresholds are unchanged.
