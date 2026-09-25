# E7 metadata implementation freeze

Date: 25 September 2026, America/Lima.

Parent prospective specification:

- `CRC2520/POLAR_MODEL_CRC`
- `reviews/e7/PREREG_E7_IBL_BRAINWIDEMAP.md`
- Git blob: `84fe7c0fee7976aa1c06fdde3c5cd0f06438e865`

This file freezes implementation details for the metadata-only panel selection.
It does not change any scientific threshold, panel-size rule, D/C/R endpoint or
confirmatory criterion.

## Release membership

Candidate insertions must be returned by the IBL public ONE/Alyx API under
dataset tag:

`Brainwidemap`

using:

`datasets__tags__name,Brainwidemap`

No neural array is requested during this step.

## Trial-count metadata

Trial count is read from Alyx session metadata (`n_trials`) when present.

If absent, the candidate is not eligible in the primary selector rather than
opening trial-level behavioral data during panel selection.

Eligibility remains:

- at least 300 trials.

## Good-unit metadata

To count IBL-good units without downloading spike times, the selector uses the
official Brain-Wide Map aggregate cluster-QC table:

`2026_Q2_IBL_et_al_BWM/clusters.pqt`

The table is accessed through the IBL Brain-Wide Map public tooling and only
metadata fields needed for insertion identity and `label >= 1` are read.

No:
- spikes.times;
- spikes.clusters;
- firing-rate matrix;
- decoder output;
- trial-aligned neural feature

is loaded during panel selection.

Eligibility remains:

- at least 20 IBL-good units per insertion.

## Deterministic panel rule

The parent preregistration is unchanged:

1. within an eligible session select the insertion with lowest SHA-256 of
   `POLAR_E7_INSERTION_V1|<pid>`;
2. within an eligible subject select the session with lowest SHA-256 of
   `POLAR_E7_SESSION_V1|<eid>`;
3. rank remaining triplets by SHA-256 of
   `POLAR_E7_PANEL_V1|<subject>|<eid>|<pid>`;
4. first 8 = development;
5. next 16 = confirmatory;
6. later triplets = reserve.

The selected panel must be committed before any E7 spike-time data are opened.
