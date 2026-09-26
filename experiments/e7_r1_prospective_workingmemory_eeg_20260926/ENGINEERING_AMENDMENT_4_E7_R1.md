# E7-R1 engineering amendment 4 — git-annex identity initialization

Previous selected run:

`36229611948`

It failed before dataset materialization and before any EEG-derived scientific
result.

## Technical defect

DataLad and git-annex were installed successfully.

The failure occurred during:

`datalad install -s https://github.com/OpenNeuroDatasets/ds004117.git e7r1_dataset`

with git-annex unable to create its local initialization commit. The job also
reported that Git user.name and user.email were not configured.

No development subject produced a scientific result JSON.

## Infrastructure-only repair

Before DataLad installation, configure a non-personal CI-local Git identity:

- `user.name = POLAR E7-R1 CI`
- `user.email = e7-r1-ci@invalid.local`

This identity is used only so git-annex can create local administrative commits
inside the ephemeral runner checkout.

No scientific code or protocol item changes:

- dataset snapshot;
- subject split;
- run membership;
- artifact rejection;
- D/C/R labels;
- epoch windows;
- features;
- classifier;
- grouped CV;
- shuffle controls;
- thresholds;
- development/confirmatory gates.

A repaired run may become the adjudicative E7-R1 continuation if selected
prospectively before inspection of any EEG-derived scientific result.
