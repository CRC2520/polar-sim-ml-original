# E7-R1 canonical scientific run — OpenNeuro transport replacement

Date: 26 September 2026, America/Lima.

## Non-adjudicative launches

- `36229070442`: missing EEGLAB reader dependency; failed before scientific output.
- `36229183361`: reader dependency fixed, but NEMAR byte transport returned HTTP 500 before scientific output.
- `36229469032`: automatically triggered by the local-dataset-root runner commit while the workflow still used the old NEMAR transport; non-adjudicative by construction.

## Final repair scope

The complete transport repair is frozen at:

`ce9d680cc3f983b18fc2b32dcde919043427aea6`

It materializes the identical public dataset from the official
`OpenNeuroDatasets/ds004117` git-annex snapshot/tag `1.0.1`, commit
`065e38865296e7707166eb3b1b561044ac62ab4c`, and passes those local BIDS
files to the unchanged scientific analysis.

No scientific subject, run, artifact rule, D/C/R label, epoch, feature, model,
cross-validation, shuffle, threshold or gate changed.

## Sole adjudicative run

- run: `36229521349`
- event: `push`
- head SHA: `ce9d680cc3f983b18fc2b32dcde919043427aea6`
- workflow: `E7-R1 prospective WorkingMemory EEG correspondence`

This selection is recorded before inspection of any subject-level scientific
result from run `36229521349`.

Reserved confirmatory subjects remain unopened unless the frozen development
gate authorizes confirmation.
