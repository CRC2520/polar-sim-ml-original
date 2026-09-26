# E7-R1 engineering amendment 2 — OpenNeuro transport repair

Previous repaired run: `36229183361`.

The EEGLAB reader dependencies installed correctly. The remaining failure was
`HTTP Error 500` while retrieving subject files from the NEMAR data host.
The error occurred before EEG loading and before any feature or decoder result
was produced.

Equivalent public dataset snapshot:
- repository: `OpenNeuroDatasets/ds004117`
- tag: `1.0.1`
- commit: `065e38865296e7707166eb3b1b561044ac62ab4c`
- DOI: `10.18112/openneuro.ds004117.v1.0.1`

Repair scope:
- change only file transport to the OpenNeuro snapshot;
- retrieve the same frozen BIDS subject/run files locally;
- allow the analysis runner to read from a local dataset root.

No scientific parameter changes: subject split, run membership, artifact
rule, D/C/R labels, windows, features, classifier, grouped CV, shuffle
controls, thresholds and gates remain identical.

Runs `36229070442` and `36229183361` produced no valid subject-level
scientific result. A later transport-repair run may be selected prospectively
as the adjudicative E7-R1 run before any subject result is inspected.
