# E7-R1 engineering amendment 3 — DataLad packaging repair

Transport-repair run `36229521349` did not reach dataset materialization.

Ubuntu 24.04 successfully resolved the APT repositories but reported:
`Package 'datalad' has no installation candidate`.

This occurred before `datalad install`, before `datalad get`, before EEG
loading, and before any subject-level scientific result.

Repair:
- install `git-annex` from APT;
- install `datalad==1.6.4` from PyPI;
- keep the same OpenNeuro snapshot `ds004117@1.0.1` and commit
  `065e38865296e7707166eb3b1b561044ac62ab4c`.

No scientific source, subject/run split, preprocessing, labels, features,
classifier, cross-validation, shuffle, threshold or gate changes.

Run `36229521349` is non-adjudicative because it failed before data
materialization. A subsequent complete transport run may be selected
prospectively before subject results are inspected.
