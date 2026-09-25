# R48 engineering amendment — development-gate environment

Canonical scientific run: `36089447230`.

All three development training/evaluation jobs completed successfully and
published their frozen result artifacts.

The downstream `development_gate` job failed before aggregation because it
invoked `r48_neural_relation.py` in a fresh runner without installing numpy
and torch. The error was:

`ModuleNotFoundError: No module named 'numpy'`.

This is an aggregation-environment defect, not a scientific failure.

Repair:
- canonical scientific artifacts were not rerun;
- no seed, checkpoint, threshold, endpoint or budget changed;
- a standard-library adjudicator consumed exactly the three canonical
  `result.json` artifacts in run `36090760519`;
- the deterministic result is 0/3 eligible and
  `R48_DEVELOPMENT_FAIL_NO_CONFIRM`.

Reserved Hard confirmation remains unopened.
