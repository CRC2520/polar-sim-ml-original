# E7-R1 canonical scientific run — final packaging/transport repair

Date: 26 September 2026, America/Lima.

Earlier launches are non-adjudicative because they failed before valid
scientific subject output:
- `36229070442`: missing EEGLAB reader dependency;
- `36229183361`: NEMAR HTTP 500 transport failure;
- `36229469032`: intermediate runner-only launch with old transport;
- `36229521349`: OpenNeuro transport workflow reached package setup but APT
  had no DataLad package, before materialization.

Final infrastructure repair:
- APT: `git-annex`;
- PyPI: `datalad==1.6.4`;
- OpenNeuro snapshot: `ds004117` tag `1.0.1`, commit
  `065e38865296e7707166eb3b1b561044ac62ab4c`.

No scientific design element changed.

## Sole adjudicative run

- run: `36229611948`
- event: `push`
- head SHA: `f73a2fe9087494a6d097880b59716f12288e87de`
- workflow: `E7-R1 prospective WorkingMemory EEG correspondence`

This selection is recorded before inspection of any subject-level scientific
result from run `36229611948`.

Confirmatory subjects remain protected by the frozen development gate.
