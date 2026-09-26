# E7-R1 canonical scientific run — infrastructure-repair replacement

Date: 26 September 2026, America/Lima.

## Superseded first launch

The first signal-opening run `36229070442` at
`6a4a355894874cb6d2365a2ccd99a1db7aa9cd8b` failed before any valid
subject-level EEG scientific result was produced.

The failure was purely technical: MNE could not read the dataset's EEGLAB
MATLAB-v7.3 files because the optional `pymatreader`/HDF5 reader stack was
absent.

No D/C/R feature, decoder score, development gate or confirmatory subject was
scientifically opened before that failure.

The repair is documented in `ENGINEERING_AMENDMENT_E7_R1.md`.

## Replacement canonical scientific run

This decision is recorded after the repaired workflow launched and before any
subject-level scientific result from that repaired run is inspected.

- canonical adjudicative run: `36229183361`
- event: `push`
- head SHA:
  `28e809f96b33ccf4f4e86e99d43c328c84b816d4`
- workflow: `E7-R1 prospective WorkingMemory EEG correspondence`

The only change from the first launch is installation of:

- `pymatreader==1.0.0`
- `h5py==3.12.1`

No scientific source, subject split, run membership, preprocessing rule,
artifact rule, D/C/R mapping, feature definition, classifier,
cross-validation, shuffled control, threshold or gate changed.

Run `36229183361` is therefore the sole adjudicative E7-R1 run.

Reserved confirmatory subjects remain unopened unless the frozen development
gate authorizes them.
