# E7-R1 engineering amendment — EEGLAB v7.3 reader dependency

Original canonical signal-opening run:

`36229070442`

Scientific source commit:

`6a4a355894874cb6d2365a2ccd99a1db7aa9cd8b`

## Technical defect

The development jobs successfully downloaded the frozen on004117 run files, but
failed before any EEG-derived scientific endpoint was computed.

MNE `read_raw_eeglab` reported:

- `ModuleNotFoundError: No module named 'pymatreader'`
- fallback `NotImplementedError` for MATLAB v7.3 files.

The dataset's EEGLAB `.set` files therefore require the optional
`pymatreader`/HDF5 reader stack.

## Scientific integrity

At the time of failure:

- no subject result JSON had been produced;
- no D/C/R feature vector or decoder score had been inspected;
- no development gate had been computed;
- no confirmatory subject was opened.

The repair is infrastructure-only.

No change is authorized to:

- subject split;
- run membership;
- preprocessing;
- artifact rejection;
- D/C/R event mappings;
- epoch or feature windows;
- feature representation;
- classifier;
- grouped cross-validation;
- shuffled-label controls;
- subject-validity rules;
- thresholds;
- development or confirmatory gates.

## Repair

Add the pinned optional EEGLAB reader dependencies to the workflow environment:

- `pymatreader==1.0.0`
- `h5py==3.12.1`

The Python scientific source remains unchanged.

A fresh run after this repair may become the adjudicative E7-R1 run because the
first canonical run failed before valid scientific development execution.
