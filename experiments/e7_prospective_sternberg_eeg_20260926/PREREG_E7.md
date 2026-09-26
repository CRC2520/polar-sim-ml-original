# E7 — prospective biological correspondence on held-out Sternberg EEG

Status: prospective. No EEG signal data from the selected dataset may be
inspected before this specification is frozen.

Freeze date: 26 September 2026, America/Lima.

## Scientific question

Do human EEG data from a working-memory task show prospective signatures that
correspond to the three current POLAR Core v1.1 contracts:

[
mathcal P_{core}^{v1.1}=D+C+R
]

under a fixed mapping declared before signal inspection?

E7 is a correspondence test, not a consciousness test and not a claim of
biological identity between POLAR and the brain.

## Dataset fixed prospectively

Primary dataset:

- OpenNeuro: `ds005095`
- NEMAR mirror: `ON005095`
- NEMAR frozen version: `v1.0.0`
- OpenNeuro DOI: `10.18112/openneuro.ds005095.v1.0.2`
- NEMAR DOI: `10.82901/nemar.on005095`
- task: `STERNBERG`
- modality: EEG
- participants: 48
- channels: 63 EEG
- sampling rate: 1000 Hz
- recordings: 48
- license: CC0

Only public task/dataset metadata were inspected before this freeze.

Task structure declared by the dataset:

- 200 trials per participant;
- five set sizes: 3, 6, 9, 12, 15;
- 40 trials per set size;
- positive and negative probes balanced within each block;
- encoding: 0.0–1.5 s after trial trigger;
- retention: 1.5–3.5 s;
- retrieval/probe: 3.5–5.0 s;
- response occurs after probe presentation and is not part of the preregistered
  early-retrieval window.

No previously published EEG outcome from this dataset may be used as a POLAR
endpoint or threshold.

## Subject split

The dataset has 48 recordings.

The split is deterministic and frozen before EEG signal access:

1. query metadata only;
2. sort subject identifiers lexicographically;
3. first 16 subjects = development;
4. remaining 32 subjects = reserved confirmation.

No subject may move between panels after any EEG signal is loaded.

If the dataset query yields other than 48 unique Sternberg EEG subjects, E7 is
`INVALID_DATASET_CONTRACT` until the discrepancy is resolved prospectively.

## Preprocessing

For each subject, independently:

1. load raw EEG only for task STERNBERG;
2. retain EEG channels only;
3. band-pass 1–35 Hz;
4. notch at 50 Hz if supported by the recording metadata/runtime;
5. resample to 128 Hz;
6. average reference;
7. epoch from trial onset to 5.0 s;
8. baseline correct using -0.20 to 0 s when available; if the raw trial trigger
   does not include a valid pre-trigger segment, baseline correction is omitted
   for all subjects and that fact is recorded;
9. reject an epoch if any EEG channel exceeds 200 microvolts peak-to-peak after
   preprocessing.

A subject is scientifically valid only if at least 120 total trials survive and
at least 15 trials survive in each set-size class.

No manual channel/epoch rejection is allowed.

## Frozen feature representation

For each retained epoch and each phase, compute log bandpower per EEG channel in:

- theta: 4–7 Hz;
- alpha: 8–12 Hz;
- beta: 13–30 Hz.

Windows:

- encoding: 0.20–1.40 s;
- retention: 1.80–3.30 s;
- early retrieval: 3.55–4.25 s.

Features are standardized within the training folds only.

The classifier is multinomial/binary logistic regression with L2 penalty and
fixed `C=1.0`. No hyperparameter tuning.

Cross-validation is stratified 5-fold, shuffled with fixed seed 7007.

## Prospective POLAR-to-biology mappings

### D — differentiated biological degrees of freedom

Operational prediction:

> During encoding, multivariate EEG features distinguish the five memory-load
> states (set sizes 3/6/9/12/15) above chance.

Per-subject endpoint:

- five-class balanced accuracy from encoding features.

Chance reference: 0.20.

D subject-pass criterion:

- balanced accuracy >= 0.28.

This is a separability/correspondence prediction, not proof that load states are
the unique biological realization of POLAR D.

### C — history-dependent biological state

Operational prediction:

> During the blank retention interval, EEG retains decodable information about
> the previously encoded set size despite the absence of the letter-set visual
> stimulus.

Per-subject endpoint:

- five-class balanced accuracy for set size from retention features.

C subject-pass criterion:

- balanced accuracy >= 0.25.

The retention window is chosen prospectively because current sensory input is
blank while the relevant set-size information comes from prior encoding.

### R — relation-sensitive retrieval state

Operational prediction:

> During early retrieval, EEG distinguishes whether the probe is a member of
> the encoded memory set, after stratifying cross-validation by set size.

Per-subject endpoint:

- binary balanced accuracy for probe relation:
  `IN_SET` versus `OUT_OF_SET`.

R subject-pass criterion:

- balanced accuracy >= 0.58.

The event parser must derive the relation label from dataset event/task metadata,
not from the participant's response button.

If the released events/logs do not expose the true probe-membership relation
independently of the behavioral response, the R endpoint is
`UNAVAILABLE`, and E7 cannot receive a full D+C+R PASS.

## Negative controls

### D control

Shuffle set-size labels within subject and phase with deterministic permutation
seed 7011. The shuffled decoder is computed once per subject.

Required specificity:
- observed encoding accuracy minus shuffled accuracy >= +0.04.

### C control

Use the same deterministic set-size shuffle in the retention phase.

Required specificity:
- observed retention accuracy minus shuffled accuracy >= +0.03.

### R control

Shuffle IN_SET/OUT_OF_SET labels within each set-size stratum using seed 7013.

Required specificity:
- observed retrieval accuracy minus shuffled accuracy >= +0.04.

## Development gate

Development panel: 16 subjects.

A subject passes the D+C+R biological correspondence conjunction only if:

- D accuracy >=0.28;
- D specificity >=+0.04;
- C accuracy >=0.25;
- C specificity >=+0.03;
- R is available;
- R accuracy >=0.58;
- R specificity >=+0.04.

Development authorizes confirmation only if:

- at least 10/16 valid subjects pass the full conjunction;
- at least 11/16 pass D;
- at least 11/16 pass C;
- at least 10/16 pass R;
- at least 14/16 subjects are scientifically valid.

Development resolutions:

- `E7_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `E7_DEVELOPMENT_FAIL_NO_CONFIRM`
- `E7_DEVELOPMENT_INVALID`

## Reserved confirmation

Confirmation panel: remaining 32 subjects.

Full E7 PASS requires:

- at least 21/32 valid subjects pass the full D+C+R conjunction;
- at least 22/32 pass D;
- at least 22/32 pass C;
- at least 21/32 pass R;
- at least 28/32 subjects are scientifically valid;
- all median observed-minus-shuffle specificities are positive.

Final resolutions:

- `E7_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_PASS`
- `E7_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_PARTIAL`
- `E7_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_FAIL`
- `E7_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_INVALID`

PARTIAL applies if confirmation is opened and exactly two of D/C/R reach their
component count thresholds while the full conjunction fails.

## Falsification interpretation

A FAIL or PARTIAL must be retained.

A full PASS would support a prospective correspondence between the current
computational contracts and specific human EEG signatures in this one Sternberg
dataset.

It would not establish:

- biological identity of POLAR with the brain;
- causal neural necessity, because this is observational EEG;
- global minimality;
- E6b independent computational replication;
- AGI/ASI;
- phenomenal consciousness.

## Data-opening governance

The dataset's raw EEG signal is considered scientifically opened for E7 once
any waveform-derived feature, spectrum, ERP, decoder score or signal quality
metric from a development subject is inspected.

Metadata needed to enumerate subjects/files and parse BIDS events is allowed
before that point.

Confirmatory subjects must not be loaded until the development gate is frozen.
