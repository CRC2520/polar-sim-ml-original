# E7-R1 — prospective biological correspondence in Modified Sternberg EEG

Status: prospective before any EEG signal from on004117/ds004117 is opened by
the POLAR E7-R1 campaign.

Freeze date: 26 September 2026, America/Lima.

## Relationship to E7

The first E7 campaign on on005095/ds005095 was invalid at development because
only 2/16 development subjects satisfied its frozen artifact-rejection and
subject-validity contract. Its confirmatory cohort remained unopened.

E7-R1 is a new dataset, new task implementation and new preregistration. It
does not relax or reinterpret the E7 thresholds after observing E7 signal.

## Scientific question

Do human EEG recordings from a modified Sternberg working-memory task show
prospectively defined signatures corresponding to:

- D — differentiated task-relevant item roles;
- C — history-dependent maintenance state;
- R — probe-to-memory-set relation at retrieval?

The current computational hypothesis remains:

[
mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]

E7-R1 is an observational correspondence study. It cannot establish causal
neural necessity or phenomenal consciousness.

## Dataset frozen prospectively

- NEMAR: `on004117`
- NEMAR version: `v1.0.0`
- OpenNeuro: `ds004117`
- OpenNeuro DOI: `10.18112/openneuro.ds004117.v1.0.1`
- NEMAR DOI: `10.82901/nemar.on004117`
- metadata repository: `nemarDatasets/on004117`
- metadata commit:
  `7f607906883e117658534aa2879d59a7a7b5d145`
- task: `WorkingMemory`
- HED version: 8.0.0
- EEG: 69 scalp channels plus periocular channels in the original acquisition
- participants: 23
- task trials organized in multiple 25-trial runs.

Only public task descriptions, event sidecars and BIDS/HED metadata were
inspected before this freeze.

## Subject split

Sort the 23 available subject IDs lexicographically.

Development:
- sub-001 through sub-008 (8 subjects).

Reserved confirmation:
- sub-009, sub-010, sub-011, sub-012,
- sub-014 through sub-024 (15 subjects).

No subject can move between panels after any EEG waveform is loaded.

Run counts are frozen in `METADATA_FREEZE_E7_R1.json`.

## Task labels fixed from HED/BIDS metadata

### D — differentiated encoding roles

Use encoding letter events:

- `event_type=show_letter, task_role=to_remember`
- `event_type=show_letter, task_role=to_ignore`

Prediction:

> EEG during visual encoding distinguishes letters that must be remembered from
> letters that must be ignored.

This tests differentiated task-relevant roles, not unique biological
realization of POLAR D.

### C — persistent maintenance state

Use maintenance-cue events:

- `event_type=show_dash, task_role=work_memory`

Label:
- `memory_cond` = 3, 5 or 7 to-be-remembered letters.

Prediction:

> During the maintenance interval, EEG distinguishes the amount of information
> previously committed to memory while no memory-set letter is currently on
> screen.

This is the preregistered history-dependent state correspondence.

### R — probe relation to maintained set

Use probe events:

- `task_role=probe_target` = IN_SET
- `task_role=probe_not_shown` = OUT_OF_SET

Prediction:

> Early retrieval EEG distinguishes whether the current probe is or is not a
> member of the just-maintained target set.

The relation label comes from task metadata, never from the participant's
button response.

## Preprocessing frozen before signal opening

Process each run independently.

1. Load EEGLAB `.set/.fdt`.
2. Select EEG channels only; periocular/non-EEG channels are excluded.
3. Band-pass 1–35 Hz.
4. Resample to 128 Hz.
5. Average reference.
6. No manual channel or epoch selection.
7. No ICA component selection.
8. No result-dependent channel subset.

### Automated epoch artifact rule

For each candidate epoch after filtering/reference:

Reject the epoch if either:

- more than 10% of EEG channels have peak-to-peak amplitude >500 microvolts; or
- any EEG channel has peak-to-peak amplitude >1500 microvolts.

This rule is fixed specifically for this separate raw high-density dataset
before signal opening. It is not a re-analysis of the invalid E7 dataset.

## Epoch windows

### D encoding role
Anchor: each relevant encoding `show_letter` event.

Epoch:
- -0.20 to +1.05 s.

Feature window:
- +0.10 to +1.00 s.

### C maintenance
Anchor: `show_dash/work_memory`.

Epoch:
- -0.20 to +1.55 s.

Feature window:
- +0.40 to +1.40 s.

### R retrieval
Anchor: probe `show_letter` with role probe_target/probe_not_shown.

Epoch:
- -0.20 to +0.85 s.

Feature window:
- +0.10 to +0.80 s.

Baseline correction is -0.20 to 0 s for all three event types when the epoch
contains that interval. If a run lacks sufficient pre-event samples for an
individual event, only that event is excluded.

## Frozen feature representation

For each EEG channel and each analysis window:

1. mean voltage in five equal-duration time bins across the feature window;
2. log bandpower in:
   - theta 4–7 Hz,
   - alpha 8–12 Hz,
   - beta 13–30 Hz.

All channel/bin/band features are concatenated.

No feature selection is permitted.

## Classifier and cross-validation

Classifier:
- logistic regression;
- L2 penalty;
- fixed C = 0.1;
- max_iter = 3000.

Features are standardized inside the training fold only.

Cross-validation:
- StratifiedGroupKFold;
- group = run number;
- n_splits = min(4, number of available runs);
- shuffle = true;
- random_state = 7107.

All runs contribute to held-out folds; no sample from a held-out run may appear
in its training fold.

Metric:
- balanced accuracy.

## Deterministic shuffled-label controls

### D shuffled control
Shuffle remember/ignore labels within each run and memory-condition stratum.

Seed:
- deterministic subject-specific seed derived from 7111.

### C shuffled control
Shuffle memory-condition labels within each run.

Seed:
- deterministic subject-specific seed derived from 7113.

### R shuffled control
Shuffle IN_SET/OUT_OF_SET labels within each run and memory-condition stratum.

Seed:
- deterministic subject-specific seed derived from 7117.

## Subject validity

A subject is scientifically valid only if all three endpoint datasets remain
adequate after automated epoch rejection.

### D
- at least 300 encoding-letter epochs total;
- at least 100 to_remember;
- at least 100 to_ignore.

### C
- at least 60 maintenance trials total;
- at least 15 trials for each memory condition 3/5/7.

### R
- at least 60 probe trials total;
- at least 25 IN_SET;
- at least 25 OUT_OF_SET.

The full subject is invalid if any endpoint fails its data-quantity contract.

## Subject-level pass criteria

### D
- observed balanced accuracy >= 0.58;
- observed-minus-shuffle specificity >= +0.04.

### C
- observed balanced accuracy >= 0.40;
- observed-minus-shuffle specificity >= +0.04.

### R
- observed balanced accuracy >= 0.58;
- observed-minus-shuffle specificity >= +0.04.

A subject passes the conjunction only if D, C and R all pass.

## Development gate

Development subjects: sub-001–sub-008.

Development authorizes confirmation only if:

- scientifically valid subjects >= 7/8;
- D passes >= 6/8;
- C passes >= 6/8;
- R passes >= 6/8;
- full D+C+R conjunction passes >= 5/8;
- median specificity for D, C and R across valid subjects is positive.

Resolutions:

- `E7_R1_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `E7_R1_DEVELOPMENT_FAIL_NO_CONFIRM`
- `E7_R1_DEVELOPMENT_INVALID`

INVALID applies if scientifically valid subjects <7/8.

## Reserved confirmation

Confirmation subjects: remaining 15 subjects.

Full PASS requires:

- scientifically valid subjects >=13/15;
- D passes >=11/15;
- C passes >=11/15;
- R passes >=11/15;
- full conjunction passes >=10/15;
- median D/C/R specificities across valid subjects are all positive.

Final labels:

- `E7_R1_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_PASS`
- `E7_R1_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_PARTIAL`
- `E7_R1_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_FAIL`
- `E7_R1_PROSPECTIVE_BIOLOGICAL_CORRESPONDENCE_INVALID`

PARTIAL applies when the panel is valid and exactly two of D/C/R reach their
component count thresholds while the full conjunction does not pass.

## Interpretation boundaries

A full PASS would support prospective observational correspondence between the
three computational contracts and three independent EEG task signatures in
this one external working-memory dataset.

It would not establish:

- causal biological necessity;
- identity between POLAR and neural organization;
- global minimality;
- E6b independent computational replication;
- subjective identity;
- AGI/ASI;
- phenomenal consciousness.

Any FAIL, PARTIAL or INVALID result must remain in the research record.
