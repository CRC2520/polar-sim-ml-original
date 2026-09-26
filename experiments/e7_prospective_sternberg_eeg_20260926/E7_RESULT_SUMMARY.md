# E7 result — prospective Sternberg EEG correspondence

Frozen development resolution:

`E7_DEVELOPMENT_INVALID`

## Canonical execution

- canonical run: `36228475030`
- scientific head SHA:
  `dbb659d3309391436b86c7a7796257be93490ae3`
- dataset: NEMAR `on005095 v1.0.0` / OpenNeuro `ds005095`
- metadata source commit:
  `e7246fbd0c0372161e8dbfbf19cdd387442472c7`
- task: Sternberg EEG
- development subjects: sub-04–sub-19
- confirmation subjects sub-21–sub-52 remained unopened.

## Dataset-quality gate

The preregistered subject-validity rule required:

- at least 120 retained trials after preprocessing/artifact rejection;
- at least 15 retained trials in each set-size class;
- epoch rejection whenever any EEG channel exceeded 200 microvolts
  peak-to-peak after the frozen preprocessing pipeline.

Observed valid development subjects:

- sub-06
- sub-16

Scientifically valid subjects: **2/16**.

Required for a valid development panel: **14/16**.

Therefore the development panel is invalid before D/C/R correspondence
adjudication.

## D/C/R development observations

Because the panel is invalid, these values are diagnostic only and cannot
support a biological-correspondence claim.

Among the two scientifically valid subjects:

### sub-06
- D encoding set-size accuracy: 0.5928
- D specificity vs shuffle: +0.3859
- C retention set-size accuracy: 0.4955
- C specificity: +0.3130
- R probe-relation accuracy: 0.5054
- R specificity: +0.0015

### sub-16
- D encoding set-size accuracy: 0.7936
- D specificity: +0.5837
- C retention set-size accuracy: 0.6612
- C specificity: +0.5167
- R probe-relation accuracy: 0.4346
- R specificity: -0.0313

Aggregate development counts:
- D pass: **2/16**
- C pass: **2/16**
- R pass: **0/16**
- joint D+C+R pass: **0/16**
- median D specificity among valid subjects: **+0.4848**
- median C specificity among valid subjects: **+0.4149**
- median R specificity among valid subjects: **-0.0149**

These numbers remain retained but are not confirmatory biological evidence.

## Why this is INVALID rather than FAIL

The preregistered protocol explicitly requires at least 14/16 scientifically
valid development subjects. Because only 2/16 survive the frozen quality gate,
the intended subject-level population test cannot be interpreted.

The correct label is therefore:

`E7_DEVELOPMENT_INVALID`

and not `E7_DEVELOPMENT_FAIL_NO_CONFIRM`.

## Confirmation protection

No EEG signal from the reserved confirmation cohort was opened by the E7
workflow.

Subjects sub-21 through sub-52 remain scientifically unopened for this E7
campaign.

## Scientific interpretation

E7 does not provide valid prospective biological correspondence evidence under
this frozen preprocessing/quality design.

The invalidity is driven by the interaction between the dataset and the
preregistered artifact-rejection/subject-validity contract, not by a
post-hoc scientific outcome.

The protocol is not relaxed after signal opening.

A follow-up biological test must use a genuinely new dataset/design and a new
preregistration rather than changing E7 thresholds after observing these data.

## Boundaries

E7 does not establish:
- biological identity of POLAR with the brain;
- causal neural necessity;
- D/C/R biological correspondence;
- E6b;
- AGI/ASI;
- phenomenal consciousness.

POLAR Core v1.1 remains unchanged.
