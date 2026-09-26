# E7-R1 result — prospective WorkingMemory EEG correspondence

Frozen development resolution:

`E7_R1_DEVELOPMENT_FAIL_NO_CONFIRM`

## Canonical adjudicative continuation

- run: `36251413857`
- scientific head SHA:
  `ed10df20c5c4966736d4a8085a6ac52a18e9a3ca`
- dataset: NEMAR `on004117 v1.0.0` / OpenNeuro `ds004117 v1.0.1`
- metadata source commit:
  `7f607906883e117658534aa2879d59a7a7b5d145`
- development subjects: sub-001–sub-008
- confirmation subjects: sub-009, sub-010, sub-011, sub-012, sub-014–sub-024

All prior E7-R1 launches are retained as non-adjudicative infrastructure
failures before scientific subject output.

## Development validity

All 8/8 development subjects satisfy the preregistered data-quantity and
artifact-quality contract.

Therefore E7-R1 development is scientifically valid and must be adjudicated as
PASS/FAIL rather than INVALID.

## Frozen development counts

Required:
- valid subjects >=7/8;
- D pass >=6/8;
- C pass >=6/8;
- R pass >=6/8;
- full D+C+R conjunction >=5/8;
- all median observed-minus-shuffle specificities positive.

Observed:
- valid subjects: **8/8**
- D pass: **8/8**
- C pass: **2/8**
- R pass: **1/8**
- full conjunction: **1/8**

Median observed-minus-shuffle specificity:
- D: **+0.2127**
- C: **+0.0178**
- R: **+0.0221**

D passes robustly at development. C and R do not meet their preregistered
subject-count thresholds, and the joint biological-correspondence conjunction
does not pass.

## Scientific interpretation

### D

The encoding-role prediction is strongly supported in the development panel.
All 8 subjects distinguish `to_remember` versus `to_ignore` encoding events
above the frozen absolute and shuffled-label specificity gates.

This is prospective evidence for a differentiated task-relevance signature in
the tested EEG data.

### C

Only 2/8 subjects pass the frozen maintenance-load decoding criterion.

The median specificity is positive but below the frozen +0.04 subject
specificity requirement for most subjects.

Therefore the preregistered history-dependent maintenance correspondence is not
supported strongly enough to open confirmation.

### R

Only 1/8 subjects passes the frozen probe-relation decoding criterion.

Median R specificity is positive but weak (+0.0221), and most subjects do not
meet the absolute + specificity gate.

Therefore the preregistered probe↔memory-set relation correspondence is not
supported at development.

## Frozen resolution

`E7_R1_DEVELOPMENT_FAIL_NO_CONFIRM`

This is a scientific development failure, not an invalid dataset result.

## Confirmation protection

No confirmatory subject was opened by the adjudicative E7-R1 run.

Reserved subjects remain unopened:
- sub-009
- sub-010
- sub-011
- sub-012
- sub-014 through sub-024.

They must not be used to rescue or retune this frozen E7-R1 design.

## Relationship to E7

E7 on ds005095/on005095 was INVALID because only 2/16 development subjects
satisfied its frozen quality contract.

E7-R1 successfully repairs dataset validity using a genuinely new dataset and a
new prospective protocol, but then fails scientifically because C and R do not
reach the frozen development gates.

Together the two campaigns provide no valid full D+C+R prospective biological
correspondence PASS.

## Boundaries

E7-R1 does not establish:
- causal neural necessity;
- biological identity between POLAR and brain organization;
- full D+C+R biological correspondence;
- E6b;
- AGI/ASI;
- phenomenal consciousness.

POLAR Core v1.1 remains unchanged.
