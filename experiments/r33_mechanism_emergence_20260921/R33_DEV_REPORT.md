# R33 formal development report and confirmatory freeze rationale

R33 asks whether generic learners trained without POLAR labels develop functional equivalents of D/C/R/A when they become capable on a hidden-dynamics adaptive-control task.

## Development history

### DEV1 — direct policy imitation

Run **35687548192** trained CURRENT_MLP, RNN and GRU to imitate oracle actions directly. Recurrent models exhibited history and relation sensitivity but generalized worse than the current-only MLP. The design therefore failed to produce a valid capability contrast and was retained as a development failure.

### Development amendment

Before any confirmatory seed or strict hold-out architecture was opened, the target was changed to generic hidden-dynamics identification: predict the uncontrolled drift `A_t x_t` from current noisy state plus previous action. A fixed analytical controller converts the predicted drift into an action.

No D/C/R/A, topology, relation, memory or access labels are used in training.

### DEV3/DEV4 — post-shift burn-in

A fixed 8-step burn-in per dynamics block was added to the training/probe loss because immediately after a topology switch the new hidden map is not identifiable from history. Full-trajectory control cost remains recorded; the success criterion uses post-burn adaptive cost.

Formal development run used for the freeze: **35687835348**.

Development medians on the held-out ring topology:

- CURRENT_MLP: control gain 0.000; prediction gain 0.000; C=0; R-transplant=0; A=0.
- RNN: control gain -0.0214; prediction gain +0.2223; D=0.01625; C=0.002613; R-transplant=0.001578; A=0.000455.
- GRU: control gain +0.1579; prediction gain -0.0725; D=0.02020; C=0.003845; R-transplant=0.000869; A=0.000649.

Thus RNN and GRU provide two complementary capable development cases under the frozen success rule: one materially improves hidden-dynamics prediction without >10% control loss, while the other materially improves adaptive control without >10% prediction loss.

The negative current-only model lacks C/R/A-like effects by construction and measurement.

The linear relation-decoding R2 is negative in development and is therefore retained only as a diagnostic. The preregistered causal R-like test remains wrong-topology history transplant damage.

## Frozen mechanism thresholds

- D rank-1 collapse damage >= 0.010
- C history-reset damage >= 0.0010
- R wrong-history transplant damage >= 0.0005
- A contextual history-access specificity >= 0.0002

## Unopened confirmatory material

At freeze time none of the following formal results had been executed:

- seeds 1707001–1707012;
- random-DAG confirmatory evaluation;
- skew-coupled confirmatory evaluation;
- WINDOW_MLP;
- LSTM.

These are opened only after the source, adjudicator and criteria in `R33_CONFIRM_FREEZE.json` are committed.
