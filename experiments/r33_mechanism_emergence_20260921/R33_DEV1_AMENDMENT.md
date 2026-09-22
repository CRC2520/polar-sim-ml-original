# R33 development amendment after DEV1

The first formal-development execution (GitHub Actions run **35687548192**, source before this amendment) used direct oracle-action imitation.

Observed median development performance showed that CURRENT_MLP outperformed RNN and GRU even though the recurrent agents had large history-reset and wrong-history-transplant effects. Therefore DEV1 did not create a valid capability contrast for the mechanism-emergence hypothesis.

This is treated as a **development-design failure**, not scientific evidence for or against D+C+R+A.

Before any confirmatory seed was opened, the learning target was changed from direct action imitation to generic hidden-dynamics identification:

- input: current noisy state observation + previous action;
- target: the uncontrolled hidden drift A_t x_t;
- no D/C/R/A, topology, relation or memory labels;
- control action is produced by the same fixed analytical readout from predicted drift and the requested target.

Rationale: current observation alone does not identify the hidden transition map. A learner must use transition history if history is useful, while the CURRENT_MLP remains a legitimate no-history control.

Other predeclared elements remain:
- training topology families;
- development/confirmatory topology split;
- development/confirmatory architecture split;
- development seeds 1706001–1706004;
- confirmatory seeds 1707001–1707012;
- post-training D/C/R/A-like probes;
- falsification logic.

No confirmatory seed, random-DAG evaluation, skew evaluation, WINDOW_MLP or LSTM formal result had been opened when this amendment was made.
