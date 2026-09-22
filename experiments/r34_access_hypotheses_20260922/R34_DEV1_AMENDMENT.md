# R34 development amendment after DEV1

The first formal-development run used 120 training episodes and 20 epochs for the new H2/H3 query-context tasks.

Observed development behavior:
- A-like contextual interaction was present in recurrent learners.
- However recurrent learners underperformed CURRENT_MLP on q=1, including the UNIFORM task in which both cue values require the same hidden-drift target.

This invalidates DEV1 as a capability test for H2/H3. It is retained as a development-design failure, not scientific evidence for or against contextual access.

Before any H2/H3 confirmatory seed was opened, the only training-budget change was to restore the R33 budget:
- training episodes: 120 -> 160
- epochs: 20 -> 30

No task definition, architecture split, topology split, probe formula, confirmatory seed range, or qualitative falsification logic was changed.

H1 uses the original R33 training routine and is unaffected by this amendment.

No confirmatory seed from 1817001–1817012 or 1827001–1827012, and no formal WINDOW_MLP/LSTM H3 result, had been opened when this amendment was made.
