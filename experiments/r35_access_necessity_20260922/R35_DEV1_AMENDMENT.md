# R35 DEV1 amendment — separating C from A before confirmatory freeze

The first R35 development implementation defined contextual access as history enabled only for q=1 and blocked for q=0.

Development showed that recurrent learners also use historical state to improve the q=0 current-only target. Therefore blocking all history at q=0 removed part of C, not only A. Under that implementation, ALWAYS could outperform CONTEXTUAL even in SELECTIVE, making the intervention invalid for testing access selection.

This is a development-design failure, not scientific evidence against A.

Before any R35 confirmatory seed (1907001–1907012), held-out architecture (WINDOW_MLP/LSTM), or confirmatory topology result was opened, the intervention was amended:

- h0 = p_hist(q=0) - p_reset(q=0)
- h1 = p_hist(q=1) - p_reset(q=1)
- contextual: use h_q matching the current context
- always: use pooled history 0.5*(h0+h1), independent of context
- random: use h0 or h1 at random with 0.5 probability
- blocked: use no historical contribution

Thus recurrent/relational history remains computed in all modes; A is the selection of which already-computed historical contribution reaches the readout.

No confirmatory seed, threshold, architecture hold-out, topology split, or adjudication rule had been opened when this amendment was made.
