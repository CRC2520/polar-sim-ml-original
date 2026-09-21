# R26 post-confirmatory technical replay note

The primary confirmatory run is GitHub Actions **35647665203** at commit
`23675d73184f2819ac4e3450911ee3af4f114001`.

While automatic R26 workflows were being closed after seed opening, a subsequent branch
push triggered an unintended second technical execution, run **35648069638**, at commit
`22ec14d065abd43d24f54e55407067f5a8ab2e43`. This occurred **after** the primary
confirmatory outcome was known and is therefore not counted as a second confirmatory
experiment or as independent replication.

The frozen experimental source SHA-256 is identical in both runs:
`5217ff2124db15eeb883c9b66816727490f74085aa1e8a4c92d4944155cf30f4`.

The scientific adjudication is unchanged:
- R26-A PASS, seed guardrail 12/12;
- R26-B PASS, seed guardrail 12/12;
- R26-C external operation PASS, seed guardrail 12/12;
- external relational advantage FAIL, seed guardrail 0/12;
- strong H_TRANSFER NOT ESTABLISHED;
- E6b/E7 remain OPEN.

The JSON files are not byte-identical because numerical linear-algebra execution changes
some last floating-point digits. The maximum absolute difference among adjudication
numeric fields is **1.62e-13** (privileged ORACLE ratio); all booleans, thresholds,
seed counts and scientific decisions are identical.

The replay artifact is GitHub Actions artifact **10661481565** with archive digest
`sha256:8b39efddaaaf4a55f844ea573c939c5275e46aab65450572d63294c9d607dee0`.
