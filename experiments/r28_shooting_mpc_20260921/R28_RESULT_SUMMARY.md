# R28 result — cancelled computational attempt

Workflow run: **35656816476**  
PR head at execution: `a740f3c3c4eb3560acf0dc58262c47358491efd3`.

R28 used a first-action shooting MPC with an LQR tail. The workflow reached the
development/confirmatory execution step but was cancelled by the 30-minute job
limit before it produced a complete result file or frozen adjudication.

Scientific disposition:

- **NO scientific inference** is drawn from R28.
- Confirmatory seeds 1517001--1517012 are treated as opened/burned and will not
  be reused.
- The cancellation is retained as an adverse engineering result: the comparator
  was computationally too expensive for the frozen execution budget.
- R29 uses a new seed family and a faster trust-region MPC comparator with an
  explicit validity prerequisite.
- E6b and E7 remain OPEN.
