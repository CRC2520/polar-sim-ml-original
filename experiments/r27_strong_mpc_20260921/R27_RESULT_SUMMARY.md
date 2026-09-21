# R27 result — invalid strong-comparator attempt preserved

Workflow run: **35656129705**  
PR head: `93504dd797d026d5a6a86c2a474d1ce90b6ce4a8`  
Artifact: `r27-strong-mpc` (ID 10664287231)  
Source SHA-256: `e0598d0df51de4bf39343bbe5d53a43cfb76f80d12bd8252d45b786df303045b`  
Confirmatory seeds: 1507001--1507012.

## Frozen primary adjudication

The primary non-inferiority claim **FAILS** because the intended MPC comparator is not a valid strong baseline in this implementation.

Confirmatory medians:
- CORE stable fraction: **1.0**
- MPC stable fraction: **0.108671875**
- CORE/GENERIC cost ratio: **0.99993523**
- MPC/FROZEN cost ratio: **623,713,971.95**
- non-inferiority seed guard: **0/12**

Although the arithmetic CORE/MPC cost ratio is extremely small, it is scientifically uninterpretable as a CORE advantage because the MPC implementation itself catastrophically fails stability.

The separate code-level field `exclusive_core_advantage_pass=true` is therefore **not accepted as scientific evidence**: its comparator-validity prerequisite fails. R27 is retained as an adverse engineering result and motivates R28 with a stabilized shooting-MPC comparator.

## Interpretation

- CORE≈GENERIC remains consistent with implementation non-privilege.
- R27 does **not** establish superiority over MPC.
- The strong classical-controller comparison remains OPEN after R27.
- E6b and E7 remain OPEN.
