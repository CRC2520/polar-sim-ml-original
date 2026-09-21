# R30 confirmatory result — valid Adaptive-LQR baseline; CORE non-inferior, not superior

Confirmatory GitHub Actions run: **35660933114**  
Execution head: `faa59f464b3839975b3902b04dc6320e29b45d94`  
Artifact: `r30-adaptive-lqr` (ID 10666647325)  
Frozen source SHA-256: `75b4d86a56072def419c5a589cc9fa462fa6ce03e8a2e49f06bdee7a9d99ff2e`  
Confirmatory seeds: **1537001--1537012**.

## Frozen adjudication

### Baseline validity: PASS

- Adaptive-LQR stable fraction median: **1.0**
- Adaptive-LQR / FROZEN cost ratio median: **0.50156531**
- validity seed guard: **12/12**

The dense adaptive LQR is therefore a valid matched classical-control baseline under the frozen criteria.

### CORE non-inferiority: PASS

- CORE stable fraction median: **1.0**
- CORE / Adaptive-LQR mean-cost ratio: **0.99988399**
- CORE / Adaptive-LQR post-shift ratio: **0.99984103**
- non-inferiority seed guard: **12/12**

CORE is effectively equivalent to the matched dense Adaptive-LQR baseline at the tested precision and margins.

### Exclusive CORE advantage: FAIL

- frozen advantage required CORE/Adaptive-LQR <= **0.90**
- observed median: **0.99988399**
- advantage seed guard: **0/12**

No privileged CORE/POLAR advantage is established.

## Scientific interpretation

R30 closes the internal strong-classical-baseline question in the following restricted sense:

1. the externally operating R26/R30 controller remains stable over 6,400-step CartPole/Pendulum trajectories with repeated physical regime changes;
2. adaptive control materially improves over the frozen nominal controller;
3. a dense Adaptive-LQR with the same recurrent state estimate, own-action history, online identification budget, action limits and nominal safety anchor matches CORE almost exactly;
4. therefore the evidence supports **implementation non-privilege / organizational equivalence**, not algorithmic POLAR superiority.

R27 and R29 remain invalid-MPC adverse records; R28 remains a cancelled computational attempt. They are not evidence for CORE.

R26 strong relational `H_TRANSFER` remains **FAIL / not established**.  
E6b independent replication remains **OPEN**.  
E7 prospective biological validation remains **OPEN**.
