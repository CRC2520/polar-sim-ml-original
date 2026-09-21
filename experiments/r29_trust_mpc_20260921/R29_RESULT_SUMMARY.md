# R29 result — comparator invalid, no CORE-vs-MPC inference

Workflow run: **35659883282**  
PR head: `7d34f557af9473ca6a60e0aff0cfd7334849e250`  
Source SHA-256: `9c6508924ce2366d4443a64711902872fe4223f21e7474e7771386832ead321e`  
Confirmatory seeds: 1527001--1527012.

## Frozen adjudication

Comparator validity: **FAIL**
- MPC stable fraction median: **0.09578125** (required >= 0.95)
- MPC/FROZEN cost ratio median: **566,862,866.69** (required <= 1.05)
- comparator-validity seed guard: **0/12**

CORE remained stable (median **1.0**) and CORE/GENERIC remained effectively equivalent
(median cost ratio **1.00002877**), but these facts do not rescue the intended
MPC comparison.

The raw CORE/MPC ratios are extremely small because the MPC comparator fails
catastrophically. They are therefore **not evidence of CORE superiority**.
The frozen adjudicator correctly reports:
- classical-control non-inferiority: **FAIL**
- exclusive CORE advantage: **FAIL**
- comparator valid: **false**

## Scientific disposition

R29 is retained as an adverse engineering result. Together with R27 and R28 it
shows that the internally constructed MPC comparators were not valid strong
baselines under these partial-observation tasks.

The next classical-control comparison therefore uses a qualified dense adaptive
LQR baseline with the same recurrent state estimate, online identification
budget, action limits and nominal safety anchor. This is a valid matched
classical baseline rather than an unstable MPC surrogate.

E6b and E7 remain OPEN. R26 strong relational H_TRANSFER remains FAIL.
