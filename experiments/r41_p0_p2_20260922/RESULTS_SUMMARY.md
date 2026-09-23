# R41 — redesign P0–P2 after R40

Execution workflow: **35818698334**  
Frozen execution head: **59f13bbcb303f141e8f3218cd029c9fecaaca68b**  
Protocol committed before execution: `experiments/r41_p0_p2_20260922/PROTOCOL_R41.md`.

R41 does not rewrite R40 or any historical freeze. It uses fresh tasks/seeds and three separate scientific questions.

## Frozen outcomes

| Scope | Resolution |
|---|---|
| P0 — strong recurrent memory comparator | **P0_DEV_FAIL_NO_CONFIRM** |
| P1 — scale-normalized D+C+R transport | **P1_SCALE_NORMALIZED_CORE_TRANSPORT_PASS** |
| P2 — causal episodic memory / bounded identity | **P2_EPISODIC_OR_IDENTITY_PARTIAL_FAIL** |

POLAR Core v1.1 remains **D+C+R with A conditional**.

## P0 — comparator competence remains open

Three fixed RecurrentPPO/MlpLstmPolicy checkpoints were trained on the fresh, unmodified third-party task POPGym RepeatFirstEasy with **1,048,576 transitions per checkpoint**.

Development results:
- seed 2106001: accuracy **0.500000**, history benefit **0.000000**;
- seed 2106002: accuracy **0.498775**, history benefit **−0.001225**;
- seed 2106003: accuracy **0.475490**, history benefit **+0.036765**.

The preregistered gate required at least 2/3 models with accuracy >=0.85 and intact-minus-reset >=0.25.

The separate supervised LSTM task-solvability instrument also failed its predeclared gate: accuracy **0.740647 < 0.98**.

Thus P0 is a development/instrument failure, not evidence that recurrent PPO is generally weak and not evidence of POLAR superiority. Confirmation is not opened and seeds **2107001–2107064 remain unused**.

Artifact: 10733010433, SHA-256 `ed3ca9b10cdb90094ecb4d234db4183ec98c58b187a8cce59d43c6411130a0a7`.

## P1 — D+C+R transport passes with scale-normalized endpoints

P1 keeps the exact R36-R1 controller and replaces transported raw-MSE thresholds with:
- normalized control capability between matched RANDOM and ORACLE;
- within-trajectory standardized lesion effects for D, C and R.

Per seed, PASS requires normalized score >=0.50, D/C/R dz >=0.20 and stability >=0.99. Support requires >=24/32 in both observation conditions.

### Standard observations
- core conjunction: **30/32**;
- normalized capability: **32/32**, median **0.845184**;
- D: **32/32**, median dz **1.159957**;
- C: **30/32**, median dz **0.274826**;
- R: **32/32**, median dz **0.727107**;
- stability: **32/32**.

### Stress observations
- core conjunction: **31/32**;
- normalized capability: **32/32**, median **0.830734**;
- D: **32/32**, median dz **1.021966**;
- C: **31/32**, median dz **0.351092**;
- R: **32/32**, median dz **0.469132**;
- stability: **32/32**.

Frozen resolution: **P1_SCALE_NORMALIZED_CORE_TRANSPORT_PASS**.

Important adverse boundary: the current-only/no-C closed-loop controller has a lower median tracking cost than the intact controller in both conditions:
- standard intact 0.0468101 vs NO_C 0.0441347;
- stress intact 0.0469354 vs NO_C 0.0422790.

Therefore P1 supports **causal/predictive transport of the D+C+R contracts and overall controller capability**, but it does **not** establish that C is universally beneficial for closed-loop tracking cost, nor implementation privilege, independent replication or consciousness.

Artifact: 10732079053, SHA-256 `6ea0b15d849f5d72a9bc236a4ffa674ba10ba8b73a4f5e4adc11fd8ec851fd7a`.

## P2 — explicitly causal episodic extension fails its strong gates

R41 introduces `EPI_REPLAY` as a versioned experimental extension. Its archive is now read by the policy: recent context-specific residuals alter myopic and two-step predictions. The test therefore avoids R40's archive-as-passive-record problem.

Confirmatory results over 32 fresh seeds:

### Archive deletion
- positive damage in **11/32** seeds;
- mean deletion damage **−0.00062520**;
- bootstrap 95% interval **[−0.00170351, +0.00045029]**;
- **FAIL**.

Deleting the archive is not reliably harmful; under this implementation it is slightly beneficial on average, with an interval spanning zero.

### Profile-specific donor memory
- different-body-minus-same-body donor damage nonnegative in **16/32** seeds;
- mean specificity **+0.00015969**;
- bootstrap 95% interval **[−0.00044784, +0.00089440]**;
- **FAIL**.

### History fork
- different-history excess action divergence >=0.05 in **10/32** seeds;
- mean excess divergence **0.043164**;
- **FAIL**.

The intact EPI_REPLAY agent has an average cost advantage over BASE of approximately **0.00119491**, but that aggregate difference cannot be attributed to continued causal usefulness of the archive because deletion does not worsen performance. No autobiographical/identity claim follows.

Frozen resolution: **P2_EPISODIC_OR_IDENTITY_PARTIAL_FAIL**.

Artifact: 10733180795, SHA-256 `faf0ee25a39bf52e34a1f46668bbfedeb8460101a5eef1fbaba895d7109952fe`.

## Scientific decision after R41

R41 improves measurement quality but does not justify a new core version.

Supported:
- D+C+R transport under a fresh internally generated domain using scale-normalized capability and standardized within-trajectory lesions.

Not supported/closed:
- a competent strong external recurrent comparator (P0);
- universal practical advantage of C in tracking cost;
- causal utility of the tested EPI_REPLAY archive;
- robust history/profile identity effects;
- global causal-state minimality;
- E6b independent replication;
- E7 biological validation;
- phenomenal consciousness.

Further work should not tune on R41's opened panels. P0 needs a separately validated task/baseline recipe; P2 requires a new episodic task where retrieval has prospective decision value by construction but is not trivially encoded in current state. P1 should next be challenged by an independently generated task family rather than another same-generator threshold change.
