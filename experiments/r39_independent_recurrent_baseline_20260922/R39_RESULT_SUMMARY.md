# R39 result — independent RecurrentPPO baseline fails development validity gate

Final R39 workflow run: **35778626788**  
PR execution head: `1068f1b98accb824e252c36f9bc86c17b01a6fb7`  
Merge commit: `1ca28660ec22ea97210c44c4620c9e9980f0c1f7`  
Evidence artifact: **10717985488**  
Artifact digest: `sha256:effe8ba5363e5c9367b3b76397573cfed5759f3eb3294c70acf2f16f358a0126`  
Experiment source SHA-256: `20e9d617d19cf5dd8744751a666f47e397574db31fae53ee5e56b81b1aac9ff8`

External baseline stack:
- POPGym 1.0.7
- stable-baselines3 2.7.1
- sb3-contrib 2.7.1
- RecurrentPPO / MlpLstmPolicy
- torch 2.7.1+cpu
- scipy 1.17.0

Training was fixed at **100,000 transitions per environment**, one predeclared
training seed per environment and one common hyperparameter configuration. No
hyperparameter sweep was performed.

## Frozen development adjudication

**Baseline validity: FAIL**

Medians over development evaluation seeds 2056001--2056004:
- RecurrentPPO overall normalized duration: **0.30668403** (required >=0.70)
- RecurrentPPO minimum-environment duration: **0.17208333** (required >=0.45)
- baseline seed guard: **0/4** (required 3/4)
- frozen R38 CORE score: **0.66572917**
- CORE − RecurrentPPO: **+0.35782118**
- CORE − current-only: **+0.52961806**
- CORE / GENERIC_ISO coordinate gap: **0.0**
- evaluator-only oracle score: **1.0**

Per-environment RecurrentPPO medians:
- PositionOnlyCartPoleMedium: **0.59791667**
- PositionOnlyCartPoleHard: **0.19361111**
- NoisyPositionOnlyCartPoleMedium: **0.26864583**
- NoisyPositionOnlyCartPoleHard: **0.17208333**

Frozen R38 CORE medians on the same development evaluation seeds:
- PositionOnlyCartPoleMedium: **1.0**
- PositionOnlyCartPoleHard: **1.0**
- NoisyPositionOnlyCartPoleMedium: **0.43093750**
- NoisyPositionOnlyCartPoleHard: **0.22000000**

## Scientific decision

**R39_DEVELOPMENT_FAIL_NO_CONFIRM**

The independently developed recurrent baseline did not satisfy its own frozen
validity gate. Therefore:

- confirmatory R39 seeds **2057001--2057012 remain unopened**;
- the large observed CORE-RecurrentPPO gap is **not evidence of POLAR superiority**;
- no RecurrentPPO hyperparameter tuning is authorized on this panel;
- R37/R38 confirmatory seeds remain unopened;
- the external strong-recurrent-baseline question is **not closed** by R39.

R39 is useful as a boundary result: one standard independently developed
RecurrentPPO configuration, despite a large fixed training budget, is not a
scientifically valid strong baseline on this particular POPGym panel. That does
not imply recurrent PPO as a method is weak, and it does not establish a
privileged POLAR algorithm.

## Boundaries

POLAR Core v1.1 remains **D+C+R with A conditional**.  
E6b independent replication remains **OPEN**.  
E7 prospective biological validation remains **OPEN**.  
Phenomenal consciousness and AGI remain unestablished.
