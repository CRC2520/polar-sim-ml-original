# B1-S v1.1: stability and comparator competence

Version B1S-MW-D-1.1.0. Development study; not a confirmatory evaluation.

## 1. Baseline and authority

The code baseline is `29feb9f66eb5192616b75581ba6c947ea863c950` and the document baseline is `c99d4e755d29dc823fa74312e39972b0ea2e8121`. The author authorized the next B1-S campaign and its Spanish edition. Previous observations, source files, numerical results and freezes are preserved by exact Git-object checks. The new work lives in `b1s/v1_1/`.

The previous selected support, inconclusive comparisons and unsuccessful competence diagnostics remain observations from v1. This campaign neither replaces them nor guarantees a favorable result.

## 2. Unchanged instrument

The prior native environment and actor classes are imported without editing them. Three Multiwalker processes retain twelve motor coordinates, raw93 centralized information, the original physics, reward and terminal rules. All six messages are computed, with a fixed aggregation denominator of two. No catalogue mapping, new plant, observation normalization or temporal actor memory is introduced.

PPO uses eight synchronous native streams; a rollout size denotes the total number of transitions across those streams. SAC uses one stream. This sampling difference is reported, not treated as identical training data. Episode ledgers use new development-only identities. The optimizer smoke tests use a separate QA seed factory and do not enter scientific results.

## 3. Prespecified training phases

The complete executable registry is in `design.py`; the numerical plan is in `PLAN.json`.

| Phase | Configurations | Initializations | Native steps per fit |
|---|---:|---:|---:|
| Structural screening | 42 sparse masks plus one dense mask | 6 | 131072 |
| Generic PPO calibration | 3 fixed profiles | 3 | 524288 |
| Generic SAC calibration | 3 fixed profiles | 3 | 524288 |
| Fresh competitive refits | 6 comparator roles | 6 | 524288 |

There are 276 screening/calibration fits and 36 competitive fits, totaling 62128128 native training transitions when all phases finish. These are engineering caps, not convergence guarantees. They increase the previous screening duration fourfold, independent discovery initializations twofold and competitive training duration sixteenfold. No extension is decided from a favorable or unfavorable outcome.

Graph PPO keeps its prior optimizer profile. Baseline profiles include the previous selected learning-rate/discount settings, a conventional profile and a conservative-rate profile. Standard SAC candidates use one gradient update per transition, while the previous-ratio profile is retained separately. The algorithm, memory and optimization differences are measured.

Calibration checkpoints at 32768, 131072, 262144 and 524288 describe learning curves. Only each terminal saved model and its own calibration-selection losses select a baseline. A SAC callback checkpoint precedes the last collection-block updates; the selection file evaluates the final saved model after those updates.

## 4. Selection and independence

Structure selection uses 16 separate development-selection episodes per discovery initialization. Minimize mean loss across six initializations, with exact ties resolved by edge count and then lexicographic mask. Empty support remains eligible. Baseline calibration uses 20 separate episodes per initialization and selects within each algorithm, without reference to graph performance.

Seal support and baseline profiles before six new competitive initializations. Evaluate them on 32 paired episodes per initialization. Time steps and episodes inside one trained policy are not independent training replications. Previous v1 observations are not pooled with v1.1.

## 5. Stability and external performance

Report every per-initialization support winner, edge frequencies with marginal Wilson intervals, the Jaccard matrix and mean Jaccard. The descriptive engineering stability diagnostic requires modal frequency at least 2/3 and mean Jaccard at least 0.6. Empty-empty Jaccard is one. A count-conditioned enumeration supplies a null reference, not a catalogue p-value or an assumption of physical exchangeability. Six initializations support only limited precision.

Preserve negative native return as the primary loss and retain displacement, falls, episode duration and requested effort separately. The competence conjunction remains loss improvement over a no-action witness greater than 130/30 and additional mean displacement at least one native unit. Apply it to every comparator. A failed diagnostic limits the conclusion; it does not justify lowering its threshold.

Paired losses are averaged first within each of six independent competitive training bundles. Nominal t intervals use five degrees of freedom and are exploratory, not familywise confirmatory. The prior practical and descriptive equivalence references remain 130/30 and half that amount. Identical selected/comparator configurations are identity controls, not additional evidence of equivalence.

## 6. Causal and functional diagnostics

Use eight new causal episodes per competitive initialization and an eight-cycle recorded intact prefix. Retain episodes ending before the prefix without replacing them. Reproduction checks returned observations, rewards and an audit fingerprint of physical bodies, joints, shaping state and random generators. The observed state vector is not a complete simulator checkpoint.

Compare intact, matched sham, each active edge, an off-support control and a two-edge factorial when available. Only route utilization changes; features, messages and weights remain unchanged. Apply the intervention once, then continue with the intact policy. Immediate action change and subsequent total policy loss are separate quantities.

For functional agreement, use the same panel of native observations from four no-action and four selected-policy probe episodes per competitive initialization, up to 64 cycles each. Compare mean absolute, RMS and maximum action differences. Mean absolute difference at most 0.01 and maximum at most 0.05 define a descriptive panel-agreement flag, not a global policy-equivalence theorem.

## 7. Resources, retention and failures

Audit information, feasibility, learning, memory, planning and decision computation separately. A common environmental budget does not make algorithms or search spaces equally expressive. Preserve differences in PPO stream count, SAC replay and target critics, update ratios and parameter counts.

Preserve every attempted fit, source/runtime hashes, episode identities and outcomes, curves, checkpoints and selection records. Store full competitive and causal traces compressed. Raw training transitions and SAC replay buffers are not all archived; cross-hardware bitwise training reproduction is not guaranteed.

Missing, invalid or failed fits prevent aggregate completion of the affected stage. No automatic scientific retry or seed replacement is allowed. Technical incidents are retained. Failure of competence or stability is a scientific finding and may coexist with successful execution of the development pipeline.

## 8. Closure and translation

Operational status is either `B1S_V1_1_DEVELOPMENT_COMPLETE_WITH_LIMITATIONS` or `B1S_V1_1_BLOCKED`. H_CAT remains NOT_EVALUABLE and H_TRANSFER remains NOT_EVALUATED. Final seeds are not generated, B1-E is not executed and no automatic confirmation is authorized.

English and Spanish reports are rendered from the same retained JSON and preserve numerical values, identifiers and decisions. The document repository imports them by immutable commit and SHA-256. Earlier LaTeX and PDF files remain unchanged.

## 9. References and status of numerical choices

Patterson et al., Empirical Design in Reinforcement Learning, JMLR 25(318), 2024, and the official Stable-Baselines3 RL Tips and Tricks guide motivate separate evaluation, multiple runs and explicit tuning budgets. They do not supply or validate this campaign's numeric caps and thresholds. Those are new prospective engineering decisions. PPO and SAC are existing algorithms, not inventions of Polar Dynamics. Reference consultation date: 13 September 2026.
