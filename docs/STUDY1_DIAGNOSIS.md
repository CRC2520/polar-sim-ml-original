# Exploratory diagnosis of the first computational study

This is a post hoc diagnosis of **previously observed Study 1 data**, not a new held-out evaluation. The frozen `polar/` sources and the original `results_corrected/` artifacts are unchanged. The purpose is to distinguish implementation defects, estimator behavior and designed tradeoffs before selecting a new research hypothesis.

## Exact failed cases

The two failed `dual_pole` evaluations are both in `switching_memory`:

| Prior seed | Recall regret | Required maximum | Excess over threshold | Other external criteria |
|---|---:|---:|---:|---|
| 1012 | 0.015247023893153959 | 0.015 | 0.000247023893153959 | All pass |
| 1020 | 0.015101448755286203 | 0.015 | 0.000101448755286203 | All pass |

Both runs recover after all five phase changes, commit no hard constraint violations and satisfy whole-trial regret and effect-discrimination requirements. Failure is therefore narrowly due to the previously specified aggregate recall criterion; it is not numerical divergence, missing memory or loss of both poles. The near-threshold failure counts alone do not establish a general inferiority of the controller. The original study was not externally preregistered, and its partly observed evaluation seeds are now explicitly treated as exploratory data.

## What the traces show

Recall occurs at zero-based steps 48–63. The stored cue-A target is **exactly correct** throughout that phase: stored-target MSE is zero. Its confidence multiplier, however, falls from **0.797687** to **0.739910**. The implemented rule is

\[
c=\frac{n}{n+1}\,0.995^{\mathrm{age}},\qquad
\widetilde y=\mathrm{memory\ value}\times c,
\]

with 16 previous observations and ages 33–48. This is deterministic amplitude attenuation. It is not a calibrated probability that the memory is correct. At recall entry the effective-target MSE is 0.008034 for seed 1012 and 0.007203 for seed 1020; it grows to 0.013279 and 0.011905 by the final recall step. Correctly storing a value and attenuating its subsequent use are separate operations.

A second contribution comes from the capability estimator. Its update uses the ratio of realized effect to own action whenever action exceeds `1e-6`:

\[
\widehat g\leftarrow \widehat g+0.35
\left[\operatorname{clip}\left(\frac{\operatorname{clip}(g a+\epsilon,0,1)}{a},0.1,4\right)-\widehat g\right].
\]

When previous targets turn off a channel, its action can become tiny compared with observation-noise SD 0.002. The ratio is then unstable, and clipping both the observed effect and the estimated ratio can create asymmetric errors. Before a previously inactive channel is needed again, the estimator can retain a distorted gain even though the true gain is one. Counts can nevertheless be large because they count updates, not informative excitation.

At recall entry, on the channels currently required by the target:

| Prior seed | Mean estimated gain | Maximum estimated gain | Mean planning gain | Estimator observation counts |
|---|---:|---:|---:|---:|
| 1012 | 1.262192 | 2.945243 | 1.220867 | 28–48 |
| 1020 | 1.350470 | 3.504983 | 1.295255 | 28–48 |

The inverse planner divides the already attenuated remembered target by its planning gain, reducing action further when that gain is too high. This is an observed effect in these data, not a claim that ratio estimation always has upward bias. Other channels also have downward errors. During the failing runs' recall phase, action clipping and resource rationing are absent; their minimum resource scale is one. These two failures therefore do not come from a resource budget or stability bound.

## Paired causal interventions

The diagnostic script uses all 30 previously observed memory-task seeds and identical targets/noise between controller interventions. It additionally includes one intervention on environmental noise. Ten conditions give **300 exploratory runs**. Unmodified `dual_pole`, `recurrent` and `no_self_model` replay the original mean-regret, recall-regret, reacquisition-regret and RMSE values exactly, with identical pass/fail classifications: 90 baseline replays, maximum checked difference zero.

| Intervention | Mean recall regret | External passes | Interpretation within this task |
|---|---:|---:|---|
| Original dual pole | 0.012783 | 28/30 | Reference |
| Matched recurrent | 0.011477 | 30/30 | Different update rule tolerates these conditions better |
| Remove capability model throughout | 0.010688 | 30/30 | Capability estimation contributes to recall loss here |
| Reset capability state at recall entry | 0.010689 | 30/30 | Removing the inherited estimator state removes much of the additional loss |
| Reset only gain at recall entry | 0.010689 | 30/30 | Replacing gains also helps while preserving count and uncertainty state |
| Freeze capability updates during recall | 0.020983 | 0/30 | Preserving inherited errors prevents corrective feedback; this intervention harms performance |
| Reset and then freeze during recall | 0.010689 | 30/30 | Initial estimator error matters more than subsequent noisy updates in this phase |
| Set stored-channel confidence to one during recall | 0.003046 | 30/30 | Confidence attenuation contributes substantial error even when stored values are exact |
| Remove noise throughout the trial | 0.010689 | 30/30 | Noiseless estimation removes the additional capability-related loss |
| Suppress estimator updates for actions below 0.02 | 0.010698 | 30/30 | Weak-excitation updates are a causal route for this loss |

The gain-only reset preserves counts and uncertainty at the intervention time. It separates replacement of the gain estimate from the broader capability reset. All exact values are in `paired_recall_effects.csv`.

For the capability reset, the paired change in recall regret is **−0.002094**, with a descriptive 95% seed-bootstrap interval **[−0.002351, −0.001825]**. For freezing updates without reset, the change is **+0.008200**, interval **[+0.007155, +0.009162]**. These are post hoc, unadjusted intervals across previously observed seeds; they are not independent confirmatory evidence. All exact effects, including adverse effects, are retained in the generated data.

The value 0.02 is a diagnostic threshold equal to ten original noise standard deviations, not a tuned optimum. A gain reset to one is well matched to this particular task's true constant gain. Unit confidence is an instrument that removes one attenuation path; it is not evidence that forgetting should be disabled in unfamiliar environments. These instruments should not be silently installed as universal fixes.

## Why removing the capability model is not a general solution

The original second task has real gain changes. There, mean regret is **0.004625** with the full controller but **0.013432** without the capability model; external passes decline from 30/30 to 29/30. The mechanism helps when adaptation to gain changes is needed and hurts when estimates are contaminated by weakly informative observations. The correct engineering target is estimator reliability and context-dependent use, not removal of the entire mechanism.

The recurrent comparator also uses the same memory, uncertainty estimate and resource workspace. Its advantage on recall therefore cannot be attributed to a distinct memory store. The main comparison is inverse-gain planning versus a projected-gradient recurrence. The two-state polar representation remains coordinate-equivalent to signed intensity; this diagnosis supplies no uniquely polar advantage.

## Consequences for the next study

1. Treat the old seeds as development/diagnostic data from this point forward. Do not relabel these interventions as a fresh held-out result.
2. Preserve the existing first-study implementation and report the failure mechanism candidly. Any new estimator or attenuation rule must have its own explicit specification.
3. Evaluate estimator reliability using excitation and predictive error. A new study should include noise and gain shifts so that both benefits and risks of adaptation can appear.
4. Separate storage quality, retrieval confidence and action strength. A confidence rule should be tested against environments where old memories become invalid, instead of assuming a remembered value is always reliable.
5. Freeze the new study's implementation, thresholds and sample-size rule before final evaluation. Include matched controls and adverse cases for the proposed contextual coupling; the present diagnosis does not establish that coupling's benefit.
6. Treat memory and self-estimation as functional mechanisms. Their measured effects are not evidence of experience, sentience or a universal law of polarity.

## Reproduction and artifacts

From the repository root:

```bash
python scripts/diagnose_study1.py --output results_study2/diagnosis_replay
python -m unittest tests.test_study1_diagnosis -v
```

The script verifies frozen first-study implementation hashes before execution. `manifest.json` records source/input/output SHA-256 values, seeds, runtime versions, all intervention definitions, replay accuracy and the data contract. `failure_cases.json` records each failed criterion explicitly. `summaries.csv` and `summaries.json` contain run-level results, `paired_recall_effects.csv` contains paired effects and uncertainty, and `previous_task_tradeoff.csv` preserves the contrasting second-task outcomes. `step_diagnostics.csv.gz` contains all per-step scalar diagnostic measurements; `failure_channel_details.csv.gz` records individual channels around recall for both failed cases. `diagnosis.tex` is the manuscript table generated directly from those results.

Complete original trajectories remain in `results_corrected/contextual/`; new diagnostic trajectories can be recreated from frozen inputs, code and declared interventions. Missing CSV values mean undefined, not zero. The intervention tests check localization in time, preserved memory, selective weak-excitation updates and explicit missing-data handling.
