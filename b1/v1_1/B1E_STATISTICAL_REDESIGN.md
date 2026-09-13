# B1-E protocol v2 candidate: fixed prospective paired-mean design

The proposed sample size is **138,688 independent P7 bundles**, fixed before final seed generation. The frozen v1 design remains **915,501 bundles per pilot**. This is a new prospective protocol candidate, not an amendment to historical observations. No Polar model, B1-E experiment, or final seed generator was executed by the statistical work. All attached method-validation outputs are development-only, nonconfirmatory, and unusable as final evidence.

## Scientific scope and schedule

The eligibility decisions come from the frozen `B1_DECISION_RULES_v1.yaml`, not from the sign of P7 development differences. P5 and P6 retain all nine registered endpoint summaries each as engineering/negative controls, with 32 prospective engineering bundles per pilot. Their positive utility and positive pairing specificity eligibility remain false. This engineering replication count does not support a powered population claim. P7 retains all nine registered interval endpoints: two source mechanism effects, their context interaction, four ordinary paired comparator differences, and two acute route differences. C5 stays descriptive and C6 stays an algebraic consistency control.

P7 collects all 12 episodes in every bundle before any hypothesis testing: C0, C1, both C2 cycles, C3, C4, C5, C6, and intact/bypass realizations in both acute contexts. It also retains the 24-event source diagnostic schedule. P6 uses the same 12-episode engineering schedule. P5 adds the separately labelled full-refill ceiling witness for 13 episodes. The 18 P5/P6 endpoints are not silently dropped or pooled into P7.

## Selected inference and preserved margins

Inference uses the paired bundle-level difference itself, with one observation per independent bundle. C2's two cycles are averaged inside that bundle. Jobs, epochs, repeated routes, and source diagnostics inside a bundle are not treated as independent replicates.

For `N >= 2`, unbiased sample variance `s²`, known support width `R`, family size `m = 9`, and family error `alpha = 0.05`, use

```
t = log(4*m/alpha) = log(720)
h = sqrt(2*s²*t/N) + 7*R*t/[3*(N-1)]
CI = clip([sample_mean-h, sample_mean+h], known support)
```

The bound follows by rescaling the one-sided inequality and allocating error to both tails. It does not plug in a development variance. See [Maurer and Pontil, Theorem 4](https://arxiv.org/pdf/0907.3740).

| Endpoint group | Support | Practical delta | Equivalence epsilon | Target half-width |
|---|---|---:|---:|---:|
| Utility and acute route | [-1, 1] | 1/64 | 1/128 | 1/256 |
| Source mechanism | [-1, 1] | 1/2 | 1/4 | 1/8 |
| Context interaction | [-2, 2] | 1/2 | 1/4 | 1/8 |

These are the existing engineering margins. None changed to reduce N. A favorable utility label requires `CI.upper < -1/64`; worsening requires `CI.lower > 1/64`; equivalence requires the whole interval strictly inside `(-1/128, 1/128)`. Imprecision remains inconclusive. Equivalence is a positive interval-containment requirement, not failure to reject a zero effect; compare the [two one-sided tests principle](https://doi.org/10.1007/BF01068419).

## Multiplicity and positive gates

All nine P7 intervals are simultaneous. Each interval's failure probability is at most `0.05/9`; the union bound controls any miss at 0.05 without independence among endpoints. Every incorrect scalar improved/equivalent/worsened declaration implies a miss by its corresponding interval. Requiring multiple favorable comparisons, and applying a hierarchy, only removes declarations from this event. This gives strong familywise protection for the scalar classifications and their composite claims.

Positive claims must pass, in order: instrument and admissible local mechanism; practical utility against both C0 and C4; pairing specificity against all C0/C2/C3/C4; context moderation. All observations are collected at fixed N even if an earlier positive gate fails. Negative and equivalence reports remain visible. This is a fixed scientific gate order over simultaneous intervals; it does **not** recycle alpha or claim the smaller critical values of an alpha-recycling procedure. Ordered testing can control FWER, but the exact hypothesis structure matters; see [Edwards and Madsen](https://doi.org/10.1002/sim.2905).

Source/sham remains `joint_manipulation=true`. The frozen causal admissibility guard remains active, so eligibility does not itself authorize a positive source-causal claim. The source diagnostic cannot demonstrate selective Gamma necessity. Acute route contrasts are distinct; no route result can rescue failed practical utility or pairing gates. An instrument PASS is not a causal PASS.

**Consequence for the current realization:** because the source/sham fixtures are joint, `source_causal_admissible=false`; this hierarchy cannot issue any aggregate positive H_mechanism, H_utility, H_pairing, or H_context claim, whatever the signs of future scalar differences. Correct outputs remain not_supported/inconclusive where applicable, and corrected scalar improved/equivalent/worsened/inconclusive results remain separately reportable. This is an explicit scientific claim limitation. It is not a hidden promise that a favorable aggregate verdict is achievable under the current fixtures, nor a reason to relax a guard or retune P7. Any future causal redesign would need another prospective protocol. Technical completion can coexist with this valid negative/inconclusive outcome path.

## Conservative development planning, not known variance

`statistics/build_design.py` independently reconstructs all six tight-margin P7 contrasts from the 384 archived calibration trials (32 bundles, 12 episodes each) using exact rational arithmetic. It verifies the full comparator/cycle/context/lesion identity set and matches the frozen v1 report. Every one of the six contrasts has 32 zero values. The original trials, identity map, selected configuration, tuning ledger, technical close, and results are referenced by repository, commit, path and SHA-256 in the JSON artifact.

For a paired difference `D` in [-1,1], `E[D²] <= Pr(D != 0)`. With zero nonzero events in 32 iid development bundles, a one-sided exact binomial upper bound, Bonferroni-corrected across six planning endpoints, is

```
q = max(1/64, 1 - (0.05/6)^(1/32))
  = 0.13895552166407366
```

The declared variance floor is 1/64. The observed zero variance is not treated as known zero. The elementary proof is `Pr(all 32 events absent) = (1-p)^32`: values `p > q` would produce that observation with probability below `0.05/6`. The joint development bound can fail with probability at most 0.05. If any development count were nonzero, the locked implementation uses the conservative second-moment ceiling 1 instead of fitting a convenient substitute.

For an independent future bundle sample, Hoeffding applied to `D² in [0,1]`, followed by a six-endpoint union bound, gives with probability at least 0.95, **conditional on valid development moment bounds and an unchanged bundle law**,

```
s² <= Vplan(N)
Vplan(N) = N/(N-1) * min(1, q + sqrt(log(6/0.05)/(2*N)))
```

The inequality uses `s² <= N/(N-1)*mean(D²)` and never assumes the unknown mean is zero. The smallest integer meeting the preserved utility half-width, together with the looser source/context worst-support conditions, is N = 138,688. At that N, `Vplan = 0.14311106135429483` and the planned radius is `0.0039062395446212454`. At N−1 it is `0.003906254618612961`, which exceeds the required `0.00390625`. There is no computationally convenient cap. The rule's worst-second-moment fallback would require 878,007 bundles.

The joint development-plus-future planning guarantee is **at least 90%**, using a conservative union bound across two 5% errors; it is not an unconditional 95% precision promise. Conditional future precision is at least 95%. The inferential family coverage remains at least 95% regardless of whether the planning bound is correct, because the final CI always uses the actual final sample variance. Development-adaptive N is fixed before the independent final sample.

This is a precision design. Conditional on valid moment bounds, combining the 95% precision and 95% coverage events provides a conservative 90% classification guarantee for sufficiently separated means, with strict interval-containment requirements for equivalence. It does not promise power at a practical-margin boundary or under every bounded high-variance distribution. The synthetic power results below are scenario-specific.

The planning information transfers only to the same P7 task, selected policies, and iid bundle law. Resource certification must verify unchanged behavior. A task/policy change invalidates this derivation before final freeze. If a future interval misses its precision target, retain it as inconclusive: do not extend N, change a margin, replace a seed, or substitute a variance floor in that CI.

## Candidate comparison

| Candidate | Disposition and reason |
|---|---|
| Fixed-sample empirical Bernstein | Selected: finite-sample bounded-mean inference accommodates paired variance without Gaussian assumptions. |
| Paired t/normal CI | Not selected for primary guarantees: exact t needs Gaussian differences; CLT validity is asymptotic and does not supply this finite-sample guarantee. |
| Paired permutation/randomization | Shared seeds alone do not imply sign-exchangeability or randomized treatment labels. Sharp-null tests do not automatically test a heterogeneous weak mean/margin null; see [Chung and Romano](https://arxiv.org/abs/1304.5939). |
| Exact/binomial methods | Used only for the across-bundle nonzero indicator in planning. A 192-job loss fraction is not Binomial(192,p) because jobs share state and history. |
| Equivalence/noninferiority | Equivalence uses simultaneous CI containment with unchanged epsilon. Noninferiority alone would not establish utility superiority or pairing necessity. |
| Holm, closed testing, fixed sequence, recycling | Potentially valid with specified elementary hypotheses. Simultaneous intervals plus restrictive gates were selected to preserve rigorous negative/equivalence reporting after failed positive gates. |

## Statistical validation

The current validation has 40 cells with 20,000 synthetic replications each. It tests N = 138,688 and a 32-observation stress condition, independent and perfectly dependent endpoint coordinates, nine utility-law scenarios including both equivalence boundaries, and a context-support rescaling case. Exact multinomial sufficient statistics are sampled from declared bounded laws. The simulator does not approximate samples by a normal distribution and imports no Polar task, controller, or final seed API.

| N = 138,688; independent coordinates | Family coverage | Correct scalar classification | All nine meet precision |
|---|---:|---:|---:|
| Zero effect | 0.99980 | 1.00000 | 1.00000 |
| Practical-margin boundary | 0.99990 | 0.00000 | 1.00000 |
| Positive equivalence boundary | 0.99995 | 0.00000 | 1.00000 |
| Negative equivalence boundary | 1.00000 | 0.00000 | 1.00000 |
| Favorable effect | 0.99985 | 1.00000 | 1.00000 |
| Adverse effect | 0.99990 | 1.00000 | 1.00000 |
| Bounded rare extreme tails | 0.99865 | 0.00000 | 0.00000 |
| Discrete job-fraction law | 1.00000 | 1.00000 | 1.00000 |
| Heterogeneous variance mixture | 0.99880 | 1.00000 | 1.00000 |
| Context range-4 boundary | 0.99835 | 0.00000 | 1.00000 |

Across all cells, minimum family coverage was 0.99835 and maximum false-declaration probability observed was 0.001 (20/20,000 in the context boundary case). The utility practical-boundary false-improvement frequency was 1/20,000. The JSON records exact binomial Monte Carlo uncertainty for every rate. Zero observed errors do not imply zero error probability. The high-tail variance exceeds the planning bound and correctly fails the precision requirement; it does not cause a smaller CI by borrowing the development variance.

An exact Binomial(0.05) upper-tail regression check is applied to each cell's miscoverage and three false-declaration rates, with QA false-alarm budget `0.01/(4*40)`. The analytical proof is the primary guarantee; simulation checks these implementations and these distributions. The original smaller synthetic QA artifact and its design/source snapshots remain archived as revision 1. Revision 2 adds independent reviewers' negative-boundary/range-rescaling tests and the nominal-error regression gate; it does not change N or the inference method.

## Reproduction and remaining gates

Run from the code repository:

```
python -B -m unittest b1.v1_1.statistics.test_statistics -v
python -B -m b1.v1_1.statistics.simulate --smoke
```

There are 11 statistical unit/contract checks, including exact archival reduction, fixed-N minimality, zero-variance uncertainty, support rescaling, missing-input rejection, nonbinding high-variance behavior, negative-control eligibility, joint-causal guards and source hashes. The saved design reproduces without writing. Full statistical simulation may be written only to a new path; archived output is never silently overwritten.

Resource feasibility and retention are certified separately. This candidate does not authorize B1-E. External preregistration, final code/config/retention freeze, independent seed custody, final entropy, and explicit resource/execution authorization remain future requirements. `ready_for_b1e_confirmatory_run=false`.
