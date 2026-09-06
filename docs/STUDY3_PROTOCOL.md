# Study 3: external evaluation of the explicit inter-polar network

Status at first commit: PROSPECTIVE; no Study 3 development or final outcomes inspected.
This is a new, finite synthetic control study within one unpublished manuscript.
The full layered architecture (Tension Engine, C/E/U, motivation, IACL and hybrid
interfaces) remains the research object. This study tests ONLY the implemented
inter-polar W/K pathways and a configured topology, not the complete architecture,
consciousness, psychological poles, AGI, ethics or a universal principle.

## 1. Questions and exclusions

H1 (tension use): does the selected network improve external tracking over the
SAME controller with K removed, without retuning?
H2 (network utility): does it improve over a separately tuned zero-network scaffold?
H3a/H3b (topology): does the nominated inter-polar ring improve over two equally
sparse alternative directed cycles, each tuned on the same development tasks?
H4 (feature specificity): does operational tension improve over a generic nonlinear
feature using the same observed inputs, state, graph support and tuning budget?

No claim will be rescued by selecting a favorable subgroup after final evaluation.
A positive H1 alone does not establish H2, H3 or H4. An exact non-named algebraic
implementation and signed-plus-intensity recoding are equivalence checks, not
independent superiority baselines. Configuring a network is not learning arbitrary
edge weights or a natural decomposition of psychological concepts.

## 2. Frozen mathematical object and information access

Use NetworkTensionModel from the preceding architecture extension, unchanged.
There is one centralized unit, FOUR operational polarities and eight nonnegative
actions, not four autonomous agents. Preserve the original contextual memory,
diagonal capability estimator, constraint map, action-feedback timing and dual
state. All current targets are observed in this study; episodic recall is NOT tested.
The estimator adapts to own measured effects online identically across conditions.
No controller sees the true environmental gain, topology, realized noise or answer key.
The contextual incompatibility chi is a declared input supplied to every controller.

For receiving pair i and source pair j=(i-1) mod 4, the W template has two entries
per receiving channel: +1 on the same source pole, -0.6 on the other pole. K has
one +1 entry per receiving channel from source-pair tension. Effective weights are
W=a*template_W, K=b*template_K, with shared gains a,b. These are 24 allocated edge
coefficients but only TWO independently tuned coupling gains, plus relaxation eta.
Tension is exactly tau=mean(abs(clip(d/G,0,1)-p),poles)+chi*p_plus*p_minus.
Full recurrence remains base + eta*(W p + K tau), then unchanged feasibility map.

Alternative sources are reverse cycle (i+1) mod 4 and rewired cycle [2,0,3,1].
They change physical routes, not merely labels or coordinates. Generic nonlinear
features are mean(tanh(2*(v-p)),poles)+chi*mean(p,poles), with v=clip(d/G,0,1).
They have the same p,d,G,chi information and no extra learned coefficients. The
feature differs in form/range; this is a matched engineered alternative, not an
exhaustive search over generic neural controllers. A generic gradient comparator
uses p + eta*G*(d-G*p)/max(1,max(G^2)), followed by the same feasibility map.
It has the same recurrent state and scaffold but no graph coefficients.

## 3. Conditions (12)

full; no_K; no_W; no_WK; retuned_zero; rewired; reverse; generic_nonlinear;
generic_equivalent; signed_intensity; gradient; zero_negative.
The three lesions inherit full's chosen eta/a/b and remove only specified routes.
Retuned_zero and gradient tune eta; all four independently tuned network models
(full, rewired, reverse, generic_nonlinear) use the same 27 candidate triples.
Generic_equivalent independently implements the full formula without polarity
labels, using the full configuration. Signed_intensity uses the invertible recoding.
Zero_negative returns zero while preserving input/feedback contracts.

Information, online estimator, constraints and time steps are identical. Parameter
counts are NOT claimed equal between graph-free and network models; counts are
reported explicitly. No dense/large trained controller is claimed resource-matched.
The targeted comparison is with limited engineered controllers, not frontier systems.

## 4. Factorial external tasks (12 cells)

Two synthetic families: signed linear cross-channel effects and cross-channel
congestion. Three plant topologies: aligned ring, independently rewired cycle,
and absent cross-effects. Two resource levels: full capacity or 0.40 of weighted
channel capacity. Thus 2*3*2=12 cells per seed and controller.

Each run has 64 action-feedback transitions, four 16-step demand phases A/B/C/A,
small sinusoidal target changes, a gain change after step 32, and one blocked
channel during steps 32..47. Costs, priorities, incompatibility, gain schedules,
noise, target demands and masks depend ONLY on seed, not plant topology or budget.
Topology and resource are independently manipulated; the same exogenous arrays
are reused across these factors and all controllers. Family differs only by its
additional nonlinear congestion term, not the demands or noise.

Effects are y=clip(g*u + 0.22*(u_source,s - 0.6*u_source,other)
 - 1[congestion]*0.35*chi_source*u_source,+*u_source,- + noise,0,1).
All cross-terms are removed in the absent-topology condition. Gains are drawn in
[0.8,1.2]; noise standard deviation is 0.005. No effects are constructed from a
controller's own tau or HGI. Actions, not internal agreement, determine outcomes.
Targets may be infeasible under scarcity; matched comparisons use identical demands.

## 5. Development, registration and untouched final sample

Mechanical tests use seeds below 800000. Development uses 801001..801006 only.
Each network evaluates eta in {0.35,0.65,1}, a,b in {-0.2,0,0.2}: 27 combinations.
Zero/gradient evaluate three eta values. Total development evaluation is
(4*27+2*3)*6*12 = 8208 runs. Every score is retained. One global configuration
per controller is chosen by mean tracking loss across all development seeds/cells,
then smaller abs(a)+abs(b), then eta,a,b for deterministic ties. No family-specific
selection, no final-data tuning and no requirement that a/b be nonzero.

The development-selected configurations, complete development results, source
hashes and this protocol will be committed PUBLICLY before final data generation.
The final runner requires that registration commit, verifies ancestry and that
its freeze file matches current bytes, and verifies all scientific source hashes.
Final seeds are 901001..901040: 40 complete paired experimental units, fixed now,
not a sample-size rule chosen after seeing effects. Final is 40*12*12 = 5760 runs,
368640 transitions. This provides finite precision, not guaranteed statistical power.
A final outcome error will be declared and the run invalidated, never silently
patched and called untouched. Replays of opened final seeds are not new evidence.

## 6. Outcomes and inferential unit

Primary: mean over all 64 steps of sum(w*(y-target)^2)/sum(w). Report raw MSE,
not a score based on HGI/INC. Unit for resampling is the COMPLETE SEED, after equal
averaging of its twelve factorial cells. Steps and channels are not replicates.
Secondary: cost=sum(cost*u)/sum(cost); post-switch MSE on steps 16..19,32..35,
48..51; return-phase MSE on steps 48..63; action clipping/saturation; hard action
violations; runtime for act+learn only. Runtime is descriptive and machine-dependent.
Do not rename post-switch loss as recovery time. No missing recovery is imputed as zero.

Use 20000 paired bootstrap resamples of the 40 seed summaries, RNG 917370.
For the FIVE directional claims H1,H2,H3a,H3b,H4, use a one-sided 99th percentile
upper confidence bound (Bonferroni alpha=0.05/5); also report ordinary two-sided
95% intervals descriptively. Percentile bootstrap coverage is approximate.
H1 and H2 require upper bound < -0.001 MSE. H3a/H3b/H4 require < -0.0005.
These are declared engineering relevance thresholds, not externally validated
clinical or psychological cutoffs. All five checks and all effect sizes are shown.

Additional gates versus retuned_zero: full's cost difference 99% upper <=0.05;
post-switch loss difference 99% upper <=0.005; zero hard constraint violations
(tolerance 1e-10). These one-sided approximate bounds are prespecified safety/
engineering gates, not an additional corrected family of superiority claims.
Validity requires exact expected trial set; complete 64-step traces; recomputed
outcomes; finite numbers; both equivalence maximum action differences <=1e-10;
and zero_negative worse than retuned_zero (99% lower bound >0.01).

## 7. Decision logic (unit-tested before final)

First invalid data/equivalence/negative controls => invalidate_study.
If all five directional tests and gates pass => support_configured_network_on_tested_tasks.
If H1,H2 and gates pass but any H3/H4 fails => support_network_not_exclusive_polarity.
If H2 and gates pass but H1 fails => network_benefit_without_tension_specific_evidence.
Otherwise => no_confirmed_network_advantage.
Separately report harm when full-minus-retuned_zero 99% lower >+0.001.
Do not convert a failed test into evidence of equivalence or universal absence.
Subgroup results by family/topology/resource are DESCRIPTIVE, never substitutes for
these decisions. No statement extends to a learned unrestricted topology or full C/E/U.
The previous Study 2 conclusion remains part of the unmodified research record.

## 8. Provenance and methodological basis

Archive per-run observed inputs, actions, measured effects, state/gain/proposal
and feature traces, initial/final mechanism snapshots. Time-invariant graph
matrices are stored once per run; per-edge contributions are reconstructible.
Retain each record's SHA-256 and hashes of small TAR parts; refuse missing,
duplicate or altered records. A separate report command recalculates outcomes
from archived observations/actions/effects and checks environment replay.
Report Python/NumPy and dependency versions. Bootstrap intervals condition on the
single development-selected configuration set; training-selection variability and
external task-family generalization are not estimated by these intervals.

References: Nosek et al. (2018), The preregistration revolution,
doi:10.1073/pnas.1708274114; Bouthillier et al. (2021), Accounting for Variance in
Machine Learning Benchmarks, arXiv:2103.03098; Agarwal et al. (2021), Deep RL at the
Edge of the Statistical Precipice, arXiv:2108.13264. Registration here is a public
versioned repository record, not independent peer review or an OSF registration.
