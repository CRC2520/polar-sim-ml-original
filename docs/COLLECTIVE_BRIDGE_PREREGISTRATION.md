# Preregistered collective-viability experiments A and C

Status: **design freeze; no confirmatory runs executed**  
Date: 2026-09-18  
Base repository commit: `7b9a0a3f7d28fab9184c3ac12a216c43c1da2fbb`

## Purpose and evidential boundary

These experiments test the proposed multilevel extension of POLAR. Experiment A identifies whether the difference between success-biased imitation and conformist transmission is mediated by private payoff. Experiment C tests the missing bridge: whether a controlled change in local regulatory architecture moves the population basin boundary while transmission and ecology are held fixed.

Neither experiment tests phenomenal consciousness. A positive result in A would consolidate a generic transmission mechanism. Only C can support the proposed cross-level coupling, and even a positive C result will not establish specificity to POLAR until matched non-polar comparators are tested.

The v1.3 delivery supplied after this design freeze is preserved under `collective/precursor_v1_3/polar_AC/`. It contains the collective engine and an earlier eight-seed A/C study. That study is precursor evidence, not execution of this strengthened protocol: it uses one initial composition, final restraint rather than separation point, an inherited binary trait, and no equivalent-coordinate control. Confirmatory execution remains blocked until the implementation freeze, power analysis, immutable seed list and environment lock required below are committed.

## Shared rules

- Use new, previously untouched seeds, generated and frozen before execution.
- Run a simulation-based power analysis before outcome inspection. Target at least 80% power for the smallest effect of scientific interest; use 30 seeds per cell as the minimum and increase to at most 50 without changing the estimand.
- Pair seeds across all conditions within each experiment and preserve complete trajectories.
- Hold population size, initial resource, delayed resource history, observation noise, mutation, episode length, learning budget and compute constant unless explicitly manipulated.
- Treat seed/trajectory, not agent-step, as the inferential unit.
- Estimate uncertainty by paired trajectory bootstrap, stratified by initial composition.
- Use two-sided confirmatory tests. Report effect sizes and intervals even when tests are not significant.
- Apply Holm correction within each experiment's declared confirmatory family. Exploratory outcomes are labelled and cannot change the confirmatory decision.
- Record exclusions mechanically before outcome analysis: corrupt/missing trace, non-finite state or violation of a frozen invariant. Do not exclude extinction or non-transition.
- Do not tune thresholds, smoothing or initial-composition grids after viewing confirmatory outcomes.

## Experiment A — private-benefit equalization

### Question

Does removing the private survival advantage of non-restraint attenuate erosion under success-biased imitation more than under conformist transmission?

### Design

Paired factorial:

- transmission: success-biased imitation vs conformity;
- payoff: original vs private-benefit equalized;
- initial restraint composition: frozen grid spanning both known separation regions, provisionally `0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70`;
- identical ecology, information, mutation and learning budget.

Equalization must be implemented as a predeclared intervention on private payoff/survival accounting. It must not directly alter resource extraction, transmission probabilities, observations or the population-level viability outcome. A calibration phase may use only non-confirmatory seeds and must be frozen before generation of confirmatory seeds. Its target is an absolute standardized survival difference between restrained and unrestrained agents no greater than 0.05 in the equalized condition.

### Primary estimand and hypothesis

For each transmission rule, estimate the shift in separation point caused by equalization:

[
Delta_m = p^{*}_{m,mathrm{equalized}}-p^{*}_{m,mathrm{original}}.
]

The primary interaction is

[
I_A=Delta_{mathrm{success}}-Delta_{mathrm{conformity}}.
]

Because lower separation points indicate easier entry into the viable/high-restraint basin, the directional prediction is `I_A < 0`: equalization lowers the boundary more under success-biased imitation. Confirmatory inference remains two-sided.

### Secondary confirmatory outcomes

1. Difference-in-differences in final restraint prevalence.
2. Difference-in-differences in population survival/viability.
3. Manipulation check: reduction of the restrained–unrestrained private survival gap.

### Decision rule

A supports payoff-mediated copying only if:

- the equalization manipulation meets its frozen tolerance;
- the 95% interval for `I_A` excludes zero after Holm correction; and
- the direction is the preregistered one.

If equalization materially changes extraction or ecological renewal, the causal interpretation fails even if the statistical criterion passes.

## Experiment C — bridge from local regulation to population dynamics

### Question

Does local regulatory quality causally move the population separation point under otherwise identical transmission and ecology?

### Endogenizing restraint

The binary restraint label must no longer be assigned independently of the controller. At each eligible decision, restraint is generated by the local policy from the same observable state. The policy must expose a scalar restraint propensity or declared action mapping. Transmission may copy the frozen transmissible state, but cannot overwrite the randomized architecture factors.

### Factorial intervention

Local architecture factors:

- vital anchor: present vs ablated;
- valuation horizon: short vs long, with exact step counts frozen in the implementation manifest;
- exploration: state-directed vs matched-rate blind exploration.

Population context:

- transmission: success-biased imitation and conformity, analysed separately and jointly;
- initial composition: the same frozen grid as A unless simulation-based power analysis, completed before runs, selects a denser grid;
- ecology and information: identical across architecture conditions.

The directed and blind exploration conditions must have matched realized exploration counts within a predeclared tolerance, so the contrast concerns allocation rather than exploration quantity. Short and long horizons must use matched parameter counts; runtime and decision evaluations must be reported.

### Primary estimands

For architecture condition `a` and transmission rule `m`, estimate separation point `p^*_{a,m}`. The primary contrast is the joint architecture effect:

[
I_C = p^*_{mathrm{ablated,short,blind}}-p^*_{mathrm{anchor,long,directed}}.
]

The directional prediction is `I_C > 0`: the complete local regulator reaches the viable basin from a lower initial restrained composition.

Three component contrasts—anchor, horizon and directed exploration—form the remaining confirmatory family. Interactions among components and with transmission are secondary unless named in the machine-readable freeze before execution.

### Mechanism checks

- local vital-margin violations;
- proportion of decisions classified as restrained;
- policy switching after resource shocks;
- access to both action poles and coactivation where the task permits it;
- realized exploration count;
- private survival and shared-resource trajectories.

### Discriminating criterion

C supports cross-level coupling only if:

- the adjusted 95% interval for `I_C` excludes zero in the predicted direction;
- at least one prespecified local mechanism check changes in the predicted direction;
- transmission, ecological parameters, compute budget and information are demonstrably matched; and
- the effect survives an equivalent-coordinate control using signed-plus-intensity representation.

Failure of the equivalent-coordinate control to differ means the result supports generic local regulatory quality, not a distinctive polar representation. A negative C result rejects or sharply narrows the multilevel bridge; it must not be rescued by relabelling poles.

## Analysis freeze required before execution

The implementation PR must add:

1. exact package hash and provenance;
2. executable environment lock;
3. source hashes for the collective engine and both interventions;
4. fixed numerical values for horizons, tolerances, population size and episode length;
5. seed-generation procedure and immutable seed list;
6. power-analysis script and selected sample size;
7. separation-point estimator, bootstrap procedure and Holm family;
8. trace schema and invariants;
9. dry-run tests using only non-confirmatory seeds;
10. a freeze JSON whose hash is reported before the first confirmatory run.

## Planned sequence

1. Preserve the verified v1.3 precursor and keep its results separate from this confirmatory design.
2. Adapt and unit-test A from the preserved engine; calibrate equalization on non-confirmatory seeds.
3. Freeze A and execute it before implementing outcome-informed changes to C.
4. Implement the endogenous local-policy bridge and equivalent-coordinate control.
5. Freeze and execute C.
6. Only after A and C, add strong non-polar comparators and transfer environments.
