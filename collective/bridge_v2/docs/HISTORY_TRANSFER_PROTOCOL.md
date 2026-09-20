# P2: matched-state history interventions and fixed-policy transfer

Status: implementation protocol; final seeds may only be run after the repository-wide implementation freeze. This is a new bridge-v2 experiment, not a reanalysis of the v1.3 precursor and not a consciousness test.

## Identification

Two histories are generated under distinct ecological renewal rates with identical training budgets. At the intervention boundary, current resource, each individual energy, alive mask, time, configuration, future random generator state, and current observational inputs are matched. Previous physical resource values remain separate from learned controller state. First-decision differences therefore test stored influence under matched current observations; later differences include feedback and continued learning.

Interventions are bidirectional wherever meaningful: Q tables only; visit counts only; Q plus visit counts; delayed physical resource history only; lineage identifiers only; and all retained historical state. The lineage intervention is a negative control when identifiers do not participate in the policy or selection. Sham cloning must exactly preserve complete trajectories. Full retained-state transplantation must reproduce the donor continuation under the same future random draws. Reinitializing present viability is an artificial matched-state intervention, not a natural ecological recovery.

Q tables and counts may interact, and ecological history may interact with both. Their effects are not additive attribution percentages. The complete two-factor learned-state by ecological-history cross is reported. Persistent history dependence does not by itself establish dynamical hysteresis or two stable attractors.

## Transfer

Train each controller using the same observations, action constraints, source ecology, and learning budget. Evaluate on a source condition and three untouched targets: reduced renewal, increased metabolic demand, and a predeclared initial resource shock. Zero-shot evaluation freezes controller learning; continued-learning evaluation is analyzed separately. A reset-controller baseline measures retained learning relative to no source learning. A conventional comparator receives exactly the same observations and task rewards. An equivalent-coordinate controller is an implementation-consistency control and cannot establish representation superiority.

The primary outcome is `alive_fraction`: the mean fraction alive over all agents, groups, and the 240 probe steps, computed separately for each seed. Secondary outcomes are mean resource/capacity, restraint frequency among eligible decisions, mean action, and the fraction of groups whose time-averaged survival is at least 0.8 and resource fraction is at least 0.2. These are accompanied by exact compact resource, alive-state and action traces. Vitality, observations, rewards and the full learning trajectory are reconstructed deterministically from complete initial state arrays, configurations, seeds and time keys. Report paired seed-level differences with uncertainty, not agent-step pseudoreplication. Include failures and extinctions. Outcomes remain descriptive if the pilot-based precision/power requirement is not met.

## Reproduction

Save each initial/transition-boundary state as a JSON recipe referencing content-addressed exact arrays, future random time keys, compact lossless resource/alive/action trajectories, configuration, seeds, and source hashes. This storage contract does not claim to retain every intermediate tensor; the remaining state and reward trajectory is deterministically reconstructible. A separate deterministic replay reloads stored states and recomputes a fixed subset of results. This is computational reproduction within the same project, not replication by an external laboratory.

## Boundaries

This study does not establish phenomenal consciousness, autonomous values, a universal law of polarity, or superiority of semantic labels. The collective task currently tests operational controllers and ecological mechanisms. New polar architecture claims require their own functioning implementation and matched causal controls.

## Frozen numerical design

The manifest is generated directly from `history_transfer.manifest()`. It fixes 30 paired final seeds (920001–920030), three pilot seeds (919001–919003), 20 agents in each of six groups, six source-learning episodes of 240 steps, and 240-step probes. Histories use renewal 0.12±0.035 and 0.22±0.035; transfer uses the source, lower renewal 0.12±0.035, metabolism 0.42, and an initial resource shock to 0.35 capacity. All four targets are retained regardless of outcome. There is no social copying during P2 training or probes: these assays isolate local learning and environmental history, not population lineage evolution. For the lineage negative control, donor labels are cyclically permuted within groups after training, preserving composition.

Inference is exploratory and descriptive: paired seed bootstrap intervals are reported without confirmatory multiplicity claims. Thirty seeds are selected for a bounded mechanism/transfer evaluation and do not imply a power guarantee. The six short training episodes have independently keyed ecological phases; this learning horizon does not establish asymptotic performance. Zero-shot evaluation freezes both values and visit counts throughout its 240 steps.

## Relationship to recovered prior experiments

The recovered v1.37 package already includes `polar_HISTERESIS2` selection-pressure cycles and corrected `polar_AVATAR_POLAR_V2` transfer with strong softmin controls, as well as RIR transfer studies. These precedents are not erased or described as pending. The present P2 assay adds matched-present-state interventions separating Q values, visit counts and delayed ecological history in bridge-v2, and a bounded within-task transfer/reconstruction assay. It is not the first POLAR transfer study, not a direct replication of the v1.37 engine, and not stronger evidence for polar naming than those earlier comparisons. It does not test lineage selection because social copying is deliberately disabled and lineage is an inert negative control after initialization.
