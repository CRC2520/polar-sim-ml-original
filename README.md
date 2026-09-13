# Polar Dynamics — functional relations and structural discovery

## Current priority: B1-S, before B1-E

**B1-E v2: ON_HOLD_PENDING_STRUCTURAL_DISCOVERY.** This is a new program-level decision, not a modification of B0 or the B1-D v1/v1.1 freezes. B1-D v1.1 remains technically B1D_COMPLETE; its null development findings and causal-admissibility limits remain unchanged.

The new question separates useful learned organization (H_REL), independently mapped catalogue correspondence (H_CAT) and transfer (H_TRANSFER). Learning physical effects is not learning organizational routing, and a useful generic graph is not automatically polar.

This branch contains **B1S_DESIGN only**: no new learner, task, controller, experiment or seed. Native-domain admission and selective interventions must be specified before implementation. Read the [implementation handoff](b1s/IMPLEMENTATION_HANDOFF.md), the [complete research plan in Spanish](https://github.com/CRC2520/POLAR_MODEL_CRC/blob/f954f1cfa96cbf92a9ae04e87301eb124d6ec73b/b1s/translations/es/STRUCTURAL_DISCOVERY_PLAN.md), the [program decision](b1s/TRANSITION.json), and the [publication audit](b1s/PUBLICATION.json). Scientific code and archived data remain unchanged.

The following sections preserve earlier study descriptions and reproduction commands; they are not commands to run B1-S or release the B1-E hold.

This repository supports one **unpublished research draft** in
[POLAR_MODEL_CRC](https://github.com/CRC2520/POLAR_MODEL_CRC).
Earlier manuscripts are development drafts retained through Git history, not
separate publications. The research studies contextual control with independent poles,
observable internal memory, resource coordination, and learned action effects.
Dynamic balance means responding to changing demands while retaining access to
both poles. It is not a requirement to make all units agree.

**These simulations do not establish consciousness, AGI, ASI, or a universal law
of polarity.** HGI and INC describe graph homogeneity and temporal/self-memory
alignment. External task performance, constraint violations, recovery, and causal
interventions are evaluated separately.

## Versions and specification

Start with [SPECIFICATION.md](SPECIFICATION.md), the normative version index.

| Version | Implementation | Role |
|---|---|---|
| Historical 2.0 | `engine_v1_locked.py`, `legacy/v2_0/` and original `results_*` | Preserved reference; known defects remain documented |
| Corrected 2.0.1 | `engine_v2_corrected.py` and root experiment scripts | Repairs historical execution, state timing, measurements and traceability |
| Contextual 2.1 | `polar/` | Explicit independent poles and functional mechanisms; a different state space |
| Coupling study 2.2 | `study2/` | Learned action-effect couplings, dynamic state estimation and finite-horizon regulation |

The original code is pinned at
[3c0be12](https://github.com/CRC2520/polar-sim-ml-original/tree/3c0be12a874e7ad46f4da0e77af2329c19cc61e8).
The original README is preserved in `legacy/v2_0/README.md`; original engine and
result directories remain untouched. `PROVENANCE.json` records source identities.
The companion repository contains one current article source and its compiled
`main.pdf`. Code versions and recorded datasets are research instruments within
that single evolving draft.

## Historical Study 2 research question

The causal hypothesis of that study concerns **using learned off-diagonal action effects
in planning**, not the names of poles. The primary planning lesion retains the
same estimator and disables those effects only in the decision rule. Diagonal,
mispaired, dense and coordinate-equivalent controls qualify the interpretation.
Dynamic balance remains task-dependent and does not require consensus or a
permanent midpoint.

- [Prospective coupling protocol](docs/STUDY2_PROTOCOL.md).
- [Coupling equations and implementation map](docs/STUDY2_CONTROLLER_SPEC.md).
- [Diagnostic analysis of the first study](docs/STUDY1_DIAGNOSIS.md).
- [Separate consciousness research protocol](docs/CONSCIOUSNESS_RESEARCH_PROTOCOL.md).

The first study is now exploratory development evidence. Its original recorded
outcomes are unchanged. Fresh final seeds in the coupling study are opened only
after source/protocol freezing and a prospective GitHub registration commit.
Registration is a public versioned record, not an independent endorsement or
an OSF registration. Tasks are designed inside this project; external replication
is a future milestone.

The 1,920 final trials show a practical benefit from using learned interactions
in planning versus the matched planning lesion (loss difference -0.0026998,
95% CI [-0.0028286, -0.0025766]). The proposed grouping is nevertheless worse
than shifted pairs with the same coefficient count and a larger dense estimator.
The research does **not** establish an exclusive polar advantage.

The frozen automatic classifier omitted the word **only** from one protocol
clause. Its raw `pivot` label is retained for audit; the literal protocol yields
**suspend_exclusive_polar_advantage_claim**. The separate
[post-run adjudication](results_study2/adjudication.md) documents this discrepancy
without changing statistics or frozen sources. Read it alongside the raw report.

## Reproduce

The recorded CPU environment uses Python 3.12.13. In a fresh virtual environment:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-repro.txt
python -m unittest discover -s tests -v
```

Study 2 uses NumPy and Matplotlib and records all raw observations, actions,
feedback and mechanistic traces in individually hashed gzip records, transported
inside TAR parts smaller than 6 MB. The regenerator verifies every archive and
record without extracting it. Large derived `trial_summaries.json` files and
redundant loose records are omitted from Git; regenerate them from the archives.

Rebuild the new study's reports and repeat the independent pilot replay:

```bash
OPENBLAS_NUM_THREADS=1 python scripts/regenerate_study2.py results_study2/coupling/pilot/manifest.json
OPENBLAS_NUM_THREADS=1 python scripts/regenerate_study2.py results_study2/coupling/final/manifest.json
OPENBLAS_NUM_THREADS=1 python scripts/audit_study2_pilot_replay.py
python scripts/run_functional_probes.py --regenerate
python scripts/adjudicate_study2.py
```

Re-execute the already registered final study into a **new** directory:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_study2.py --split final --out results_study2/replay_final --freeze docs/STUDY2_FREEZE.json --registration-sha fde8d0ac9b56b035a0a53c40c8cd5b79b2bc6b32
```

This reproduces existing evidence; its seeds are no longer unseen. It is not a
new confirmatory study. Runtime measurements can differ across machines and
loads even when numerical actions agree. Source/protocol mismatches fail
explicitly rather than silently reinterpreting archived data.

Run corrected reference experiments (8 conditions, 5 recorded seeds):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_p0.py --out results_corrected/p0 --seeds 2025 2026 2027 2028 2029
```

Reproduce the first contextual benchmark (13 controllers, 2 task suites, 5 development
and 30 held-out seeds, 96 transitions per run):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_benchmarks.py --output results_corrected/contextual
```

Use a different output directory to retain a prior run. `--smoke` creates a
separately labelled diagnostic subset; it must not replace the full benchmark.
The protocol is frozen internally before execution, not externally preregistered.
Development and held-out data come from variants of the same two task families.
No parameters were fitted to those scores in the original execution. Subsequent
diagnostic interventions reuse those data explicitly as exploratory evidence.

## Regenerate reports without running agents

```bash
python scripts/regenerate_p0_reports.py --index results_corrected/p0/index.json
python scripts/regenerate_p0.py
python scripts/compare_legacy.py
python scripts/regenerate_reports.py results_corrected/contextual
```

The P0 index links full state/control traces, paired controls, configuration and
seeds. Contextual traces join environment and controller records by split, task,
seed and transition. Arrays retain full floating-point precision in compressed
JSON. Latent mechanism snapshots and deterministic updates support reconstruction
between snapshots. The contextual exporter verifies checksums, evaluator source,
the full expected seed/task set and time steps before rebuilding metrics.
Missing or censored recovery stays unavailable; it is never reported as zero.

## Evidence and limitations

- [Priority acceptance map](docs/PRIORITY_ACCEPTANCE.md).
- [Corrected reference equations](docs/SPEC_LEGACY_CORRECTED.md) and
  [experiment repairs](docs/EXPERIMENTS_P0_CORRECTED.md).
- [Contextual equations and operational polarity proxies](docs/SPEC_CONTEXTUAL.md).
- [External evaluation protocol](docs/EVALUATION_PROTOCOL.md) and
  [machine-readable protocol](docs/evaluation_protocol.json).
- Corrected reference: `results_corrected/p0_reports/` and descriptive changes
  from original endpoints in `results_corrected/comparison/`.
- Contextual results: `results_corrected/contextual/generated/results.md`,
  full tables, paired bootstrap intervals and figures in the same directory.
- [Revision report](docs/REVISION_REPORT.md) links the actual verification evidence.
- [Six-step research progress and decision](docs/RESEARCH_PROGRESS.md).
- [Internal implementation review and public registration](docs/INTERNAL_REVIEW_STUDY2.md).
- Study 2: [final results](results_study2/coupling/final/generated/results.md)
  and [frozen protocol](docs/STUDY2_FREEZE.json).

Signed-plus-intensity coordinates contain exactly the same information as two
independent poles. Their comparison tests implementation equivalence. A matched
recurrent controller changes the update rule while retaining other mechanisms;
any observed difference cannot be attributed uniquely to polarity. Ablation
intervals are exploratory and unadjusted. Cue-addressed memory, heuristic
uncertainty and resource allocation are limited engineered functions. They are
not validated psychological measures or evidence of subjective experience.

## License

The original [LICENSE](LICENSE), CC BY-NC 4.0, remains in effect. The companion
manuscript is an unpublished research draft and has not been peer reviewed.


## Inter-polar network extension 0.1

The original layered architecture remains the research object. See [the explicit network tension specification](docs/NETWORK_TENSION_SPEC.md) for the new directed W/K pathways, engineering tests, legacy phase audit and remaining empirical requirements. This extension is not a new confirmatory study and does not change the frozen Study 2 conclusion.


## Integrated P0-P2 gap realization

The original layered architecture now has a bounded integrated implementation in `integrated_polar/`, with separate internal state/intention/action, learned directed effects, priority-sensitive planning, content broadcast, gated memory and institutional constraints. See [the Spanish gap-closure report](docs/P0P2_CIERRE_GAPS.md). The 128 tests passed. The registered P1 decision is `engineering_verified_with_partial_empirical_support`: priority use and return-cue memory pass the practical criteria; coupled planning and the longer horizon do not. The P2 probability forecast is calibrated on the tested domain, but its experimental review policy harms tracking and is not enabled by default. General ethics, intrinsic motivation, philosophical semantics and consciousness are not established. Prior Studies 1–3 remain unchanged.

The new experiment entry point is `python -m gap_resolution.run regenerate`; final seeds are already observed and their replays are not new confirmation. `docs/P0P2_ADDITIONAL_AUDIT.json` records post-run diagnostic/capability verification, separate from the original final records.
