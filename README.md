# Polar Dynamics — corrected reference and contextual prototype

This research revision implements the ten priority corrections to the original
Polar Dynamics proposal. It studies contextual control with independent poles,
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

The original code is pinned at
[3c0be12](https://github.com/CRC2520/polar-sim-ml-original/tree/3c0be12a874e7ad46f4da0e77af2329c19cc61e8).
The original README is preserved in `legacy/v2_0/README.md`; original engine and
result directories remain untouched. `PROVENANCE.json` records source identities.
The companion [manuscript revision](https://github.com/CRC2520/POLAR_MODEL_CRC/tree/research/polar-v2.1-priority-corrections)
contains the revised article and separately preserved original material.

## Reproduce

The recorded CPU environment uses Python 3.12.13. In a fresh virtual environment:

```bash
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-repro.txt
python -m unittest discover -s tests -v
```

Run corrected reference experiments (8 conditions, 5 recorded seeds):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_p0.py --out results_corrected/p0 --seeds 2025 2026 2027 2028 2029
```

Run the frozen contextual benchmark (13 controllers, 2 task suites, 5 development
and 30 held-out seeds, 96 transitions per run):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/run_benchmarks.py --output results_corrected/contextual
```

Use a different output directory to retain a prior run. `--smoke` creates a
separately labelled diagnostic subset; it must not replace the full benchmark.
The protocol is frozen internally before execution, not externally preregistered.
Development and held-out data come from variants of the same two task families.
No parameters are fitted to held-out scores.

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

Signed-plus-intensity coordinates contain exactly the same information as two
independent poles. Their comparison tests implementation equivalence. A matched
recurrent controller changes the update rule while retaining other mechanisms;
any observed difference cannot be attributed uniquely to polarity. Ablation
intervals are exploratory and unadjusted. Cue-addressed memory, heuristic
uncertainty and resource allocation are limited engineered functions. They are
not validated psychological measures or evidence of subjective experience.

## License

The original [LICENSE](LICENSE), CC BY-NC 4.0, remains in effect. Historical
publication claims in the preserved README are not evidence that this revision
has been peer reviewed.
