# B1 Development / Calibration

This isolated implementation executes **B1-D: B1 Development / Calibration** and prepares **B1-E: B1 Confirmatory Evaluation**. It provides no B1-E execution command or final-seed generator. All development records are marked `development_only=true`, `confirmatory=false`, and `reusable_as_final=false`.

Read [B1D_IMPLEMENTATION_SPEC.md](B1D_IMPLEMENTATION_SPEC.md) for the task laws, comparator family, exact C6 control, calibration separation and interpretation limits. Generated `B1D_RESULTS.json`, `B1D_RESULTS.md`, `B1D_AUDIT.json`, and `B1D_MANIFEST.json` report the actual execution and readiness; implementation availability alone does not mean a stage completed.

## Sources and scope

- Code source: `c958049743f989b5c1f8866018ea48b60bae9ed7` in `CRC2520/polar-sim-ml-original`.
- Immutable B0 publication: `c6ba1e836b6c96636ca29be7913c9c5ab9facc62` in `CRC2520/POLAR_MODEL_CRC`.
- B0 contracts are preserved in `contracts/frozen_b0/` and checked against their frozen hashes.
- Historical Study 1–3 and P0–P2 remain outside this implementation. P1 retains `engineering_verified_with_partial_empirical_support`; P2 retains `bounded_specific_capabilities`.

P5 is a physical-ceiling negative control; P6 is an information-redundancy and expected-ceiling negative control. Neither is eligible for positive utility or pairing superiority in B1-v1. P7 is a prospective coordination candidate, without a presumed advantage. A deterministic synthetic mechanism diagnostic checks the imposed law and its instrument; it does not establish consciousness or a natural polarity.

## Run and inspect

Use Python 3.10 or later and PyYAML, from the repository root. The dedicated stage interface is `python -m b1.run`. Stage gates enforce their dependencies; no final-evaluation stage is provided.

```bash
python -m b1.run prepare
python -m b1.run instrument
python -m b1.run tuning
python -m b1.run calibration
python -m b1.run export
```

`prepare` accepts an optional `--code-commit SHA` for explicit implementation provenance. It freezes implementation and configuration inputs before tuning. `instrument` runs the test suite and deterministic diagnostics. Tuning requires a passing gate and unchanged input hashes; calibration requires verified tuning closure. These stages write under `b1/` and `b1/development_results/`. A stopped or blocked stage must be reported, not bypassed by proceeding manually.

After export, run smoke verification into a new output directory:

```bash
python -m b1.run smoke --output /tmp/polar-b1-smoke
```

Only `smoke` accepts `--output`; use a directory that does not already exist. Consult `python -m b1.run --help` for the installed interface.

Run the independent instrument and validation fixtures without launching an architecture benchmark or final study:

```bash
python -m unittest discover -s b1/tests -v
```

Inspect the read-only provenance validator's supported options:

```bash
python -m b1.validate_b1 --help
```

The full validator requires Git history and the original source tree. Its explicit sparse mode checks only available local material and cannot establish remote or complete historical integrity. Generated outputs must retain their original hashes, seed-use ledgers and failed configurations. Do not edit a freeze to conceal a failed gate.

## Implementation map

| Directory | Responsibility |
|---|---|
| `contracts/` | Canonical B0 imports and strict hash verification |
| `tasks/` | Separate P5/P6/P7 state machines and event instrumentation |
| `controllers/` | C0–C5, full exact C6 conjugacy, resource limits and invariants |
| `interventions/` | Source/sham mechanism diagnostics |
| `metrics/` | Frozen primary loss and guardrails |
| `evaluation/` | Development episodes, trace export, precision plan and decision interpreter |
| `seeds/` | Development-only namespace and gated calibration access |
| `tests/` | Instrument, corruption, decision and preservation fixtures |
| `development_results/` | Actual retained development traces and stage records |
| `manifests/` | Implementation/configuration provenance and historical inventory |

The tunable comparators use eight predeclared finite configurations with equal declared development opportunities; this is configuration selection, not neural training. C4 is fixed and does not pretend to undergo eight searches. C2 retains and averages both fixed cycles. C0 and C3 can reconstruct ordinary useful relationships. The C1 route may be redundant, so a null acute-lesion effect is legitimate. C5 deliberately doubles capacity and is descriptive. C6 conjugates the full realization with exact rational coordinate arithmetic and zero tolerance; it is not a speed comparison.

## Explicit blockers and safeguards

`B1-CONFLICT-P5-C4-REFILL-001` records differing frozen P5 C4 action recipes in the JSON and protocol. The affected C4 constructor stops. Both recipes can be examined as named physical-ceiling fixtures, but neither is silently selected as the authoritative comparator. P5 architecture comparisons remain blocked until a prospective resolution is documented without rewriting B0.

The memory/useful-operation caps are provisional engineering allocations. Their equality is not a certificate of the smallest adequate common compute budget, and a missing latency/resource certification keeps the matching audit unpassed. A Γ intervention that also changes information, feasibility, memory, compute or the action repertoire is labeled `joint_manipulation` and cannot support a Γ-specific verdict.

Tuning is restricted to development bundles 0–31. Calibration bundles 32–63 require a verified tuning closure and are reserved for technical checks with comparator identities masked. Final seeds are unavailable. The frozen 27-endpoint precision rule determines a future sample requirement; infeasibility keeps confirmatory readiness false rather than authorizing a smaller convenient sample.
