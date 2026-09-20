# Reproducing and checking POLAR R8

The scientific source was frozen before confirmation at **`8823bda542f8a6c7c8e827a494d3acffb9b88a97`** (tree `cbdd351f7f5eca6d1c44eac9c49f4e811929d89d`). `r8_completion/FREEZE_R8.json` has SHA-256 `e7951144111cf38df6600157e51876dccae507d0c09450bc6e5145031f0ccc4f`. The later commit on `research/r8-corrections-20260920` adds audit, analysis, packaging utilities, historical pilot snapshots and this guide; it does **not** replace the frozen scientific source or retrospectively freeze the utilities. Use that branch's published post-run revision for the commands below. Do not recreate the freeze with `--create`.

The recorded runtime is Python **3.12.14**, NumPy **2.3.5**, SciPy **1.17.0**, OpenBLAS **0.3.30**, x86-64 Linux **6.18.44**, glibc **2.39**, OpenBLAS architecture **SkylakeX**. Both BLAS and OpenMP used one thread for final execution. The surrounding environment also had PyTorch **2.14.0+cpu**, scikit-learn **1.8.0** and Matplotlib **3.10.8**; these are not substitutes for the frozen R8 runtime contract. Exact numerical replay on another CPU/library stack is not guaranteed.

## Restore evidence, then check it

Run from the repository root. Name the checkout directory **`POLAR_reconciled`** when running the calibration audit below: the preserved factual-data reuse record includes that relative prefix, which the auditor resolves from the checkout's parent. Download all evidence ZIPs together with `R8_EVIDENCE_INDEX.json` and `R8_EVIDENCE_ARCHIVES.sha256`. Set `R8_DATA_ARCHIVES` to their absolute directory. Each ZIP is independently extractable; data member paths are relative to this repository root. Python sources, pilot source snapshots and `r8_completion/` come from Git, not the ZIPs. Extraction below preserves existing files; the subsequent audits detect any inconsistent pre-existing evidence. Repeated root-level `R8_ARCHIVE_*` entries describe individual ZIPs; consult the master index for the complete inventory.

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export R8_DATA_ARCHIVES=/absolute/path/to/downloaded/R8_EVIDENCE
(cd "$R8_DATA_ARCHIVES" && sha256sum -c R8_EVIDENCE_ARCHIVES.sha256)
for archive in "$R8_DATA_ARCHIVES"/*.zip; do
  unzip -n "$archive" -d .
done
python -m r8_completion.provenance
```

Use an unused output directory for each recheck; existing records are not overwritten. The population audit reconstructs metrics and replays representative first-seed cells. The integrated audit reconstructs saved metrics, checks every recorded SHA/NPZ CRC and replays the first checkpoint in all three domains. These are technical rechecks of the same evidence, not additional independent seeds.

```bash
export R8_RECHECK=/tmp/polar-r8-recheck-20260920
mkdir -p "$R8_RECHECK"
python r8_results/audit_population.py --phase final \
  --q r8_results/Q_final --a r8_results/A_final \
  --threshold r8_results/threshold_final \
  --selection r8_results/threshold_pilot/THRESHOLD_SELECTION.json \
  --calibration r8_results/population_development_v2 \
  --output "$R8_RECHECK/population" --replay --authorized-final-replay
python r8_results/audit_integrated.py \
  --input r8_results/integrated_final --report "$R8_RECHECK/integrated.json"
python r8_results/compile_confirmation.py --output "$R8_RECHECK/analysis"
python r8_results/crosscheck_integrated_inference.py \
  --audit "$R8_RECHECK/integrated.json" --compiled-dir "$R8_RECHECK/analysis" \
  --report "$R8_RECHECK/inference.json"
```

`compile_confirmation.py` reads the canonical restored `r8_results/Q_final` and `r8_results/integrated_final/shard_*/seed_*` paths. Compare its success vectors, six Holm-adjusted decisions and descriptive fields with `r8_results/analysis_final/`; audit metadata may contain different output paths. The separate `audit_population_integrity.py` writes `POPULATION_FINAL_INTEGRITY.json` into its input root and therefore should not be invoked over the already restored completed record.

## Regenerate the frozen final campaigns

These commands perform full scientific reruns and can take substantial time. They use the archived, frozen development calibration and threshold selection rather than retuning on final outcomes. They write only to new locations under `R8_REGEN`; retain the original evidence for comparison. Population seeds are **941001–941030** and integrated seeds **942001–942030**, selected by the frozen runners.

```bash
export R8_REGEN=/tmp/polar-r8-regeneration-20260920
python -m r8_completion.population --part Q --phase final --workers 4 \
  --calibration r8_results/population_development_v2/CALIBRATION_FROZEN.json \
  --output "$R8_REGEN/Q_final"
python -m r8_completion.population --part A --phase final --workers 4 \
  --output "$R8_REGEN/A_final"
python -m r8_completion.population --part threshold --phase final --workers 4 \
  --selection r8_results/threshold_pilot/THRESHOLD_SELECTION.json \
  --output "$R8_REGEN/threshold_final"
python -m r8_completion.integrated --phase final \
  --output "$R8_REGEN/integrated_final"
```

The integrated runner also accepts `--seeds` followed by explicit frozen seed IDs for selected shards; without it, all 30 run. Its audit accepts either direct `seed_*` directories or `shard_*/seed_*`. Audit reruns with the same commands above, substituting `R8_REGEN` inputs and fresh audit outputs. Seed-level traces, checkpoints and reconstructed metrics are the comparison targets; execution durations, aggregation layout and absolute-path metadata are not scientific invariants. Compiling rerun summaries with the current post-run compiler requires restoring those reruns to its documented canonical paths in a separate checkout, with integrated seeds inside `shard_*` directories.

Historical integrated pilots are preserved separately as `results/r8_integrated_pilot_v1/` and `results/r8_integrated_pilot_v2/`, with their exact `SOURCE_INTEGRATED.py` snapshots in Git. Those snapshots document the earlier implementations; running the final source on pilot seeds does not recreate an earlier version. Do not replace final `r8_completion/integrated.py` with a pilot snapshot or apply the audit's development-only source-drift exemption to final data. Calibration development v1/v2, pilots and adverse outcomes remain separate evidence; none is pooled with the 30 final seeds.

## Post-freeze Python files to publish with this guide

The following eleven files are utilities or provenance snapshots, not newly frozen scientific implementations:

```text
r8_results/audit_integrated.py
r8_results/audit_population.py
r8_results/audit_population_integrity.py
r8_results/audit_population_final/audit_population_source.py
r8_results/audit_population_pilot_verified/audit_population_source.py
r8_results/compile_confirmation.py
r8_results/crosscheck_integrated_inference.py
r8_results/package_r8_evidence.py
r8_results/test_package_r8_evidence.py
results/r8_integrated_pilot_v1/SOURCE_INTEGRATED.py
results/r8_integrated_pilot_v2/SOURCE_INTEGRATED.py
```

Reports and result JSON/NPZ remain in the external evidence delivery. Packaging instructions are included there as `r8_results/R8_PACKAGING.md`. This guide introduces no new experiment or result.
