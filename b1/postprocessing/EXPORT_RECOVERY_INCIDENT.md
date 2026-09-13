# Closed calibration export incident

The original `python -B -m b1.run export` failed with exit code 2: `StageError: Missing or duplicated actual calibration bundle diagnostics`. Its failure is preserved; the original exporter was not repaired or rerun.

The immutable summary contains 42 rows through bundle 45. All 83 compressed shards were hash checked and their record/byte counts verified. They contain 2304 diagnostic events for all 96 pilot/bundle records. The cause of the incomplete summary stream is unknown.

A separate postclosure reporting module verifies every archived diagnostic event against the retained deterministic fixture before using its metadata. It rebuilds the redundant JSONL representation, proves byte-for-byte equality of every one of the original 42 rows, and derives the other 54 records. The sidecar retains original execution metadata: `executed_new_clone_worlds=true` refers to the already archived calibration executions. This postprocessor runs zero worlds and opens zero seeds.

All original source, raw data, snapshots, and technical-closure bytes remain unchanged. New reporting Python files are outside the experimental freeze; consequently the original runner inventory gate is not satisfied by this additive tree. The independent reporting verifier checks the frozen inventory's bytes and separately identifies the reporting additions. It supplies no permission to resume or rerun any experimental stage.

The standalone export uses the frozen decision evaluator, precision calculation, endpoints, thresholds and descriptive formulas. Only the identified diagnostics input sidecar and reporting provenance differ. The result remains B1D_BLOCKED: P5 architecture comparison and resource certification are unresolved. Recovered diagnostics add no independent empirical evidence and allow no O3/O4 promotion or B1-E run.

See `EXPORT_RECOVERY_INCIDENT.json` for hashes, original tool-output provenance, derived-row keys, and the exact checks.
