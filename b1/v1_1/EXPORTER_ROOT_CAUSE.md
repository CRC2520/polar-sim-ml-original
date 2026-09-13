# Exporter v1.1: original incident investigation

The original cause is **undetermined**. The historical exporter remains failed and unchanged. The new pipeline is separately verified as fail-closed.

The 42 original rows form a complete newline-terminated prefix for bundles 32–45. The archived 83 gzip shards contain 82,176 events, including 2,304 diagnostic events for all 96 pilot/bundle rows. New replay reproduces 42 original rows and 54 previous sidecar rows byte-for-byte.

A known control gap is identified: technical closure checks architecture trial counts and guardrails but does not assert the 96 diagnostic IDs before snapshot. That omission explains why a technically closed snapshot can coexist with an incomplete summary; it does not establish the cause of the missing writes.

| Hypothesis | Finding | Retained evidence |
|---|---|---|
| incomplete_grouping_key | not_reproduced | The retained prefix contains unique (pilot,bundle) IDs; historical run_bundle_diagnostics appends once per complete bundle and does not group an in-memory table. |
| overwrite | not_established | The canonical calibration stream opens with mode x and append_row only writes to that stream. Later filesystem overwrite or partial copy is not covered by retained execution telemetry. |
| deduplication | not_reproduced | No deduplication step exists between run_bundle_diagnostics and its stream; 42 retained IDs are all unique. |
| cycle_omission | not_consistent_with_prefix | Diagnostic rows are task-law source/sham bundles, with cycle not applicable. Architecture C2 cycles 1/2 are independently present in full archive coverage. |
| context_omission | not_consistent_with_prefix | Every retained diagnostic bundle contains both kappa1 and kappa2; 2,304 archived diagnostics cover all 96 bundles and both contexts. |
| route_bypass_omission | not_consistent_with_prefix | The diagnostic stream contains joint source/sham laws, not route-only trials. Intact and bypassed architecture schedules are independently complete in the authenticated archive. |
| premature_flush | not_reproduced | Historical append_row flushes after every row; replaying the 96 retained rows through its extracted unchanged function retains 96 rows exactly. There is no fsync durability record. |
| exception_handling | not_established | Exceptions inside calibration propagate after finally archives traces. Successful technical closure requires reaching code after the with/finally blocks. The historical closure omits a diagnostic-row completeness assertion. |
| partial_shard_ingestion | not_reproduced | The original exporter reads the 42 row JSONL directly; new authenticated replay consumes all 83 shards and 82,176 records. Partial redundant-summary copying remains possible but unproven. |
| ordering | not_reproduced | The 42 rows form the exact bundle 32–45 prefix in pilot order P5,P6,P7. New exporter validates ID sets and rejects duplicates independently of input order. |
| filtered_records | not_reproduced | The original exporter parses every nonfiltered JSONL line before checking 96 expected IDs. The new archive parser inspects every record kind; none are filtered out before coverage checks. |
| file_finalization | not_established | The summary stream context manager exits before technical closure snapshots. A complete newline-terminated prefix is retained. No file-operation log or crash/fsync telemetry establishes why later rows are absent. |

Replaying only the unchanged historical append function over 96 existing records produces a byte-identical 96 row file. No model, controller, scientific seed or task is run by this investigation.

The prospective exporter pins the old manifest, validates every shard hash and all complete event IDs (pilot, bundle, comparator, context, route, cycle, replicate and epoch), checks schema and publishes summary/audit/manifest in one atomic directory rename only after all checks pass. The manifest has a separate SHA256 file. Missing or duplicate records, absent key dimensions and malformed causal fields fail before any summary is published.

The closure is **RESOLVED_FOR_FUTURE_PIPELINE**. The historical incident and undetermined cause remain recorded limitations. All artifacts are development-only, nonconfirmatory and unusable as B1-E final data.
