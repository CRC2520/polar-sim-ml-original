"""Reproducible read-only investigation plus separate versioned incident reports.

Only retained rows are passed through the extracted historical append function;
no original runner, controller, task or seed derivation is executed here.
"""
import ast
from datetime import datetime, timezone
import io
import re
from pathlib import Path
import tempfile

from .core import ROOT, FLAGS, canonical, digest, document, read_json, verify_v1


def investigate():
    preservation = verify_v1()
    directory = ROOT / "development_results/calibration"
    original_path = directory / "bundle_diagnostics.jsonl"
    derived_path = ROOT / "postprocessing/bundle_diagnostics.recovered.jsonl"
    original = original_path.read_bytes()
    derived = derived_path.read_bytes()
    rows = [read_json_line(line) for line in derived.splitlines()]
    original_rows = [read_json_line(line) for line in original.splitlines()]
    # Execute only append_row's AST, with a known serialization implementation.
    # The old source file itself and all experimental entry points stay untouched.
    tree = ast.parse((ROOT / "run.py").read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "append_row")
    namespace = {"canonical_bytes": canonical}
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "historical_append_row_extracted", "exec"), namespace)
    with tempfile.TemporaryDirectory() as temporary:
        path = Path(temporary) / "rows.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for row in rows:
                namespace["append_row"](stream, row)
        reproduced = path.read_bytes()
    ids = [(r["pilot"], r["bundle_index"]) for r in original_rows]
    checks = [
        ("incomplete_grouping_key", "not_reproduced", "The retained prefix contains unique (pilot,bundle) IDs; historical run_bundle_diagnostics appends once per complete bundle and does not group an in-memory table."),
        ("overwrite", "not_established", "The canonical calibration stream opens with mode x and append_row only writes to that stream. Later filesystem overwrite or partial copy is not covered by retained execution telemetry."),
        ("deduplication", "not_reproduced", "No deduplication step exists between run_bundle_diagnostics and its stream; 42 retained IDs are all unique."),
        ("cycle_omission", "not_consistent_with_prefix", "Diagnostic rows are task-law source/sham bundles, with cycle not applicable. Architecture C2 cycles1/2 are independently present in full archive coverage."),
        ("context_omission", "not_consistent_with_prefix", "Every retained diagnostic bundle contains both kappa1 and kappa2; 2,304 archived diagnostics cover all96bundles and both contexts."),
        ("route_bypass_omission", "not_consistent_with_prefix", "The diagnostic stream contains joint source/sham laws, not route-only trials. Intact and bypassed architecture schedules are independently complete in the authenticated archive."),
        ("premature_flush", "not_reproduced", "Historical append_row flushes after every row; replaying the96retained rows through its extracted unchanged function retains96rows exactly. There is no fsync durability record."),
        ("exception_handling", "not_established", "Exceptions inside calibration propagate after finally archives traces. Successful technical closure requires reaching code after the with/finally blocks. The historical closure omits a diagnostic-row completeness assertion."),
        ("partial_shard_ingestion", "not_reproduced", "The original exporter reads the42row JSONL directly; new authenticated replay consumes all83shards and82,176records. Partial redundant-summary copying remains possible but unproven."),
        ("ordering", "not_reproduced", "The42rows form the exact bundle32–45 prefix in pilot order P5,P6,P7. New exporter validates ID sets and rejects duplicates independently of input order."),
        ("filtered_records", "not_reproduced", "The original exporter parses every nonfiltered JSONL line before checking96expected IDs. The new archive parser inspects every record kind; none are filtered out before coverage checks."),
        ("file_finalization", "not_established", "The summary stream context manager exits before technical closure snapshots. A complete newline-terminated prefix is retained. No file-operation log or crash/fsync telemetry establishes why later rows are absent."),
    ]
    replay = read_json(ROOT / "v1_1/exporter/archive_replay/AUDIT.json")
    result = {"version": "1.1", "root_cause": "undetermined",
              "root_cause_reproduced": False, "recorded_utc": datetime.now(timezone.utc).isoformat(),
              "known_control_gap": "Historical technical calibration closure does not assert diagnostic summary count/IDs before snapshot; the old exporter correctly rejects the incomplete summary.",
              "original_expected_rows": 96, "original_retained_rows": len(original_rows),
              "original_unique_ids": len(set(ids)), "original_is_exact_42_row_prefix": original == b"".join(derived.splitlines(keepends=True)[:42]),
              "original_complete_newline_framing": original.endswith(b"\n"),
              "historical_append_only_replay": {"rows": len(rows), "byte_identical": reproduced == derived,
                 "scientific_task_executions": 0, "seeds_generated": 0,
                 "method": "extract only append_row AST; serialize retained96rows to temporary file; close and compare"},
              "investigated_hypotheses": [{"hypothesis": h, "finding": s, "evidence": e} for h, s, e in checks],
              "authenticated_archive_replay": {"events": replay["validated_records"], "diagnostic_events": replay["archive_diagnostic_events"],
                  "diagnostic_rows": replay["diagnostic_rows_exported"], "original_rows_equal": replay["original_rows_byte_identical"],
                  "prior_derived_rows_equal": replay["previously_derived_rows_byte_identical"]},
              "preservation": preservation, "original_export_status": "FAILED_RETAINED_UNCHANGED",
              "resolution_scope": "RESOLVED_FOR_FUTURE_PIPELINE", "historical_failure_retroactively_fixed": False,
              "remaining_limitations": ["No retained filesystem or process telemetry establishes the original truncation cause.",
                   "The original failure remains immutable provenance.", "Historical recovery and QA are not new experimental evidence.",
                   "Joint source/sham observations cannot be interpreted as selective Gamma lesions."],
              "source_sha256": {"b1/run.py": digest(ROOT / "run.py"),
                   "b1/evaluation/runner.py": digest(ROOT / "evaluation/runner.py"),
                   "b1/development_results/calibration/bundle_diagnostics.jsonl": digest(original_path),
                   "b1/postprocessing/bundle_diagnostics.recovered.jsonl": digest(derived_path),
                   "b1/CALIBRATION_TECHNICAL.json": digest(ROOT / "CALIBRATION_TECHNICAL.json")}, **FLAGS}
    if len(rows) != 96 or not result["original_is_exact_42_row_prefix"] or reproduced != derived:
        raise RuntimeError("Incident evidence unexpectedly differs")
    for hypothesis in result["investigated_hypotheses"]:
        hypothesis["evidence"] = readable_prose(hypothesis["evidence"])
    return result


def read_json_line(line):
    from b1.contracts.loader import strict_json
    return strict_json(line.decode())


def readable_prose(text):
    # Keep canonical identifiers P5/C2/kappa1 intact while separating counts.
    text = re.sub(r"([a-z]{2,})(\d{2,}|\d+(?:,\d{3})+)", r"\1 \2", text)
    return re.sub(r"(\d{2,}(?:,\d{3})*|\d+(?:,\d{3})+)([a-z]{2,})", r"\1 \2", text)


def main():
    result = investigate()
    (ROOT / "v1_1/EXPORTER_ROOT_CAUSE.json").write_bytes(document(result))
    lines = ["# Exporter v1.1: original incident investigation", "",
             "The original cause is **undetermined**. The historical exporter remains failed and unchanged. The new pipeline is separately verified as fail-closed.", "",
             "The42original rows form a complete newline-terminated prefix for bundles32–45. The archived83gzip shards contain82,176events, including2,304diagnostic events for all96pilot/bundle rows. New replay reproduces42original rows and54previous sidecar rows byte-for-byte.", "",
             "A known control gap is identified: technical closure checks architecture trial counts and guardrails but does not assert the96diagnostic IDs before snapshot. That omission explains why a technically closed snapshot can coexist with an incomplete summary; it does not establish the cause of the missing writes.", "",
             "| Hypothesis | Finding | Retained evidence |", "|---|---|---|"]
    lines += [f"| {r['hypothesis']} | {r['finding']} | {r['evidence']} |" for r in result["investigated_hypotheses"]]
    lines += ["", "Replaying only the unchanged historical append function over96existing records produces a byte-identical96row file. No model, controller, scientific seed or task is run by this investigation.", "",
              "The prospective exporter pins the old manifest, validates every shard hash and all complete event IDs (pilot, bundle, comparator, context, route, cycle, replicate and epoch), checks schema and publishes summary/audit/manifest in one atomic directory rename only after all checks pass. The manifest has a separate SHA256 file. Missing or duplicate records, absent key dimensions and malformed causal fields fail before any summary is published.", "",
              "The closure is **RESOLVED_FOR_FUTURE_PIPELINE**. The historical incident and undetermined cause remain recorded limitations. All artifacts are development-only, nonconfirmatory and unusable as B1-E final data.", ""]
    text = readable_prose("\n".join(lines)).replace("cycles1", "cycles 1")
    (ROOT / "v1_1/EXPORTER_ROOT_CAUSE.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
