"""Atomic development QA implementation of the prospective retention policy.

This component consumes records; it cannot generate seeds or run experiments.
The future confirmatory driver, authorization and custody are intentionally not
implemented in B1-D. Full records are spooled until a bundle is audited, so a
failure found at its end cannot cause earlier records to be lost.
"""
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from copy import deepcopy

from .policy import canonical, causal_record, full_trace_required, storage_gate
from .validation import record_key, validate_key


class RetainedBundle:
    def __init__(self, directory, identity, *, expected_records, reserve_bytes,
                 worst_case_bundle_bytes, expected_ids, identities=None):
        required = {"pilot", "bundle_index", "source_commit", "config_sha256",
                    "task_sha256", "seed_usage", "development_only", "confirmatory",
                    "reusable_as_final"}
        if not required.issubset(identity):
            raise ValueError("Incomplete Level A identity")
        if (identity["development_only"] is not True or identity["confirmatory"] is not False
                or identity["reusable_as_final"] is not False):
            raise ValueError("Only development retention QA is authorized")
        for name, size in (("source_commit", 40), ("config_sha256", 64), ("task_sha256", 64)):
            if not isinstance(identity[name], str) or re.fullmatch(f"[0-9a-f]{{{size}}}", identity[name]) is None:
                raise ValueError("Invalid source/config/task identity")
        if not isinstance(identity["seed_usage"], list) or not identity["seed_usage"]:
            raise ValueError("A nonempty seed or synthetic-fixture identity is required")
        for seed in identity["seed_usage"]:
            namespace = seed.get("namespace")
            if namespace not in ("PD-B1-D-v1.1", "PD-B1-D-v1.1-QA"):
                raise ValueError("Unapproved namespace; confirmatory streams are unavailable")
            if seed.get("synthetic_fixture") is True:
                if namespace != "PD-B1-D-v1.1-QA":
                    raise ValueError("Synthetic fixtures belong only to QA")
            elif type(seed.get("seed")) is not int or not 0 <= seed["seed"] < 2**64:
                raise ValueError("Missing explicit development seed")
        if type(expected_records) is not int or expected_records <= 0:
            raise ValueError("A fixed predeclared record count is required")
        expected_ids = [validate_key(key) for key in expected_ids]
        self.expected_ids = frozenset(expected_ids)
        if len(self.expected_ids) != expected_records or len(expected_ids) != expected_records:
            raise ValueError("Expected ID plan must be complete and duplicate-free")
        if any((key[1], key[2]) != (identity["pilot"], identity["bundle_index"]) for key in self.expected_ids):
            raise ValueError("Expected ID plan contains another bundle")
        self.plan_sha256 = hashlib.sha256(canonical(sorted(self.expected_ids))).hexdigest()
        self.identities = deepcopy(identities or {})
        self.seen = set()
        self.validation_failed = False
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if any(self.directory.iterdir()):
            raise ValueError("Retention output must be new and empty")
        storage_gate(self.directory, next_bundle_worst_case_bytes=worst_case_bundle_bytes,
                     reserve_bytes=reserve_bytes)
        self.identity = deepcopy(identity)
        self.expected = expected_records
        self.count = 0
        self.failed = False
        self.guardrails = {}
        self.has_diagnostic = False
        self.stage = Path(tempfile.mkdtemp(prefix="retention-stage-", dir=self.directory))
        self.streams = {}
        self.digests = {}
        self.level_counts = {}
        self.closed = False
        for level in ("B", "C_full", "C_diagnostic"):
            path = self.stage / f"{level}.jsonl.gz"
            self.streams[level] = gzip.GzipFile(filename=path, mode="wb", compresslevel=6, mtime=0)
            self.digests[level] = hashlib.sha256()
            self.level_counts[level] = 0

    def _append(self, level, row):
        raw = canonical(row) + b"\n"
        self.streams[level].write(raw)
        self.digests[level].update(raw)
        self.level_counts[level] += 1

    def append(self, record):
        if self.closed:
            raise ValueError("Already closed")
        try:
            # Preserve even a malformed submitted record when it is serializable.
            # Any rejection poisons completion; callers cannot catch it and then
            # publish a complete manifest from only the accepted prefix.
            self._append("C_full", record)
            if self.validation_failed or self.count >= self.expected:
                raise ValueError("Unexpected or late trace record")
            event_id = record_key(record, self.identities)
            if event_id not in self.expected_ids:
                raise ValueError("Unexpected complete retention event ID")
            if event_id in self.seen:
                raise ValueError("Duplicate complete retention event ID")
            if "namespace" in record:
                namespaces = {seed["namespace"] for seed in self.identity["seed_usage"]}
                if record["namespace"] not in namespaces:
                    raise ValueError("Record namespace differs from Level A identity")
                if record["source_commit"] != self.identity["source_commit"]:
                    raise ValueError("Record source differs from Level A identity")
            projected = causal_record(record)
            self._append("B", projected)
        except Exception:
            self.validation_failed = True
            raise
        event = record["event"]
        self.failed |= bool(event["controller_failure"])
        for key, value in event["guardrails"].items():
            self.guardrails[key] = self.guardrails.get(key, 0) + value
        if record.get("record_type") == "BUNDLE_SOURCE_DIAGNOSTIC":
            self._append("C_diagnostic", record)
            self.has_diagnostic = True
        self.count += 1
        self.seen.add(event_id)

    def finish(self, outcomes, invariant_audit):
        if self.closed:
            raise ValueError("Already closed")
        for stream in self.streams.values():
            stream.close()
        self.closed = True
        checks = invariant_audit.get("checks") if isinstance(invariant_audit, dict) else None
        audit_valid = (isinstance(checks, dict) and bool(checks)
                       and all(type(value) is bool for value in checks.values())
                       and invariant_audit.get("status") in ("PASS", "FAIL")
                       and (invariant_audit["status"] == "PASS") == all(checks.values()))
        if (self.validation_failed or self.count != self.expected or self.seen != self.expected_ids
                or not outcomes or not audit_valid):
            # Leave complete staged evidence in place for investigation. No
            # final manifest is published and no partial summary is released.
            raise ValueError("Incomplete bundle; staged evidence preserved")
        # A summary failure also requires full retention even when no individual
        # task event marked the controller failure.
        audit_pass = invariant_audit["status"] == "PASS"
        self.failed |= not audit_pass
        self.failed |= any(any(bool(row.get(field)) for field in
                               ("controller_failure", "controller_failures", "failures",
                                "resource_cap_binding", "failed_audit", "instrumentation_failure"))
                           for row in outcomes)
        self.failed |= any(any(row.get("guardrails", {}).values()) for row in outcomes)
        retain_full = full_trace_required(self.identity["bundle_index"], failed=self.failed,
                                         guardrails=self.guardrails)
        # Full C contains the entire B projection. Store it once; B remains
        # directly recoverable by causal_record(), without rerunning a model.
        levels = ["C_full"] if retain_full else ["B"]
        if self.has_diagnostic and not retain_full:
            levels.append("C_diagnostic")
        manifest = {"version": "1.1", "identity": self.identity, "records": self.count,
                    "expected_plan_sha256": self.plan_sha256,
                    "complete_event_ID_coverage_pass": True,
                    "full_trace_required": retain_full, "complete": audit_pass,
                    "invariant_audit_pass": audit_pass,
                    "level_B_evidence_location": "C_full.jsonl.gz" if retain_full else "B.jsonl.gz",
                    "guardrails": self.guardrails, "controller_failure": self.failed,
                    "artifacts": {}, "development_only": True,
                    "confirmatory": False, "reusable_as_final": False}
        for level in levels:
            path = self.stage / f"{level}.jsonl.gz"
            blob = path.read_bytes()
            assert hashlib.sha256(gzip.decompress(blob)).hexdigest() == self.digests[level].hexdigest()
            manifest["artifacts"][path.name] = {"sha256": hashlib.sha256(blob).hexdigest(),
                                               "bytes": len(blob), "records": self.level_counts[level],
                                               "raw_sha256": self.digests[level].hexdigest()}
        level_a = {**self.identity, "outcomes": outcomes, "guardrails": self.guardrails,
                   "invariant_audit": invariant_audit, "trace_identity": manifest["artifacts"],
                   "full_trace_required": retain_full}
        a_path = self.stage / "A.json.gz"
        a_path.write_bytes(gzip.compress(canonical(level_a) + b"\n", compresslevel=6, mtime=0))
        manifest["artifacts"][a_path.name] = {"sha256": hashlib.sha256(a_path.read_bytes()).hexdigest(),
                                             "bytes": a_path.stat().st_size, "records": 1}
        # Move artifacts first, publish the only completion marker last.
        for name in manifest["artifacts"]:
            (self.stage / name).replace(self.directory / name)
        manifest_name = "MANIFEST.json" if audit_pass else "FAILED_MANIFEST.json"
        (self.stage / manifest_name).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        (self.stage / manifest_name).replace(self.directory / manifest_name)
        shutil.rmtree(self.stage)
        if not audit_pass:
            raise ValueError("Failed invariant audit; complete evidence preserved")
        return manifest
