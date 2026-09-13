# Fail-closed exporter v1.1

This version authenticates retained input, validates complete event IDs and schema,
then publishes the complete audit, summary and manifest in an atomic directory
rename. It never overwrites an existing export. The historical exporter, its
failed status, technical closure and all original traces remain unchanged.

Run from the code repository:

```bash
python -B -m unittest b1.v1_1.exporter.test_exporter -v
python -B -m b1.v1_1.exporter.core verify-v1
python -B -m b1.v1_1.exporter.core archive --output /tmp/new-archive-export
python -B -m b1.v1_1.exporter.qa --output /tmp/new-qa-run
```

The test suite uses temporary directories and preserves every tracked artifact.
The archive command starts no tasks or seeds. It checks the pinned B1-D v1
manifest and all 480 artifact hashes, all 83 trace shards and 82,176 complete
event IDs, then reconstructs 96 diagnostic rows. The 42 original rows and 54
previously recovered rows must match byte-for-byte. A separate SHA256 file
authenticates the output manifest.

The historical ID is `(record_kind, pilot, bundle_index, comparator, context,
route_status, cycle, replicate_id, epoch)`. The historical adapter resolves masked
comparator IDs through the frozen identity map. `cycle=0` is explicitly not
applicable for diagnostic and physical witness records. Joint source/sham
diagnostics use `joint_source_not_selective`; they never become selective Gamma
lesions. Architecture routes distinguish intact and bypassed records.

The normalized development API is
`export_development(directory, plan_path, plan_sha256, output, *, index_sha256)`.
Its expectation plan contains a complete `expected_ids` list, a declared
development namespace, and `fixed_before_execution=true`. The caller must freeze
that plan before collecting events. The archive-index hash is supplied separately
after collection. Every record must match the plan namespace and exact schema.
The supported namespaces are `PD-B1-D-v1.1` and `PD-B1-D-v1.1-QA`; no final-data
execution or seed generator is provided. `export_qa` additionally requires the QA
namespace. The namespace itself does not establish a scientific freeze; the
surrounding run must retain its pre-execution plan commitment.

The retained QA fixture declares its ID plan before task execution and checks the
plan hash afterward. It runs 16 P6/P7 QA episodes, including both C2 cycles, C6,
both contexts and intact/bypassed routes, producing 1,536 events. Selected v1
policies are unchanged. C6 event hashes and rational loss equal C1 exactly. This
pipeline check is development-only and contributes no experimental evidence.

The original truncation cause remains undetermined. Investigation covers all 12
requested hypotheses and reproduces the unchanged append function over retained
records without reproducing truncation. A definite control gap is the missing
diagnostic completeness check at historical technical closure. Future reporting
now fails closed, while the original incident remains recorded provenance.
