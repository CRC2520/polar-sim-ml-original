# B1-D development results — derived reporting recovery

The original exporter failed because its immutable summary retained 42 of 96 diagnostic rows. This separate postclosure exporter reconstructed the redundant sidecar from 2,304 hash-verified archived events: 42 rows match the original byte-for-byte and 54 are derived reporting records. No new worlds, trials or seeds were opened. See `postprocessing/EXPORT_RECOVERY_INCIDENT.md`. The original failure and frozen artifacts remain unchanged.

Status: **B1D_BLOCKED**. All observations are development-only, nonconfirmatory and forbidden for reuse as final observations.

The implementation and configuration grid were content-frozen before tuning. All registered attempts, including failures, were retained. The technical calibration block opened only after a verified tuning/configuration freeze; descriptive identities were disclosed only after its technical summary closed.

| Pilot | Comparator | Cycle | Context | Route bypass | Bundles | Mean development loss | Failed attempts |
|---|---|---:|---|---|---:|---:|---:|
| P6 | C0 | 1 | ordinary | False | 32 | 0.03027344 | 0 |
| P6 | C1 | 1 | kappa1 | False | 32 | 0.03027344 | 0 |
| P6 | C1 | 1 | kappa1 | True | 32 | 0.03027344 | 0 |
| P6 | C1 | 1 | kappa2 | False | 32 | 0.03027344 | 0 |
| P6 | C1 | 1 | kappa2 | True | 32 | 0.03027344 | 0 |
| P6 | C1 | 1 | ordinary | False | 32 | 0.03027344 | 0 |
| P6 | C2 | 1 | ordinary | False | 32 | 0.03027344 | 0 |
| P6 | C2 | 2 | ordinary | False | 32 | 0.03027344 | 0 |
| P6 | C3 | 1 | ordinary | False | 32 | 0.03027344 | 0 |
| P6 | C4 | 1 | ordinary | False | 32 | 0.03027344 | 0 |
| P6 | C5 | 1 | ordinary | False | 32 | 0.03027344 | 0 |
| P6 | C6 | 1 | ordinary | False | 32 | 0.03027344 | 0 |
| P7 | C0 | 1 | ordinary | False | 32 | 0.06266276 | 0 |
| P7 | C1 | 1 | kappa1 | False | 32 | 0.06266276 | 0 |
| P7 | C1 | 1 | kappa1 | True | 32 | 0.06266276 | 0 |
| P7 | C1 | 1 | kappa2 | False | 32 | 0.06266276 | 0 |
| P7 | C1 | 1 | kappa2 | True | 32 | 0.06266276 | 0 |
| P7 | C1 | 1 | ordinary | False | 32 | 0.06266276 | 0 |
| P7 | C2 | 1 | ordinary | False | 32 | 0.06266276 | 0 |
| P7 | C2 | 2 | ordinary | False | 32 | 0.06266276 | 0 |
| P7 | C3 | 1 | ordinary | False | 32 | 0.06266276 | 0 |
| P7 | C4 | 1 | ordinary | False | 32 | 0.06266276 | 0 |
| P7 | C5 | 1 | ordinary | False | 32 | 0.06266276 | 0 |
| P7 | C6 | 1 | ordinary | False | 32 | 0.06266276 | 0 |

C2 cycles remain separate in this table; paired comparisons average both equally. C5 is descriptive. C6 is an exact re-encoding check, not an independent trained competitor.

Canonical precision planning requires **N = 915,501** independent bundles. The measured partial schedule projects approximately **614.27 serial hours** and **373.57 GB** of compressed traces. These are host-specific linear extrapolations, not an approved complete-run budget. P5 architecture execution remains missing.

Unresolved blockers:

- `B1-CONFLICT-P5-C4-REFILL-001`: Canonical P5 C4 refill actions conflict; neither labeled witness is silently selected.
- `B1-RESOURCE-CERTIFICATE-PENDING`: Smallest common memory/operation cap, worst-input latency and complete protocol budget remain uncertified.
- `B1D-ORIGINAL-EXPORT-PROVENANCE-INCIDENT`: Original diagnostic summary remains incomplete and original export failed; derived reporting recovery is documented separately without repairing the frozen execution.
- `B1E-RESOURCE-INFEASIBLE-CURRENT-DISK`: Projected partial-schedule compressed traces exceed currently available disk.

No final seeds were generated, no B1-E run was performed, and no historical results were modified by this runner. Source/sham diagnostics check imposed task laws; they do not supply independent empirical support or prove pairing utility.
