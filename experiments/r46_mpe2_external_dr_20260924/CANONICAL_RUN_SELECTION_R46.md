# R46 canonical workflow-run selection

Date: 24 September 2026, America/Lima.

This decision is recorded after the R46 workflow launch and before the engineering smoke or any scientific development adjudication is available.

## Canonical scientific execution

- GitHub Actions run: `36029741503`
- event: `push`
- scientific head SHA: `bd0189bb1853dcc0715d19d2ffa0241bdf62dc82`
- workflow: `R46 external D/R causal transport on MPE2`

Only this run may adjudicate the R46 scientific result.

The branch workflow is subsequently changed to manual `workflow_dispatch` so documentation/result commits cannot open accidental duplicate scientific panels.

## Frozen source boundary

Scientific source for run 36029741503 contains:

- `PREREG_R46.md`;
- `r46_mpe2_dr.py`;
- `verify_r46.py`;
- the initial push-trigger workflow.

Later documentation or workflow-trigger-only commits do not alter the scientific runner/protocol executed by run 36029741503.

No R46 result, threshold, seed or endpoint was inspected before this canonical-run selection.
