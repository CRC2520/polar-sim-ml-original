# R47 canonical workflow-run selection

Date: 24 September 2026, America/Lima.

This decision is recorded after R47 launch and before engineering-smoke or scientific-development adjudication is inspected.

## Canonical scientific execution

- GitHub Actions run: `36043862460`
- event: `push`
- scientific head SHA: `a0ed7720f0b0a03f39e6d6ff10720f30e7c3b624`
- workflow: `R47 external integrated D+C+R on POPGym Concentration`

Only this run may adjudicate the R47 scientific result.

## Frozen source boundary

Run 36043862460 executes the source containing:
- `PREREG_R47.md`;
- `r47_concentration_dcr.py`;
- `verify_r47.py`;
- the gated workflow.

Later documentation, result-summary or workflow-trigger-only commits cannot replace, rescue, pool with or overturn this canonical execution.

No R47 scientific result, development metric or confirmatory seed has been inspected before this canonical-run selection.
