# R45 canonical workflow-run selection

Date: 23 September 2026, America/Lima.

This decision is recorded while both repaired R45 development runs are still in progress and before any repaired development result or competence-gate adjudication is available.

## Cause

The R45 workflow temporarily contained both `push` and `pull_request` triggers. Commit:

`e1052cf9e766d5e6f8a3fcfd6f40a472a093cea1`

therefore launched two executions of the same repaired scientific panel.

## Prospectively selected canonical run

**Canonical scientific run:**

- GitHub Actions run: `35924763722`
- event: `push`
- head SHA: `e1052cf9e766d5e6f8a3fcfd6f40a472a093cea1`

All R45 development and, if authorized, confirmatory adjudication must use this run only.

## Non-adjudicative duplicate

The following execution is an accidental infrastructure duplicate and must not be used to select, replace, rescue, average, pool, or overturn the canonical result:

- GitHub Actions run: `35924770044`
- event: `pull_request`
- head SHA: `e1052cf9e766d5e6f8a3fcfd6f40a472a093cea1`

If it completes, its artifacts are engineering duplicates only.

This rule applies even if the duplicate produces a more favorable result, finishes earlier, or differs because of nondeterministic execution details.

## Prevention

The branch workflow is subsequently changed to manual `workflow_dispatch` only so later documentation/result commits do not launch additional automatic R45 panels.

No scientific variable is changed by this canonical-run selection.
