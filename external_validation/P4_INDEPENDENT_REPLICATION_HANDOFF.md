# P4-EXT — independent replication handoff protocol

Status: **prospective external validation requirement**. This document is a handoff, not a completed replication.

## Purpose

Test whether the P2/P3 findings reproduce when the confirming party is independent of the POLAR development process.

Independence requires all of the following:
1. independent experiment operator/team;
2. independently written implementation of the task/controller contract;
3. no reuse of POLAR experiment code other than file-format schemas and equations;
4. new seed allocation chosen by the external team after implementation freeze;
5. preregistered analysis before outcome inspection.

## Minimum replication targets

Primary targets:
- P2 E1: generic learner equals constrained relational learner when both recover the same effective relation set, and both outperform random relations;
- P2 E2: online relation discovery changes after the hidden regime switch and improves post-switch reward vs frozen/no-relation controls;
- P2 E3: three lesion families show distinct primary/workspace/meta/source signatures;
- P2 E4: viability and bounded performance on logistic harvesting, RC thermal and queue-service equations;
- P3 A: persistent contextual memory improves early re-entry after long regime absence;
- P3 B: recipient own-history predicts its outcomes better than foreign identity memory;
- P3 C: deficit-derived priorities reduce recovery latency and unsafe exposure after resource shocks;
- P3 D: local adaptation and population transmission produce separable effects on shared-resource dynamics.

## Recommended external design

- >=20 independent seeds per target.
- Freeze implementation hash before final seeds.
- Publish exact dependency lock, source hash, seed list and machine-readable outputs.
- Do not tune thresholds using POLAR final seeds.
- Report every excluded/failed seed.
- Include at least one alternate implementation language or framework.
- Blind the analyst to controller labels until metrics are finalized when feasible.

## Replication decision

A target is independently replicated only if the external preregistered criterion is met.
A failed replication remains a failure even if the original implementation can reproduce itself.

No file committed to the POLAR repositories may be described as an independent replication merely because it was rewritten or rerun by the same development process.
