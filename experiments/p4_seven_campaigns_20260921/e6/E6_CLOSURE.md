# E6 closure record — replication status

Date: 2026-09-21

## E6a — Internal clean-room / cross-implementation robustness

**CLOSED — PASS**

Same-program but separately implemented controls reproduced the central mechanisms with materially different implementations:

- alternative LSTM reproduced reactivation-dependent updating in 4/4 tested seeds;
- independent MLP + Jacobian/SVD polarity analysis recovered the hidden causal axes with mean/min alignment >0.9998 in 4/4;
- independent PLS quotient implementation yielded approximately d4 MSE 0.0056 versus d3 MSE 0.0719 and cross-agent alignment approximately 0.998 in 4/4.

This closes dependence on one exact implementation for the tested E1–E3 mechanisms.

## E6b — Independent scientific replication

**HANDOFF COMPLETE / EXTERNAL EVIDENCE OPEN**

The scientific requirement cannot be self-certified by the originating program. Closure requires a separate researcher/team that:

- writes its own implementation from the frozen specification;
- does not reuse experiment executables from this repository;
- freezes criteria before inspecting its final outcomes;
- uses new seeds and preferably independently implemented generators;
- reports adverse/null outcomes without threshold changes;
- publishes environment/dependency provenance.

The replication package and frozen target are documented in `INDEPENDENT_REPLICATION_HANDOFF.md`.

## Status semantics

E6 is operationally closed as a research handoff:
- internal implementation-robustness question: closed PASS;
- external independent-evidence requirement: formally handed off and still scientifically open.

The manuscript must not state that an independent replication has already occurred.
