# Auditable derivation and scope of resource accounting

This is a finite-source certificate for the declared logical accounting model. A scalar/key slot is the recursive unit defined by the unchanged `b1.controllers.core.scalar_count`: one scalar leaf and one dictionary key each count one; lists contribute their entries. A rational number is one numeric slot. Python class code, allocator headers, interpreter state and object internals are not scalar slots. In particular, this is not a theorem about CPython byte allocations or hardware instructions. Those quantities are separately measured and explicitly limited in the certificate.

## Finite shape bounds

The frozen task contract has three cells, 32 epochs and two service jobs per cell and epoch. Controllers act only at epochs 0–31. The bound deliberately includes epoch-32 history sizes.

| Object | Maximum per cell | Source argument |
|---|---:|---|
| P5 pending delivery | 1 | `P5Task.step` appends at most once; the one-epoch delay is drained by `_prepare_boundary` |
| P6 reports | 32 | `P6Task.step` appends at most one assay record each epoch |
| P6 arriving reports | 1 | One assay per prior epoch and fixed delay one |
| P6 selector history | 32 | At most one selector event appended each epoch |
| P6 feedback | 2 | Routine capacity and service demand are two |
| P6 pending assay | 1 | One assay per epoch, drained at next boundary |
| P7 instances | 2 | Frozen `instances_per_cell` |
| P7 completion events | 2 | At most one repair completion and one deployment completion per boundary |
| P7 pending deployment | 1 | One deployment can begin each epoch, with one-epoch delay |
| P7 snapshots | 32 | At most one deployment/snapshot appended per epoch |
| P7 rollbacks | 0 | Frozen transition never appends a rollback |

`maximum_observations` spells out every field in these schemas and repeats the maximum-sized records. Simultaneous maxima need not be reachable; overcounting is intentional. Recursive scalar counts give complete three-cell input bounds S = 129 (P5), 3,312 (P6), 1,110 (P7). Altering schemas, task laws, horizon, or capacities invalidates this certificate.

## Legacy counter bound

Let n = 3; route payload p = 6, 10, 10 scalar slots respectively; route primitive bound r = 2, 3, 3. There are s = 3 graph slots for C0–C4, six for C5. C0 reconstruction and the acute bypass may call `_route` a second time, so all arms are bounded by 2nr even when they use less. The maximum returned action size A is 12 for P5 and 30 for P6/P7.

The exact upper-bound expression for the source's `_tick` metric is:

`legacy <= S + 2*n*r + s*p + n*(D + 3) + A`.

The final `3` accounts for recipient dispatch and two gain applications. D = 16 for P5; D = 240 for P6; D = 29 for P7. P6 D sums dispatch, all feedback iterations, typed-route handling, all 32 raw records and the `32*bit_length(32)` sorting charge even though several branches are mutually exclusive. P7 D includes both instance inspections and both possible route consumptions. No branch is selected using an outcome. The standalone P5 fixed controller executes its complete arithmetic rule and uses 75 counter units, below the 228 P5 shared bound; its smaller counter does not require padded instructions.

Thus the maximum legacy counts for C0–C4 are 228, 4,119 and 1,284; C5 counts are 246, 4,149 and 1,314. These are bounds on the existing instrumented counter, not a claim that the old counter includes every interpreter operation.

## Persistent controller storage

The frozen policy keeps `last_actions` and one raw snapshot, or two for C5. Its parameters are one eight-slot configuration entry plus two exact activation gains; fewer than 64 scalar/key slots suffices for every entry. Hence budgeted persistent state is bounded by `2 + A + retention*S + 64`. Its C0–C4 maxima are 207, 3,408 and 1,206. C5 maxima are 336, 6,720 and 2,316.

All remaining stored controller attributes (constants, clock, configuration, route-consumption list, previous decision report, scalar metadata) are explicitly counted in `scalar_count(controller.__dict__)`. The static additional allowance is 2,048 scalar/key slots: the configuration is below 64; constants plus clock below 256; two gain/scalar metadata and counters below 64; at most six 17-slot routed payload records plus six 8-slot consumption records fit below 256; the remaining report and metadata fit below 256. Their sum is below 896; 2,048 leaves conservative reserve, without allocating or executing padding. The monitor rejects both the pilot-specific total bound and the common 16,384 model-state allowance. `Limits`, numeric rationals and other scalar objects are counted as scalar objects, not expanded interpreter allocations.

## Expanded deterministic accounting

The new metric supplements the incomplete legacy counter with declared full-buffer charges. Its coefficients **define an explicit approximation**; they are not claimed to reproduce actual interpreter pass counts.

| Charge | Defined logical allowance | Source work represented |
|---|---:|---|
| Raw-buffer passes | `12*S` | Complete raw copy; observation scalar accounting; route inspection; per-recipient decision reads; record fallback and ordering; raw-history retention; monitoring reads |
| State passes | `4*M` | Budgeted state inspection, copying/replacement and verification of old/new state |
| Route passes | `16*P` | Constructing/copying typed payloads, slot metadata, routing and retained report payloads |
| Candidate bookkeeping | `32*R` | History candidate inspection, key access, sorting support and route eligibility bookkeeping, supplementing the legacy sorting charge |
| Fixed bookkeeping | `512` | Fixed-size configuration/dispatch/report operations |

Here M is the measured budgeted state, P is actual total route-payload size, and R is the total number of P6 report candidates (zero for P5/P7). The monitor charges `legacy + 12*S + 4*M + 16*P + 32*R + 512`. Replacing each quantity with its finite maximum gives a mathematical bound on this **defined approximation**, independent of outcomes and host timing. The common bound also substitutes `R <= 32*3` for pilots that never use those candidates, adding conservative headroom. C0–C4 expanded maxima are 6,476, 61,559 and 23,492; C5 maxima are 7,298, 75,317 and 28,442.

Temporary workspace uses the declared buffer-pass proxy `12*S + 8*M + 16*P + 512`. This reserves two sets of four state-buffer charges for simultaneously live old/new state. It is an auditable common logical opportunity measure; it is **not a proof of a worst-case allocation in bytes**. CPython peak allocation, shared-process RSS, CPU and wall latency are sampled separately and their claims remain quantified and limited. The fixed controller may use fewer buffers; no dummy copies are made to force actual equality.

The powers-of-two envelopes (16,384 scalar state, 65,536 legacy operations, 262,144 expanded logical units, 131,072 workspace proxy units) exceed these formulas, with C5 doubled. They were chosen before scientific P5 comparisons and without a score-dependent rule. Neither a smallest-cap claim nor an exact hardware-cost matching claim is made. Bound breaches fail before a result can be certified; changing a task dimension requires a new prospective specification.
