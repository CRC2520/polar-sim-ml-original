# R11a pilot adjudication — benchmark infeasibility discovered before confirmation

Pilot run: GitHub Actions 35538031010, seeds 971001–971004.
Artifact SHA-256: d6c3a71d3a1b8f418ae41020a64b3b0cd5be0052ed666c8ee07987b7304e1ffc.

All eight allowed adapter configurations failed the pilot validity requirement because
the inherited R10 cyclic_buffer benchmark yielded alive fraction 0.0 after the
declared 64-step adaptation window for every configuration. This is not treated as
evidence selecting configuration H; all scores were invalid (-1e9).

A direct feasibility audit of the frozen R10 environment shows the issue is upstream
of R11 tuning: for constant actions 0/1/2/3 over 50 deterministic tapes, mean alive
fractions are approximately 0.099/0.097/0.093/0.080 and none approaches the frozen
0.80 criterion. The repair_queue constants likewise peak near 0.52 mean alive fraction.
Therefore the R10 E4 absolute viability gate cannot serve as a valid discriminator
without first establishing that the task admits such performance.

No R11 confirmatory seed has been run. R11a is abandoned as a development pilot.
R11b keeps all substantive hypotheses but replaces only the invalid transfer benchmark
with new independently coded families that must pass a pre-execution feasibility audit.
New pilot seeds are used. The original R10 E4 FAIL and R11a pilot remain unchanged.
