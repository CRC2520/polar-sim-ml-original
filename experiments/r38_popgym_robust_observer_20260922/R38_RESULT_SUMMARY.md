# R38 development result — robust observers improve POPGym but do not clear the frozen external gate

Frozen resolution:

`R38_DEVELOPMENT_FAIL_NO_CONFIRM`

Run: **35769153215**  
Artifact: **10713681513**  
Artifact digest: `sha256:66ab40632165ae39813b85a179516b22428610d52005c58c3c2368f80e64fdb7`  
Source SHA-256: `67ceab61814575c26ba8fd1fff9f3a47c7cae0df9cd925639351046ff3892144`  
Development seeds: **2046001–2046008**  
Confirmatory seeds **2047001–2047012 remain unopened**.

## Preregistered selection result

No observer variant is eligible because every variant misses the unchanged R37 absolute median CORE threshold of 0.70.

| Variant | CORE score | CORE−NO_C | min env CORE | iso gap | seed guard | Eligible |
|---|---:|---:|---:|---:|---:|---|
| KF_ADAPT | 0.64367 | +0.50936 | 0.19979 | 0 | 8/8 | No |
| KF_R01 | 0.63680 | +0.50291 | 0.19052 | 0 | 8/8 | No |
| KF_R04 | 0.64508 | +0.51579 | 0.21021 | 0 | 8/8 | No |
| KF_R09 | **0.65846** | **+0.52425** | **0.22271** | 0 | 8/8 | No |

All variants satisfy the recurrence-effect, isomorphic-equivalence and oracle criteria and every development seed satisfies the looser per-seed guard. The failure is specifically the preregistered absolute external-control median criterion.

## Interpretation

R38 improves on the R37 finite-difference observer but does not close external noisy-observation robustness. The result remains scientifically useful:

- recurrent state produces a large external benefit over current-only control;
- a generic coordinate-isomorphic implementation remains exactly equivalent;
- stronger generic state observers improve robustness;
- none is sufficiently strong to justify opening confirmation under the frozen 0.70 criterion.

No variant is selected and confirmation is **not authorized**. Seeds 2047001–2047012 remain unopened.

After two external development campaigns (R37/R38), further same-program tuning on this POPGym panel would risk converting external validation into benchmark-specific optimization. The next priority is independently developed memory-capable baselines / independent replication, not another observer sweep.

POLAR Core v1.1 remains D+C+R with A conditional. R37 confirmatory seeds 2037001–2037012 and R36-R2 confirmatory seeds 2027001–2027012 also remain unopened. E6b and E7 remain OPEN.
