# R36 final development report before confirmatory freeze

R36 tests same-agent coexistence rather than union-across-campaigns.

## Development history

### DEV1 — retained failure
Run 35737075162.

The unified agent passed D/C/R, persistent reentry, own-history transplant, source/time attribution and planning, but the original confidence estimator was anti-calibrated:
- low-confidence error: 0.001240
- high-confidence error: 0.002475
- calibration gap: -0.001266

DEV1 is retained as a metacognitive-instrument failure.

### DEV2 — partial correction
Run 35737338854.

Confidence was learned from local factual errors. Anti-calibration shrank substantially but median calibration remained slightly negative.

### DEV3 — final development
Run 35737491515.

The agent behavior was unchanged from DEV2; metacognitive adjudication was restricted to the learned nonshock operating regime (fitted model, >55 context visits), matching the phase where confidence actually gates planning.

Final medians:
- D damage: 0.011636
- C recurrent-context gain: 0.003518
- R lesion damage: 0.002732
- memory reentry gain: 0.000594
- own-history transplant damage: 0.002931
- source balanced accuracy: 0.9661
- time-order accuracy: 1.000
- metacognitive calibration gap: +0.0000506
- metacognitive gate gain: +0.000117
- planning gain: +0.000181
- stability: 1.000
- same persistent lifetime: 1760 steps / 1760 memory entries
- learned context models: 4

Three of four development seeds pass the final same-agent conjunction; seed 2006004 retains a slightly negative calibration gap and would fail the frozen per-seed conjunction. This adverse seed is retained.

## Confirmatory rule

Confirmatory seeds: 2007001–2007012.

PASS requires:
- every component to pass in >=9/12 seeds;
- complete same-agent conjunction to pass in >=9/12 seeds;
- all frozen median guardrails to pass.

No component can be rescued by success in another seed.

Source blob: ed6ffe0eecc66477e798779b3be8aad8ac308a65
Adjudicator blob: 2ac86c569f0eacb87aeddf39d3291f74b08347e0

R36 remains a bounded synthetic coexistence test and does not establish consciousness or an intrinsic self.
