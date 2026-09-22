# R36 DEV1 amendment — metacognitive confidence anti-calibration

R36 DEV1 executed the full unified-agent panel successfully, but revealed a design defect before confirmatory seeds were opened.

DEV1 medians:
- D damage: +0.01206
- C recurrent-context gain: +0.00422
- R lesion damage: +0.00272
- memory reentry gain: +0.000571
- own-history transplant damage: +0.00295
- source balanced accuracy: 0.9662
- time-order accuracy: 1.000
- planning gain: +0.000240
- stability: 1.000

However the original confidence estimate was anti-calibrated:
- low-confidence error: 0.001240
- high-confidence error: 0.002475
- calibration gap (low - high): -0.001266

Therefore DEV1 cannot support the metacognition component of the unified conjunction.

Before any confirmatory seed 2007001–2007012 was opened, the confidence estimator was amended only as follows:
- every model stores its own factual one-step prediction errors together with the state where they occurred;
- expected current error is estimated from distance-weighted nearest historical states within the same learned context model;
- confidence is a monotonic inverse transform of this locally predicted factual error, with a sample-support factor;
- confidence continues to gate myopic versus two-step planning.

The environment, D/C/R lesions, persistent memory, source/time queries, donor transplant, planning comparator, seed split and confirmatory hypothesis were not changed.

DEV1 is retained as a development-design failure, not evidence against metacognition.
