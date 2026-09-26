# R10 E5b — post-confirmatory validity-controlled replication

This corrective experiment is frozen AFTER the original E5 result was observed.
It never overwrites E5. Original E5 remains FAIL 0/12.

Reason: the original attribution environment produced median alive fraction near 0.05,
making self/source/time probe failures hard to distinguish from trajectory collapse.

New untouched seeds: 964001–964012.

The R9 full agent is trained exactly as in R10 on ecology_train, then evaluated with
frozen learned parameters in a new 640-step periodic attribution environment.
Online workspace state may update as in R9.

Validity gate per seed:
- alive fraction >= 0.80
- at least 40 second-half observations in each of self/world binary classes
- all three source classes represented in the second half

If the validity gate fails, the seed cannot support precursor interpretation.

The same five base endpoints are retained:
- self/world balanced accuracy >= .70
- three-way source balanced accuracy >= .55
- time-since-world-event balanced accuracy >= .45
- uncertainty/error AUC >= .65
- counterfactual best-action accuracy >= .45

The same three lesion endpoints are retained:
- noMemory worsens delayed-memory forecast MAE >= .02
- permuted_content lowers return >= .01
- noCross lowers return >= .01
- each lesion alive-fraction change <= .10

A valid seed passes only if validity + base + lesion conjunctions all pass.
Global PASS requires >=9/12.
No threshold from original E5 is relaxed. This is a validity-controlled new experiment,
not a rescue or reinterpretation of the original E5.
