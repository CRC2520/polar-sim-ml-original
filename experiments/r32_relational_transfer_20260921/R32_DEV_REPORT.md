# R32 formal development report

Development was restricted to seeds **1606001–1606004** and the two declared development topology families, **sparse directed** and **modular directed**. Confirmatory seeds 1607001–1607012 were not executed before the freeze.

GitHub Actions development run: **35685412213**  
Execution head: `b8fe6a9d78a04e2c264b306dcdae356b8d71d899`  
Artifact ID: **10676312863**  
Artifact digest: `sha256:d0cc16225783aba23d182bd4df28e53242f729399f02743d83c753ef085374e2`

Observed development medians:

- relation lesion damage under relational shift: **0.0018161684554114034**
- diagonal null damage: **0.0007733368489708735**
- gain-shift null damage: **0.0008905539483756807**
- static-relational null damage: **0.0007659052090555394**
- specificity margin: **0.0006814881668670692**
- CORE/GENERIC post-shift cost ratio: **0.9982933907994002**
- CORE/FACTORIZED ratio: **0.30015453468438524**
- CORE/FROZEN ratio: **0.10528249876437823**
- relation-recovery gain vs NO_R: **0.4362825150006488**
- wrong-relation intervention damage: **0.0027899223069468852**
- correct-relation rescue: **0.0030175460657254505**
- CORE stable fraction: **1.0**

These values were used only to set the frozen margins in `R32_CONFIRM_FREEZE.json`. The strict held-out topology classes **random_dag** and **skew** were not used for threshold tuning.

The confirmatory adjudication must retain all failures; no threshold can be relaxed after opening confirmatory seeds.
