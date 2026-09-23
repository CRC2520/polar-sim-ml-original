# R42 development report and confirmatory freeze

Development run 35825673856 used only seeds 2136001–2136004 on sparse_mixed.

All four development cases passed the preregistered native conjunction and instrument-validity checks.

Medians:
- normalized score: 0.884917
- D dz: 1.365986
- C dz: 0.414367
- R dz: 0.858714
- stability: 1.000000
- native/rotated action agreement: 1.000000
- normalized score gap: 0.000000

No scientific source or threshold adjustment is made after development.

The confirmatory source is frozen at Git blob:
9936e8364f212563d6d103fe1cdb285b71541ec3

Confirmatory seeds 2137001–2137016 are opened only after this freeze.
Held-out families:
- ring_signed
- lowrank_skew

PASS requires >=12/16 native conjunction passes in EACH family, median native/rotated action agreement >=0.98 in EACH family, and median normalized-score gap <=0.02 in EACH family.

This remains same-program evidence and cannot close E6b.
