# Study 3: external network evaluation

Decision: **no_confirmed_network_advantage**
Validity: True; separate harm flag: False.
5760 complete runs; 40 paired seeds; 12 factorial cells; 64 transitions per run.

|Controller|Mean external MSE|Mean cost|Post-switch MSE|
|---|---:|---:|---:|
|full|0.009743124|0.406559|0.015329031|
|no_K|0.009743124|0.406559|0.015329031|
|no_W|0.009743124|0.406559|0.015329031|
|no_WK|0.009743124|0.406559|0.015329031|
|retuned_zero|0.009743124|0.406559|0.015329031|
|rewired|0.009743124|0.406559|0.015329031|
|reverse|0.009743124|0.406559|0.015329031|
|generic_nonlinear|0.009743124|0.406559|0.015329031|
|generic_equivalent|0.009743124|0.406559|0.015329031|
|signed_intensity|0.009743124|0.406559|0.015329031|
|gradient|0.010632588|0.406432|0.018973978|
|zero_negative|0.229296706|0.000000|0.246909349|

## Five prespecified contrasts
Full minus comparator; negative favors full. One-sided 99% bounds implement alpha .05/5.

|Claim|Mean difference|95% interval|99% upper|Required upper below|Pass|
|---|---:|---|---:|---:|---|
|tension|+0.000000000|[+0.000000000, +0.000000000]|+0.000000000|-0.001000|False|
|network|+0.000000000|[+0.000000000, +0.000000000]|+0.000000000|-0.001000|False|
|topology_rewired|+0.000000000|[+0.000000000, +0.000000000]|+0.000000000|-0.000500|False|
|topology_reverse|+0.000000000|[+0.000000000, +0.000000000]|+0.000000000|-0.000500|False|
|feature|+0.000000000|[+0.000000000, +0.000000000]|+0.000000000|-0.000500|False|

## Gates and controls
{
  "gates": {
    "cost": true,
    "post_switch": true,
    "constraints": true
  },
  "equivalence": {
    "generic_equivalent": 0.0,
    "signed_intensity": 2.220446049250313e-16
  },
  "negative": {
    "mean": 0.219553582247734,
    "low95": 0.20907180113867624,
    "high95": 0.23044849193020037,
    "lower99_one_sided": 0.20697950181164948,
    "upper99_one_sided": 0.23261572135462438
  }
}

All subgroup contrasts are descriptive. No result changes the frozen Study 2 conclusion.
The tested configured templates are limited implementations, not the full layered architecture or evidence of consciousness.
