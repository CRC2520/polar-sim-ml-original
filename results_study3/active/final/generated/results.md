# Separately registered active-K panel

Decision: **no_confirmed_network_advantage**; harm flag: **False**.
2880 runs; 40 NEW final seeds; 12 cells; 6 conditions. Does not replace primary Study 3.

|Controller|MSE|Cost|Post-switch MSE|
|---|---:|---:|---:|
|full|0.011273843|0.389478|0.018367472|
|no_K|0.010378349|0.408052|0.016245697|
|retuned_zero|0.010378349|0.408052|0.016245697|
|rewired|0.011316721|0.389427|0.018405041|
|reverse|0.011320120|0.389238|0.018418171|
|generic_nonlinear|0.013642145|0.430955|0.019310803|

|Contrast|Difference|95% interval|99% upper|Threshold|Pass|
|---|---:|---|---:|---:|---|
|tension|+0.000895494|[+0.000803756, +0.000996343]|+0.001016456|-0.001000|False|
|network|+0.000895494|[+0.000803756, +0.000996343]|+0.001016456|-0.001000|False|
|topology_rewired|-0.000042879|[-0.000118489, +0.000029888]|+0.000041982|-0.000500|False|
|topology_reverse|-0.000046277|[-0.000134290, +0.000033271]|+0.000046845|-0.000500|False|
|feature|-0.002368302|[-0.002690542, -0.002046348]|-0.001988593|-0.000500|True|

{
  "gates": {
    "cost": true,
    "post_switch": true,
    "constraints": true
  },
  "cost": {
    "mean": -0.018573991156221797,
    "low95": -0.019521564929417274,
    "high95": -0.017655107199683082,
    "lower99_one_sided": -0.019698043040272787,
    "upper99_one_sided": -0.017486930809445363
  },
  "post_switch": {
    "mean": 0.002121774954281116,
    "low95": 0.0018337202068892443,
    "high95": 0.002467951267887857,
    "lower99_one_sided": 0.0017842573843554462,
    "upper99_one_sided": 0.002543532654307099
  },
  "max_action_difference": 0.3880944262027918
}
