"""R11 constants fixed before pilot except explicitly enumerated pilot choices."""
PILOT_SEEDS=tuple(range(971001,971005))
FINAL_SEEDS=tuple(range(973001,973013))
GLOBAL_REQUIRED=9
ADAPT_STEPS=64
E2=dict(reward_margin=.02, alive_margin=-.01, ecology_guard=-.01)
E3=dict(balanced_accuracy=.85, mse_gain=.02, perm_damage=.02)
E4=dict(alive=.80, reward=.35, reward_improvement=.05, alive_improvement=.10)
E5=dict(alive=.80, self_world=.70, source=.55, time=.45, auc=.65, counterfactual=.45,
        memory_drop=.01, content_drop=.01, cross_drop=.01, max_alive_damage=.10)
PILOT_GRID=dict(forgetting=(.97,.985,.995), l1=(0.,.0001,.0005), min_samples=(3,5,8),
                reliability_margin=(0.,.002,.005), adapter_weight=(.40,.60,.75),
                risk_floor=(.20,.25,.30))
