"""Shared prospective constants. No values are selected from final outcomes."""
VERSION = 'R9.1'
PILOT_SEEDS = (950101, 950102, 950103)
FINAL_SEEDS = tuple(range(952001, 952081))
DOMAINS = ('ecology_train', 'ecology_delay9', 'inventory_transfer', 'thermal_transfer')
KINDS = ('full', 'dense', 'scalar', 'fixed')
VARIANTS = ('full', 'noCross', 'constant_gate', 'permuted_gate', 'scalar',
            'permuted_content', 'dense', 'fixed', 'block_viability',
            'block_resources', 'noMemory', 'frozenMemory', 'noGoalRevision',
            'observation_shock', 'dense_shock')
PROTOCOL = dict(version=VERSION, steps=320, training_episodes=8,
                calibration_episodes=2, fit_every_episodes=2,
                network_iterations=80, ridge=.01, l1=.01,
                history=16, horizons=(1, 4, 12), epsilon=.20,
                q_alpha=.10, q_gamma=.95, energy_floor=.20,
                calibration_quantile=.90, absolute_alive_floor=.80,
                native_epsilon=0., network_features=48, network_outputs=4,
                selection_candidates=('off', 'constant_quarter', 'constant_half', 'constant_one',
                                      'context_zero', 'context_small', 'context_large'),
                context_thresholds=(0., .001, .01),
                shock_steps=(80, 96), shock_magnitude=.15)
