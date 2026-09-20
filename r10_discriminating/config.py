"""Frozen R10 discriminating experiment constants."""
PILOT_SEEDS = tuple(range(961001, 961004))
FINAL_SEEDS = tuple(range(963001, 963013))
STEPS = 320
ATTR_STEPS = 640
CSD_BLOCKS = ("ERR","GOAL","HIST","MEM","PRED")
CSD_THRESHOLDS = dict(action_agreement=.90, memory_mae=.08, uncertainty_mae=.05,
                      gate_agreement=.90, return_loss=.03, alive_loss=.02)
E2_REWARD_MARGIN = .01
E2_ALIVE_MARGIN = -.005
E3_GATE_ACCURACY = .85
E3_MSE_GAIN = .02
E3_PERM_DAMAGE = .02
E4_ALIVE = .80
E4_RETURN = .35
E4_RETURN_NONINFERIOR = -.01
E4_ALIVE_NONINFERIOR = -.02
E5 = dict(self_world=.70, source=.55, time=.45, auc=.65, cf_action=.45,
          memory_damage=.02, content_return=.01, cross_return=.01,
          max_alive_damage=.10)
GLOBAL_REQUIRED = 9
