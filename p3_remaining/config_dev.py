"""P3 development constants. No confirmatory seeds until a later freeze."""
DEV_SEEDS=(997001,997002,997003,997004)
STRESS_SEEDS=tuple(range(997100,997120))
GLOBAL_REQUIRED=3

# A — continual autonomy
A_RECURRENT_GAIN=.075
A_FINAL_REWARD=.90
A_RETENTION=.98

# B — autobiographical self / identity-specific history
B_PRED_GAIN=.05
B_TRANSPLANT_PRED_DAMAGE=.20
B_TRANSPLANT_REWARD_DAMAGE=.20

# C — endogenous goal-priority adaptation
C_ALIVE_MIN=.99
C_PRIORITY_ACC=.90
C_SHOCK_RECOVERY_GAIN=.65
C_UNSAFE_REDUCTION_MIN=.015

# D — individual -> population -> ecology
D_FULL_ALIVE_MIN=.99
D_RESOURCE_GAIN=.02
D_RESTRAINT_GAIN=.08
D_TRANSMISSION_SHIFT=.15
D_ECOLOGY_TRANSMISSION_EFFECT=.05
