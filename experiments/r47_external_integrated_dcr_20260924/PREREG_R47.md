# R47 — external integrated D+C+R on one POPGym Concentration agent

Status: prospective before R47 scientific development seeds are opened.
Date: 24 September 2026, America/Lima.

## Starting boundary

R45 supports external recurrent-state contract C on historical POPGym StatelessCartPole.
R46 supports external functional D and R binding/reidentification on MPE2, but D/R and C are still distributed across different external tasks.

R47 tests the missing conjunction:

> Do differentiated causal identities (D), persistent historical state (C), and online relation binding/reidentification (R) coexist and perform separable causal work in one agent solving one unmodified external task?

POLAR Core v1.1 remains unchanged before R47:

[
\mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]

## Exact external benchmark

Repository: `proroklab/popgym`.

Exact historical paper-source commit:

`e397e5eac9965f9963d18c9f455cd1983bca14fb`.

Environment:

`popgym.ConcentrationHard`.

The external source is installed unchanged. R47 does not alter:
- the deck;
- shuffle;
- observation or action spaces;
- rewards;
- episode length;
- termination rules.

Historical environment contract:
- one deck, 52 positions;
- rank matching, 13 distinct visible identities;
- 52 discrete actions;
- face-down marker distinct from all rank identities;
- native episode length 104 actions;
- `obs_requires_prev_action=True`.

The controller therefore retains the immediately previous action in every condition. R47's C lesion removes **persistent cross-pair card/location memory**, not the benchmark-declared one-step action context.

## Why ConcentrationHard jointly exercises D, C and R

A successful agent must:
1. preserve distinct rank identities rather than collapsing them (**D**);
2. retain previously observed face-down-card information across later pairs (**C**);
3. bind each observed rank to its location and retrieve the appropriate location when the matching relation becomes actionable (**R**).

The deck is externally shuffled independently on every reset, so the location↔rank relation must be learned again in every episode.

## No training / no environment fitting

R47 is a frozen functional causal experiment:
- no neural training;
- no hyperparameter sweep;
- no reward shaping;
- no access to hidden state by scientific conditions;
- no result-dependent controller change after scientific seeds open.

Only `ORACLE_STATE` may use `env.get_state()`, and only as an evaluator reference.

## Controller state shared by scientific conditions

The intact controller maintains:
- last action;
- current within-pair first-card action;
- persistent map `position -> internal identity`;
- remembered failed position-pairs, to prevent degenerate retry loops.

A positive reward marks the current pair as solved; solved cards are also externally visible face-up and are never selected again.

All conditions use the same deterministic scan order and pair-selection tie-breaking.

## Conditions

### FULL_DCR

- each of the 13 ranks remains a distinct internal identity;
- card observations are retained across pairs;
- the online position↔identity relation is stored and queried correctly.

### NO_D_COLLAPSE

- persistent memory and relation machinery remain available;
- every observed rank is mapped to the same internal identity before storage/query;
- position channels remain distinct.

This removes identity differentiation while preserving historical storage and relational lookup machinery.

### C1_ONLY

- exact current rank identities remain differentiated;
- position↔identity binding machinery for the current pair remains available;
- long-term card/location memory and failed-pair history are erased after every completed two-card attempt;
- previous action/current pair-local state remain available.

This is the persistent-state lesion.

### WRONG_R_CYCLIC

- exact rank identities remain differentiated;
- cross-pair memory remains persistent;
- when searching for a remembered partner of identity (s), the controller queries the relation for

[
(s+1)mod 13
]

instead of (s).

All observations and stored identities remain intact; only the learned identity↔location relation is misbound.

### GENERIC_ISO

Every rank identity is recoded internally using the bijection

[
phi(s)=12-s,
]

and all memory/relation operations occur in that recoded space.

Because equality and pair membership are preserved under a bijection, external actions must exactly match FULL_DCR.

This is the coordinate/representation non-privilege control.

### ORACLE_STATE

Evaluator only. Uses `env.get_state()` to select known matching face-down cards directly.

ORACLE_STATE does not enter any lesion effect and cannot rescue an invalid FULL_DCR agent.

## Episode and statistical unit

One statistical seed contains **4 paired episodes per condition**.

Episode seeds:

[
100	imes seed + {0,1,2,3}.
]

All conditions receive the exact same four external shuffles for a given statistical seed.

Per episode record:
- total native return;
- successful matches;
- completion fraction = successful matches / 26;
- number of actions used;
- termination status.

Per-seed endpoints are the mean over its four episodes.

## Primary causal endpoints

[
Delta_D = Completion_{FULL}-Completion_{NO_D},
]

[
Delta_C = Completion_{FULL}-Completion_{C1},
]

[
Delta_R = Completion_{FULL}-Completion_{WRONG_R}.
]

Isomorphic guards:

[
G^{completion}_{iso}
=
|Completion_{FULL}-Completion_{ISO}|,
]

[
G^{return}_{iso}
=
|Return_{FULL}-Return_{ISO}|.
]

The same-seed integrated conjunction passes only when FULL capability and all three lesion effects pass simultaneously.

## Development panel

Scientific development seeds:

`2186001–2186016` (16 seeds).

Development authorizes confirmation only if all median criteria pass:
- FULL completion >= **0.95**;
- FULL native return >= **0.30**;
- (Delta_D >= +0.35);
- (Delta_C >= +0.50);
- (Delta_R >= +0.25);
- both GENERIC_ISO median gaps <= **1e-12**;

and at least **13/16** seeds satisfy the same-agent guard:
- FULL completion >=0.90;
- FULL return >=0.20;
- D effect >=+0.20;
- C effect >=+0.30;
- R effect >=+0.15;
- both iso gaps <=1e-12.

If development fails, confirmatory seeds remain unopened.

Development resolutions:
- `R47_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `R47_DEVELOPMENT_FAIL_NO_CONFIRM`.

## Reserved confirmatory panel

Reserved seeds:

`2187001–2187032` (32 seeds).

They remain inaccessible to the workflow unless development authorizes confirmation.

The confirmatory panel uses the exact same source, conditions, endpoint definitions and thresholds.

Confirmation requires:
- all median development criteria;
- at least **28/32** same-agent seed guards.

## Final frozen adjudication

A confirmatory PASS is:

`R47_EXTERNAL_INTEGRATED_DCR_PASS_SAME_PROGRAM`.

A confirmatory failure is:

`R47_EXTERNAL_INTEGRATED_DCR_FAIL`.

A development stop remains:

`R47_DEVELOPMENT_FAIL_NO_CONFIRM`.

## Valid interpretation of a PASS

A PASS may support that, in one unmodified external POPGym ConcentrationHard task and one persistent functional agent:
- differentiated identities D perform causal work;
- cross-pair persistent historical state C performs causal work;
- online identity↔location binding R performs causal work;
- all three lesion effects coexist on the same statistical seeds;
- a bijective internal recoding has no behavioral privilege.

Because the intact agent acquires its position↔identity map online from external observations, R47 may support **external online relational acquisition/binding** in this bounded task. It does not establish that an arbitrary neural architecture will learn R from reward optimization.

## Boundaries retained

R47 does not establish:
- global mathematical minimality of D+C+R;
- universal sufficiency across all adaptive systems;
- POLAR implementation or algorithmic superiority;
- independent-team replication E6b;
- prospective biological correspondence E7;
- AGI/ASI;
- phenomenal consciousness.

R47 is still designed, executed and adjudicated by the same research program.
