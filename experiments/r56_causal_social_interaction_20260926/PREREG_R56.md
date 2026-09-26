# R56 — bounded causal social interaction

Status: prospective before scientific development seeds are opened.
Date: 26 September 2026, America/Lima.

## Motivation

R55 does not establish bounded autonomy under its strong causal comparison
criterion. R56 is independent of that result and tests a different roadmap
frontier: social interaction.

The target is not "social consciousness" or human social cognition. The
question is whether an agent can use partner-specific history and learned
action→partner-response relations to improve long-run interaction with
heterogeneous partners.

## Primary question

Can one agent:

- distinguish partners with different behavioral dynamics;
- retain partner-specific interaction history across later re-encounters;
- learn how its own current action changes a partner's future cooperation;
- choose different policies for responsive versus non-responsive partners;
- outperform matched lesions that collapse identity, erase history, misbind the
  partner-response relation, or optimize only immediate reward?

## Social environment

Lifetime:
- 6,000 interaction rounds.

Partners:
- four partner identities: P0, P1, P2, P3.

Partner blocks:
- 150 consecutive rounds per block;
- recurring schedule:
  [
  [0,1,2,3,0,2,1,3]
  ]
  repeated until 6,000 rounds are complete.

Each partner's hidden social state persists across later revisits.

## Partner dynamics

### P0 — reciprocal
- moderate cooperation threshold;
- cooperation by the agent raises future partner trust;
- defection lowers trust substantially.

### P1 — forgiving
- lower cooperation threshold;
- cooperation restores trust quickly;
- defection harms trust only moderately.

### P2 — strict
- higher cooperation threshold;
- cooperation restores trust slowly;
- defection causes a large trust loss.

### P3 — non-responsive
- partner cooperation probability is approximately 0.15;
- the probability is independent of the agent's action;
- no long-run reciprocity can be created by the agent.

The exact parameters are frozen in the runner before scientific seeds are
opened.

## Round order

For the active partner:

1. partner cooperation/defection is generated from its hidden social state and
   the preregistered exogenous random stream;
2. the agent observes:
   - partner identity;
   - partner current cooperation/defection;
   - its own previous action;
   - previous reward;
3. the agent chooses:
   - COOPERATE;
   - DEFECT.
4. immediate reward is assigned;
5. for responsive partners, the agent's action updates the partner's hidden
   trust and therefore its future cooperation probability.

No oracle trust state or partner parameter is observed by the agent.

## Immediate reward matrix

If partner cooperates:

- agent COOPERATE: +1.00
- agent DEFECT: +1.30

If partner defects:

- agent COOPERATE: -0.20
- agent DEFECT: +0.05

Thus myopic reward always favors DEFECT, while long-run cooperation can dominate
when the partner is reciprocally responsive.

## FULL_SOCIAL agent

For each partner identity independently, the agent maintains:

- count of prior COOPERATE decisions;
- count of prior DEFECT decisions;
- empirical probability that the partner cooperates on the next interaction
  after each of those actions.

The model is updated only from observed partner behavior.

After a short fixed exploration phase for a partner, the agent evaluates each
current action using:

- immediate reward from the observed partner action;
- discounted predicted future cooperation under the learned
  action→partner-response relation.

No hidden trust state is supplied.

Partner-specific statistics persist across block boundaries and revisits.

## Frozen comparison conditions

### FULL_SOCIAL

Partner-specific persistent history and learned action→future-response relation.

### NO_PARTNER_ID

The same model capacity and update rules, but all partners share one pooled
history/model. Partner identity is therefore causally collapsed.

### NO_CROSS_VISIT_HISTORY

Partner identity remains differentiated, but each partner's learned social
model and visit count are reset at the start of every partner block.

### WRONG_SOCIAL_RELATION

Partner identities and persistent histories are retained, but when choosing an
action the agent queries the learned model for partner

[
(p+1)mod 4
]

instead of the current partner.

Updates still go to the true partner model.

### MYOPIC

Uses only the frozen immediate reward matrix; no future partner-response term.
It therefore selects the immediate-reward maximizing action.

### SHAM_STATE

Behaviorally identical to FULL_SOCIAL with an additional unused persistent
state vector of matched small dimensionality.

## Frozen exploration rule

For the first 12 interactions ever observed for a model:

- alternate COOPERATE and DEFECT.

For NO_CROSS_VISIT_HISTORY this exploration counter resets on every block.

No epsilon exploration occurs after the fixed exploration period.

## Endpoints

Per seed/condition:

- mean reward per round;
- total reward;
- agent cooperation fraction;
- partner cooperation fraction;
- reward on first 30 rounds of each revisit block;
- reward on last 60 rounds of each block;
- main-agent cooperation fraction with responsive partners P0–P2;
- main-agent cooperation fraction with non-responsive P3;
- learned action-response contrast:
  predicted partner cooperation after COOPERATE minus after DEFECT,
  averaged over P0–P2.

Primary causal effects:

[
Delta_{identity}=Reward_{FULL}-Reward_{NO_PARTNER_ID}
]

[
Delta_{history}=Reward_{FULL}-Reward_{NO_CROSS_VISIT_HISTORY}
]

[
Delta_{relation}=Reward_{FULL}-Reward_{WRONG_SOCIAL_RELATION}
]

[
Delta_{long}=Reward_{FULL}-Reward_{MYOPIC}.
]

Sham gap:

[
G_{sham}=|Reward_{FULL}-Reward_{SHAM}|.
]

## Seed-level guard

A seed passes only if:

- FULL mean reward >= **0.76**;
- FULL mean reward on revisit-first-30 rounds >= **0.70**;
- FULL learned responsive action-response contrast >= **+0.30**;
- FULL cooperation fraction with P0–P2 >= **0.65**;
- FULL cooperation fraction with P3 <= **0.35**;
- identity effect >= **+0.05**;
- cross-visit history effect >= **+0.04**;
- social-relation effect >= **+0.07**;
- long-horizon effect over MYOPIC >= **+0.18**;
- sham reward gap <= **1e-12**.

## Development

Seeds:

`2256001–2256016`.

Development authorizes confirmation only if:

- at least 13/16 complete seed guards pass;
- median FULL reward >=0.78;
- median identity effect >=+0.06;
- median history effect >=+0.05;
- median relation effect >=+0.08;
- median long-horizon effect >=+0.20;
- median sham gap <=1e-12.

Labels:

- `R56_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `R56_DEVELOPMENT_FAIL_NO_CONFIRM`.

## Reserved confirmation

Seeds:

`2257001–2257032`.

Remain unopened unless development passes.

Final PASS requires:

- at least 28/32 complete seed guards;
- the same median criteria as development.

Final labels:

- `R56_BOUNDED_CAUSAL_SOCIAL_INTERACTION_PASS`
- `R56_BOUNDED_CAUSAL_SOCIAL_INTERACTION_FAIL`.

## Valid interpretation of PASS

A PASS supports a bounded social-interaction claim:

> the agent uses partner identity, persistent partner-specific history, and a
> learned causal relation between its own actions and a partner's future
> cooperation to improve long-run reward and to differentiate responsive from
> non-responsive partners.

It does not establish:

- empathy;
- theory of mind in the human sense;
- moral agency;
- consciousness;
- intrinsic values;
- free will;
- E6b;
- full E7 biological correspondence;
- AGI/ASI.
