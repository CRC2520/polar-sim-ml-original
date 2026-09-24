# R46 — external D/R causal transport on MPE2

Status: prospective before R46 development seeds are opened.
Date: 24 September 2026, America/Lima.

## Starting boundary

R45 closed the direct reproduced recurrent-checkpoint comparator boundary for the persistent-state contract C:

`R45_DIRECT_REPRODUCED_CHECKPOINT_COMPATIBILITY_PASS`.

R46 does **not** retune R45 and does not change POLAR Core v1.1:

[
mathcal P_{core}^{v1.1}=D+C+R,qquad A 	ext{conditional}.
]

R46 addresses the remaining external functional boundary for:

- **D** — differentiated causal degrees of freedom;
- **R** — relation-specific binding and reidentification when the externally generated episode relation changes.

R46 does not attempt an independent-team replication and therefore cannot close E6b.

## External benchmark fixed before execution

External package: Farama Foundation MPE2.

Release: `v1.1.0`.

Exact source commit:

`7590d9d52791e321974d4fda6090fb18f34dbf49`.

R46 installs that source commit directly and does not modify its environment code.

Two externally maintained MPE2 tasks are used:

1. **Simple Reference v3** — two agents, three landmarks, two-way communication, episode-random target landmarks.
2. **Simple Crypto v3** — Alice/Bob share an episode-random private key while Eve observes the public message.

Environment defaults are retained except that rendering is disabled. The native 25-cycle episode length and discrete action spaces are unchanged.

## Why these tasks

### Simple Reference

Each agent observes:
- its own velocity;
- the three landmark relative-position channels;
- the target color that the **other** agent should pursue;
- communication from the other agent.

The environment independently redraws the target landmark each episode. Therefore the active relation:

[
	ext{partner goal}ightarrow	ext{communication symbol}ightarrow
	ext{landmark slot}
]

must be selected again after episode reset.

### Simple Crypto

Alice observes the current target bit and private key, Bob observes the current private key and Alice communication, and Eve observes only Alice communication. The key is redrawn by the external environment each episode.

This supplies a second external family in which relational organization is:

[
(	ext{message},	ext{key})ightarrow 	ext{cipher},qquad
(	ext{cipher},	ext{key})ightarrow 	ext{message}.
]

## No training / no benchmark fitting

R46 is a causal transport experiment, not a new learning campaign.

All controller equations and lesion rules are frozen in the R46 runner before scientific seeds are opened. There is:

- no neural training;
- no hyperparameter sweep;
- no result-dependent policy search;
- no environment modification;
- no reward shaping.

This deliberately tests whether the D/R organizational distinctions have causal value in externally generated task relations. It does **not** test external learnability of R; learnability remains supported by prior internal experiments. R46 tests external **binding and reidentification**.

# Panel A — Simple Reference

## Episode structure

Each statistical seed defines six externally reset episodes.

Episode 0 is a relation-cache warmup and is excluded from scientific endpoint aggregation.

Episodes 1–5 are the frozen evaluation sequence.

Each episode retains the native 25 cycles.

## Conditions

### FULL_DR

A deterministic communication/navigation controller:

1. differentiates the three landmark-relative-position channels;
2. decodes the externally supplied target color for the partner;
3. sends a target symbol;
4. decodes the partner's current communication;
5. selects the corresponding current landmark relation;
6. moves toward that landmark using only native observation channels.

### NO_D

The three landmark-relative-position channels are replaced inside the controller by their channel mean before target selection. Communication and the relation decoder remain otherwise intact.

This removes differentiated spatial causal degrees of freedom while preserving dimensionality at the controller interface.

### WRONG_R

All native landmark channels remain distinct, but the received target relation is bound to the wrong landmark using a fixed cyclic permutation.

This tests relation-specific causal binding without removing information channels.

### FROZEN_PREVIOUS_R

The controller preserves differentiated landmark channels but, on evaluation episodes 1–5, uses the partner relation decoded in the **previous episode** instead of the current relation.

The external environment independently redraws targets at reset. This is the preregistered R reidentification lesion.

### GENERIC_ISO

A fixed bijective internal relabeling of landmark/symbol coordinates is applied and exactly inverted at the action boundary. Native external actions are behaviorally equivalent to FULL_DR.

Expected role: implementation-coordinate non-privilege control.

### STATIONARY

No physical movement. Used only as a normalization floor.

### ORACLE_RELATION

Evaluator reference using the environment's true current target relation for movement. It does not enter any lesion comparison and is not a deployable POLAR controller.

## Endpoint

For each episode, cost is:

[
J=-operatorname{mean}_{t,agents}(reward).
]

For each seed, costs are averaged over evaluation episodes 1–5.

Normalized score:

[
S=rac{J_{stationary}-J}
        {J_{stationary}-J_{oracle}}.
]

Seeds with normalization denominator <=1e-8 are invalid and cause the panel to fail rather than being dropped.

Per-seed causal effects:

[
Delta_D=S_{FULL}-S_{NO_D},
]

[
Delta_{R,binding}=S_{FULL}-S_{WRONG_R},
]

[
Delta_{R,rebind}=S_{FULL}-S_{FROZEN_PREVIOUS_R}.
]

Coordinate gap:

[
G_{iso}=|S_{FULL}-S_{GENERIC_ISO}|.
]

The runner also records the fraction of episode transitions in which the current received target relation differs from the cached previous relation.

# Development gate

Development seeds:

`2176001–2176016` (16 seeds).

Development authorizes confirmation only if all median criteria pass:

- FULL normalized score >= **0.70**;
- D effect (Delta_D) >= **+0.20**;
- R binding effect >= **+0.20**;
- R reidentification effect >= **+0.10**;
- GENERIC_ISO gap <= **1e-12**;
- relation-change fraction >= **0.50**;

and at least **13/16** seeds satisfy:

- FULL score >=0.60;
- D effect >=+0.10;
- R binding effect >=+0.10;
- R reidentification effect >=+0.05;
- iso gap <=1e-12;
- relation-change fraction >=0.40.

If development fails, all confirmatory seeds remain unopened.

Development resolutions:

- `R46_DEVELOPMENT_AUTHORIZE_CONFIRM`
- `R46_REFERENCE_DEV_FAIL_NO_CONFIRM`

# Confirmation panel B1 — fresh Simple Reference

Reserved confirmatory seeds:

`2177001–2177032` (32 seeds).

Same controllers, episode structure and frozen endpoints.

B1 passes only if all median criteria from development pass and at least **28/32** seed guards pass.

# Confirmation panel B2 — Simple Crypto cross-task transport

Reserved confirmatory seeds:

`2177101–2177132` (32 seeds).

Each statistical seed defines 12 externally reset episodes.

Episode 0 is a relation-cache warmup and is excluded. Episodes 1–11 are evaluated.

## Conditions

### FULL_DR

- Alice differentiates current target and current private-key channels.
- Alice sends a key-conditioned cipher.
- Bob reconstructs the target using the current key.
- Eve receives only the public cipher.

The frozen binary relation is equivalent to a one-time-pad XOR over the two active native symbols; unused communication dimensions remain untouched.

### NO_D

At Alice, current target and key channels are merged by their elementwise mean and the same merged symbol is used for both causal roles before encryption. Bob remains intact.

This removes target/key differentiation at the encoder without altering the external environment.

### NO_R_DIRECT

Alice sends the target directly without the key-conditioned relation. Bob and Eve decode the direct signal.

This preserves differentiated inputs but removes the private relational transformation.

### FROZEN_PREVIOUS_R

Alice encrypts the current target using the **previous episode's** key relation while Bob decodes using the current externally supplied key.

This tests failure to reidentify the current episode relation.

### GENERIC_ISO

A bijective internal recoding of the active binary symbols is conjugated through the relation and inverted before native actions. External behavior is exactly equivalent to FULL_DR.

## Crypto endpoint

For each episode, the good-team score is the mean native reward of Alice and Bob over the 25 cycles.

Per seed, the score is averaged over episodes 1–11.

Effects:

[
Delta_D=R_{FULL}-R_{NO_D},
]

[
Delta_{R,direct}=R_{FULL}-R_{NO_R_DIRECT},
]

[
Delta_{R,rebind}=R_{FULL}-R_{FROZEN_PREVIOUS_R}.
]

B2 passes if medians satisfy:

- FULL good-team reward >= **0.65**;
- D effect >= **+0.40**;
- direct-relation effect >= **+0.40**;
- reidentification effect >= **+0.30**;
- GENERIC_ISO absolute gap <= **1e-12**;

and at least **26/32** seeds satisfy:

- FULL reward >=0.40;
- D effect >=+0.20;
- direct-relation effect >=+0.20;
- reidentification effect >=+0.20;
- iso gap <=1e-12.

# Final frozen adjudication

R46 final resolution is:

### PASS

`R46_EXTERNAL_D_R_TRANSPORT_PASS_SAME_PROGRAM`

only if development authorizes confirmation and **both B1 and B2 pass**.

### PARTIAL

`R46_EXTERNAL_D_R_TRANSPORT_PARTIAL`

if confirmation runs but only one confirmatory family passes.

### FAIL

`R46_EXTERNAL_D_R_TRANSPORT_FAIL`

if confirmation runs and neither family passes.

A development failure remains:

`R46_REFERENCE_DEV_FAIL_NO_CONFIRM`.

# Interpretation boundaries

Even a full R46 PASS may support only:

- external functional necessity of differentiated channels D in the tested MPE2 tasks;
- external relation-specific causal binding;
- external episode-to-episode relation reidentification;
- transport across two externally maintained task families;
- lack of privileged coordinate implementation if GENERIC_ISO remains equivalent.

It does **not** establish:

- external learnability of R;
- a single external task requiring the complete D+C+R conjunction;
- global minimality or universal sufficiency of D+C+R;
- POLAR implementation superiority;
- E6b independent replication;
- E7 prospective biological correspondence;
- AGI/ASI;
- phenomenal consciousness.

R45 remains the external C comparator result. R46 must not retroactively convert separate-task evidence into a same-agent D+C+R proof.

# Governance

- Exact MPE2 tag/commit, source runner and protocol hashes are recorded.
- Engineering smoke uses seed 2175000 only.
- Development and confirmatory seeds are disjoint.
- Confirmatory seeds are conditionally blocked until development gate PASS.
- Raw per-seed results and adjudication JSON are exported as GitHub Actions artifacts.
- No threshold, lesion, endpoint or seed may change after development execution begins.
