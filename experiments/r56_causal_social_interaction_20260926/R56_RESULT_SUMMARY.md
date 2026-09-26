# R56 result — bounded causal social interaction

Frozen development resolution:

`R56_DEVELOPMENT_FAIL_NO_CONFIRM`

## Canonical execution

- run: `36252774670`
- scientific head SHA:
  `4e21f0bfd536e56fd0cff4c0f916123f48a4d931`
- development seeds: 2256001–2256016
- confirmatory seeds 2257001–2257032 remain unopened.

## Development medians

- FULL mean reward: **0.3470**
- revisit-first-30 reward: **0.3543**
- learned responsive action→partner-response contrast: **+0.3714**
- cooperation with responsive partners P0–P2: **0.3351**
- cooperation with non-responsive P3: **0.0040**
- partner-identity effect: **+0.2139**
- cross-visit history effect: **+0.0327**
- social-relation effect: **−0.0395**
- long-horizon effect over MYOPIC: **+0.2187**
- sham gap: **0**

Complete seed guards: **0/16**.

## Interpretation

R56 produces several bounded positive diagnostics:

1. Partner identity has large functional value relative to a pooled model.
2. The learned model detects that COOPERATE versus DEFECT changes future
   cooperation for the responsive partners.
3. The non-myopic agent outperforms the purely immediate-reward policy.
4. The agent learns to defect with the non-responsive partner P3.

However the strong social claim fails:

- absolute FULL reward is well below the frozen competence gate;
- responsive-partner cooperation is too low;
- cross-visit memory gain is below threshold;
- most importantly, querying the wrong partner relation does not cause the
  expected damage; its median effect is negative.

Thus the agent contains social-history signals but does not use the learned
partner-response relation in the preregistered robust causal manner.

The correct resolution is:

`R56_DEVELOPMENT_FAIL_NO_CONFIRM`.

## Confirmation protection

Seeds 2257001–2257032 were not opened.

## Boundaries

R56 does not establish:
- robust bounded causal social interaction under the full gate;
- empathy;
- human-like theory of mind;
- moral agency;
- intrinsic values;
- consciousness;
- E6b;
- full E7 correspondence;
- AGI/ASI.

Positive diagnostic subeffects must not be promoted into a social-cognition
PASS.
