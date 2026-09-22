# R35 preregistration — representation-independent causal necessity of contextual access

Status: development protocol before confirmatory freeze  
Date: 22 September 2026  
Parent evidence: POLAR Core v1.0, R33_INCONCLUSIVE_CAPABILITY, R34 H1 inconclusive / H2 supported / H3 not supported.

## Question

Is context-dependent access itself causally useful after recurrent history C and relational information R have already been computed?

R35 does not search for a privileged neural gate. It intervenes on the causal contribution of learned historical state to the common output interface.

## Representation-independent access intervention

For each trained architecture and evaluation state, compute cue-conditioned historical contributions:

- h0 = p_hist(q=0) - p_reset(q=0)
- h1 = p_hist(q=1) - p_reset(q=1)

The history state is still computed in every condition. Only selection of the already-computed historical contribution at the readout is manipulated.

Four access modes are evaluated:

1. contextual: p_reset(q) + h_q
2. always: p_reset(q) + 0.5*(h0+h1), a cue-independent pooled historical contribution
3. random: p_reset(q) + h_r where r is Bernoulli(0.5)
4. blocked: p_reset(q)

This amendment follows the retained DEV1 failure in R35_DEV1_AMENDMENT.md and occurred before confirmatory freeze.

Thus C and R remain available upstream; A alone is manipulated at the causal interface.

## Tasks

SELECTIVE:
- q=1 target requires hidden drift/history
- q=0 target is current-only

UNIFORM:
- q is present with the same distribution
- both q values require hidden drift/history

A genuine selective-access effect predicts:
- SELECTIVE: contextual should beat always, random and blocked
- UNIFORM: always should be at least as good as contextual
- therefore the SELECTIVE-vs-UNIFORM interaction should be positive

## Architecture split

Development:
- CURRENT_MLP
- RNN
- GRU

Confirmatory:
- CURRENT_MLP
- RNN
- GRU
- WINDOW_MLP
- LSTM

Strict architecture hold-outs:
- WINDOW_MLP
- LSTM

## Topology split

Development evaluation:
- directed ring

Confirmatory:
- random-DAG
- skew-coupled

## Seed split

Development: 1906001–1906004.

Confirmatory seeds 1907001–1907012 remain unopened until source, adjudicator and thresholds are frozen.

## Adjudication logic

R35 can support A necessity only if:
- multiple capable memory architectures show positive SELECTIVE contextual advantage against all three access lesions;
- the effect is robust by seed;
- at least one strict-heldout architecture passes;
- CURRENT_MLP is a null control;
- the SELECTIVE advantage is specific relative to UNIFORM, where always-access should not be worse.

If capable architectures retain performance when contextual selection is destroyed, A as a core membership condition is weakened.

If architecture capability is insufficient to test the intervention, the result is INCONCLUSIVE rather than a necessity FAIL.

## Boundaries

Same-program synthetic evidence only. R35 cannot establish global mathematical minimality, independent replication, biological correspondence, phenomenal consciousness, AGI or ASI.
