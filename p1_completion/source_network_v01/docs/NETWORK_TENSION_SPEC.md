# Inter-polarity tension extension 0.1

Status: separately versioned engineering extension, NOT a new performance study,
confirmation of the full architecture, or consciousness evidence. Frozen Study 1/2
sources, data and the negative exclusive-advantage conclusion remain unchanged.

## Architecture preserved

The research object remains the original layered proposal: perception/context;
Tension Engine; Conscious Integration C; Ethical Self-Regulation E; Polar
Unconscious U; Motivational Objectives; IACL; planning/action; hybrid integration.
The manuscript restores these foundations and diagrams in its single main.pdf.
This extension implements only the missing explicit BETWEEN-polarity route.
It reuses the contextual model, rather than claiming all proposed layers exist.
The resource allocator is not content broadcast; admissibility is not normative
reasoning; cue memory is not a full unconscious or autobiographical subsystem.

## Variables and operational tension

State p has shape (control units, types, 2), independent channels in [0,1].
Flatten the first two axes to N polarities, keeping adjacent pole channels.
Orientation a=p+ - p- and activity q=p+ + p- are not tension.
With the controller's effective target d and own planning gain G:

    v = clip(d/G, 0, 1)
    mismatch_i = mean_s |v_i,s - p_i,s|
    conflict_i = chi_i p_i,+ p_i,-
    tau_i = mismatch_i + conflict_i       (0 <= tau <= 2)

chi is supplied contextual incompatibility in [0,1], not a psychological
measurement or learned normative property. Matching coactivation with chi=0 is
not conflict. Inactivity with unmet demand can produce tension. Hidden targets
are filtered by the inherited observation/memory interface before this calculation.

## Directed propagation

W[2N,2N] maps channel activities; K[2N,N] maps polarity tension. Rows receive;
columns send. This extension prohibits within-polarity W blocks and own-polarity
K entries so between-polarity paths are explicitly isolated. Both signs are allowed.

    base = p + eta*(d/G-p)
    proposal = base + eta*(W@p + K@tau)
    next_p = inherited_feasibility_map(proposal)

Every term consumes pre-update state, synchronously. eta is the existing clipped
response multiplier. The unchanged feasibility map applies clipping, permissions
and resource allocation. It is not an exact optimal projection. Zero W/K reproduces
the original contextual model, including subsequent learning. W and K are configured
hypotheses in 0.1, NOT learned network structure. No consensus force is added.

Traces retain every input to this calculation, per-edge products, each aggregate
term, base/network proposals, constraints, actions and feedback. Source hashes
accompany the deterministic probe. Pole reversal transforms W on both channel
axes and K on its receiving axis, as well as inherited state and future inputs.
Signed-plus-intensity coordinates remain equivalent.

Graph effects can propagate through multiple steps. Shared budgets provide an
additional indirect route even with W=K=0. Therefore the probe uses a nonbinding
budget and disables memory and gain learning. No general contraction theorem is
claimed: output bounds are not asymptotic stability or evidence of useful control.

## Executed engineering checks

`python -m unittest discover -s tests -v` includes 14 added tests (82 total at this
revision): baseline equivalence, directed/inhibitory routes, mismatch and contextual
conflict, multi-step propagation, separate lesions, constraints, trace reconstruction,
coordinate equivalence, pole relabeling, hidden input isolation and feedback order.

`python scripts/probe_network_tension.py` writes full paired traces in
`docs/network_tension_probe.json`. Removing K while retaining W, observations,
initial state and budget yields first-step receiving effect 0.1248 and second-step
downstream effect 0.024336; the isolated polarity remains unchanged. These numbers
are activity differences, NOT improvements in reward, intelligence or consciousness.

## Progressive legacy audit

`CRC2520/polar-sim-ml@22891d52600f600573d88839ae237bfd6d723a6b` is inspected read-only.
Phase 7 genuinely reads its cross-node influence matrix. Phase 30 computes but
never reads total_inputs, and its detached critic loss cannot update the actor.
An instrumented 20-step replay records zero actor gradient tensors and zero actor
parameter change. ReLU/tanh plus normalized positive subweights also imply HGI >=
0.9 for ten nodes by construction, with no access to negative activation.

Run `python scripts/audit_progressive_legacy.py --repo /path/to/polar-sim-ml --out
 docs/legacy_progressive_audit.json`. It verifies source blob hashes. This is a
specific audit of phases 7 and 30, not validation of all intermediate scripts.
No write is made to the inspected repository and no historical result is relabeled.

## Next empirical requirement

Predefine the tension proxy, learned/configured topology, tuning budget and unseen
seeds. Compare zero-network, W-only, K-only, equally sparse shuffled topology,
capacity-matched generic controllers and exact coordinate controls. Include crossed,
absent and misaligned dependencies; separate topology from budget/goal changes.
Use external performance, constraints, cost and recovery as endpoints. Keep the
current Study 2 decision `suspend_exclusive_polar_advantage_claim` unchanged until
new evidence supports a genuinely new, predeclared hypothesis.
