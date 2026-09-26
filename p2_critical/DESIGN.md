# POLAR P2-Critical — development design

This campaign addresses four remaining critical gaps after P0/R14 and P1.

1. **POLAR vs GENERIC isomorphic relational learner.**
   Both learners receive identical 16-dimensional observations, the same training/test data,
   the same base linear terms, exactly eight pair-interaction slots and identical ridge fitting.
   POLAR constrains the eight relations to a disjoint matching; GENERIC selects the eight highest
   residual interactions with no matching constraint. A random equal-size relation set is a
   negative control. Resolution can be either POLAR-specific superiority or practical equivalence
   when the generic learner reconstructs the same relations. We do not force a superiority claim.

2. **Relational discovery inside one integrated closed-loop agent.**
   A single agent learns pair relations only from factual action consequences, updates memory,
   uncertainty and a contextual cross-route, acts with the learned model, and must rediscover the
   relation set after a mid-trajectory regime change. Evaluator pair labels are never policy inputs.

3. **Theory-discriminating functional lesions.**
   The same synthetic cognitive task instantiates separable primary, workspace/broadcast,
   higher-order confidence and relational-tension pathways. Three targeted lesions must yield
   distinct functional signatures. This tests discriminating computational predictions only;
   it does not adjudicate phenomenal consciousness or prove any neuroscience theory.

4. **Standard-equation OOD validation.**
   The controller is evaluated on logistic harvesting, first-order RC thermal control and queue
   service dynamics, defined independently of POLAR's internal variables. Only a fixed target
   identification prefix is allowed. This is OOD validation against standard mathematical task
   families, not an independent-team replication.

Development seeds may tune numerical constants and thresholds. Confirmatory seeds will be new and
will not be generated or executed until a later freeze commit.
