from __future__ import annotations
import unittest, numpy as np
from r10_completion.core import orthogonal, ridge_fit, ridge_predict
from r10_completion import exp2_isomorphic, exp3_gate, exp5_integrated

class R10Tests(unittest.TestCase):
    def test_orthogonal_ridge_isomorphism(self):
        rng=np.random.default_rng(1); x=rng.normal(size=(300,12)); y=rng.normal(size=(300,4))
        q=orthogonal(12,2)
        p=ridge_predict(x,ridge_fit(x,y,.03))
        g=ridge_predict(x@q,ridge_fit(x@q,y,.03))
        self.assertLess(np.max(np.abs(p-g)),1e-9)
    def test_e2_exact_control(self):
        r=exp2_isomorphic.run(970101)
        self.assertTrue(r["pass_exact"])
        self.assertEqual(r["matched_structural"]["active_coefficients_per_output"],[8,8,8,8])
    def test_gate_deterministic(self):
        a=exp3_gate.run(970101); b=exp3_gate.run(970101)
        self.assertEqual(a,b)
        self.assertTrue(np.isfinite(a["confirmatory"]["learned_reward"]))
    def test_integrated_same_tape_sham(self):
        ag=exp5_integrated.train(970101)
        intact=exp5_integrated.evaluate(ag,971001,None)
        sham=exp5_integrated.evaluate(ag,971001,"sham")
        for key in ("primary_balanced_accuracy","routing_accuracy","meta_brier","counterfactual_regret_improvement","polar_reward_advantage"):
            self.assertAlmostEqual(intact[key],sham[key],places=12)
if __name__=="__main__": unittest.main()
