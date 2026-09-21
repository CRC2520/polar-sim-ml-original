import unittest, numpy as np
from p3_remaining.experiments import experiment_a,experiment_b,experiment_c,experiment_d,ContinualRegimeEnv,AutobiographicalEnv,GoalEnv

class P3Smoke(unittest.TestCase):
    def test_a_deterministic(self):
        self.assertEqual(experiment_a(997001),experiment_a(997001))
    def test_b_finite(self):
        r=experiment_b(997001)
        self.assertTrue(np.isfinite(r["own_reward"]) and np.isfinite(r["own_mae"]))
    def test_c_finite(self):
        r=experiment_c(997001)
        self.assertTrue(np.isfinite(r["adaptive"]["reward"]))
    def test_d_finite(self):
        r=experiment_d(997001)
        self.assertTrue(np.isfinite(r["full"]["resource"]))
    def test_env_shapes(self):
        for E in (ContinualRegimeEnv,AutobiographicalEnv,GoalEnv):
            e=E();o=e.reset(11)
            self.assertTrue(np.asarray(o).ndim==1)

if __name__=="__main__": unittest.main()
