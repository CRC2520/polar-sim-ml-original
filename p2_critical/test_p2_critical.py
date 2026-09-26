import unittest, numpy as np
from p2_critical.experiments import experiment1,experiment2,experiment3,experiment4,LogisticHarvest,ThermalRC,QueueService

class P2CriticalSmoke(unittest.TestCase):
    def test_determinism_e1(self):
        self.assertEqual(experiment1(991001),experiment1(991001))
    def test_e2_finite(self):
        r=experiment2(991001)
        self.assertTrue(np.isfinite(r["adaptive"]["reward_post"]))
    def test_e3_finite(self):
        r=experiment3(991001,3000)
        self.assertTrue(np.isfinite(r["full"]["meta_brier"]))
    def test_standard_envs(self):
        for C in (LogisticHarvest,ThermalRC,QueueService):
            e=C(); o=e.reset(1)
            for i in range(10):
                o,r,d,info=e.step(i%4)
                self.assertTrue(np.isfinite(o).all() and np.isfinite(r))
                self.assertIn("alive",info)

if __name__=="__main__": unittest.main()
