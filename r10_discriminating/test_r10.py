import unittest
import numpy as np
from r10_discriminating.experiments import CyclicBufferEnv, RepairQueueEnv, AttributionEnv, experiment3

class R10Smoke(unittest.TestCase):
    def test_new_envs_finite(self):
        for cls in (CyclicBufferEnv,RepairQueueEnv):
            e=cls(); o=e.reset(123)
            self.assertEqual(np.asarray(o).shape,(3,))
            for i in range(20):
                o,r,done,info=e.step(i%4)
                self.assertTrue(np.isfinite(o).all() and np.isfinite(r))
                self.assertIn("alive",info)
    def test_attribution_labels(self):
        e=AttributionEnv(); e.reset(456)
        seen=set()
        for i in range(100):
            _,r,_,info=e.step(i%4)
            self.assertTrue(np.isfinite(r))
            seen.add(info["source"])
            self.assertIn(info["agebin"],(0,1,2,3))
        self.assertTrue(seen.issubset({0,1,2}))
    def test_gate_challenge_deterministic(self):
        a=experiment3(963001); b=experiment3(963001)
        self.assertEqual(a,b)
        self.assertTrue(np.isfinite(a["adaptive_mse"]))

if __name__=="__main__":
    unittest.main()
