import unittest, numpy as np
from r11_causal.agent import CompactCausalAgent
from r11_causal.benchmarks import relational_benchmark,gate_benchmark,CyclicBufferEnv,RepairQueueEnv,AttributionEnvV3

class R11Smoke(unittest.TestCase):
    def test_agent_shapes(self):
        a=CompactCausalAgent(1); p=a.prepare(np.array([.7,.7,.5]))
        self.assertEqual(p["pred"].shape,(4,4)); self.assertIn(p["action"],range(4))
    def test_relational_deterministic(self):
        self.assertEqual(relational_benchmark(971001),relational_benchmark(971001))
    def test_gate_finite(self):
        r=gate_benchmark(971001)
        self.assertTrue(all(np.isfinite(r[k]) for k in ("accuracy","adaptive_mse","gain","permutation_damage")))
    def test_envs(self):
        for C in (CyclicBufferEnv,RepairQueueEnv,AttributionEnvV3):
            e=C();o=e.reset(7)
            self.assertEqual(np.asarray(o).shape,(3,))
            for i in range(10):
                o,r,d,info=e.step(i%4);self.assertTrue(np.isfinite(o).all() and np.isfinite(r))
if __name__=="__main__":unittest.main()
