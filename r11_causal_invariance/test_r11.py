import unittest, numpy as np
from r11_causal_invariance.core import OnlineAdapter
from r11_causal_invariance.environments import RelationalEnv, AttributionTransferEnv

class R11Smoke(unittest.TestCase):
    def test_adapter_finite(self):
        a=OnlineAdapter(3)
        x=np.array([.1,.2,.3]); y=np.array([.4,.5,.6,.7])
        for i in range(20): a.update(x,i%4,y,y+.1)
        self.assertTrue(np.isfinite(a.predict_raw(x)).all())
    def test_envs(self):
        for env in (RelationalEnv(False),RelationalEnv(True),AttributionTransferEnv()):
            o=env.reset(123)
            self.assertEqual(np.asarray(o).shape,(3,))
            for i in range(30):
                o,r,d,info=env.step(i%4)
                self.assertTrue(np.isfinite(o).all() and np.isfinite(r))
                self.assertIn('alive',info)
if __name__=='__main__': unittest.main()
