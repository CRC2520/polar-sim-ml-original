import unittest
from copy import deepcopy
import numpy as np
from integrated_polar import IntegratedAgent,AgentConfig
from integrated_polar.layers import CalibratedMonitor
from integrated_polar.projection import project
from gap_resolution.tasks import run_trial,DEV_SEEDS,FINAL_SEEDS,CELLS,MODES
from gap_resolution.evaluate import decision,CLAIMS,verify_record,CAPABILITY_SEEDS
from gap_resolution.run import check_registration


class GapEvaluationTests(unittest.TestCase):
    def test_design_and_seeds(self):
        self.assertEqual(len(FINAL_SEEDS)*len(CELLS)*len(MODES),1056)
        self.assertFalse(set(DEV_SEEDS)&set(FINAL_SEEDS));self.assertFalse(set(CAPABILITY_SEEDS)&set(FINAL_SEEDS))
    def test_decision_matches_all_registered_clauses(self):
        flags={x[0]:True for x in CLAIMS}
        self.assertEqual(decision(False,flags),'invalid_evaluation')
        self.assertEqual(decision(True,flags),'bounded_functional_improvements_supported')
        for k in flags:self.assertEqual(decision(True,{**flags,k:False}),'engineering_verified_with_partial_empirical_support')
        with self.assertRaises(ValueError):decision(True,{})
    def test_registration_is_required(self):
        with self.assertRaises(ValueError):check_registration(None)
    def test_record_replay_and_tamper_detection(self):
        r=run_trial(791401,('fixed',.6),'full')
        verify_record(r)
        bad=deepcopy(r);bad['frames'][0]['effect'][0]+=.01
        with self.assertRaises(ValueError):verify_record(bad)
        bad=deepcopy(r);bad['frames'].pop()
        with self.assertRaises(ValueError):verify_record(bad)
    def test_projection_matches_independent_bisection(self):
        rng=np.random.default_rng(791402)
        for _ in range(60):
            x=rng.uniform(-.5,1.5,8);c=rng.uniform(.5,2,8);w=rng.uniform(.1,5,8);budget=rng.uniform(0,2)
            lo=0.;hi=max(1,float(np.max(w*x/c)))
            if c@np.clip(x,0,1)<=budget:reference=np.clip(x,0,1)
            else:
                for _ in range(100):
                    mid=(lo+hi)/2
                    if c@np.clip(x-mid*c/w,0,1)>budget:lo=mid
                    else:hi=mid
                reference=np.clip(x-hi*c/w,0,1)
            np.testing.assert_allclose(project(x,c,budget,weights=w),reference,atol=1e-10)
    def test_saturation_can_hide_a_real_content_difference(self):
        a=IntegratedAgent(AgentConfig(tanks=1));b=IntegratedAgent(AgentConfig(tanks=1));b.workspace.cut.add('planner')
        o={'state':np.array([.35]),'cue':'A','event':{'targets':[[.8]]*3,'priorities':[[1.]]*3}}
        ua,ub=a.act(o),b.act(o)
        np.testing.assert_allclose(ua,ub,atol=1e-10)
        self.assertNotEqual(a.goals.targets[0,0],b.goals.targets[0,0])
    def test_review_does_not_bypass_feasible_mandatory_minimum(self):
        low=CalibratedMonitor.restore({'edges':[0.],'probabilities':[0.,0.],'base':0.})
        a=IntegratedAgent(AgentConfig(tanks=1,meta_threshold=.55),low)
        o={'state':np.array([.35]),'cue':'A','event':{'targets':[[.8]]*3,'priorities':[[1.]]*3},'budget':1.,'normative':{'mandatory_min':[.2,0]}}
        u=a.act(o);self.assertTrue(a.trace['requested_review']);self.assertGreaterEqual(u[0],.2)


if __name__=='__main__':unittest.main()
