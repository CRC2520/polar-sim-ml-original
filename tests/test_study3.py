import unittest
import numpy as np
from study3.core import *
from study3.analysis import classify,CONTRASTS
from study3.run import verify_registration


class StudyThreeTests(unittest.TestCase):
    def setUp(self):self.selected={m:dict(eta=.65,a=-.2,b=.2) for m in TUNED};self.selected['retuned_zero'].update(a=0.,b=0.);self.selected['gradient'].update(a=0.,b=0.)
    def test_design_counts_and_disjoint_seeds(self):
        self.assertEqual(sum(len(candidates(m)) for m in TUNED)*len(DEV_SEEDS)*len(CELLS),8208)
        self.assertEqual(len(FINAL_SEEDS)*len(CELLS)*len(MODES),5760)
        self.assertFalse(set(FINAL_SEEDS)&set(DEV_SEEDS))
    def test_factorial_topology_budget_does_not_change_other_inputs(self):
        first=environment(791001,CELLS[0])
        for c in CELLS:
            e=environment(791001,c)
            for k in ('target','gains','noise','costs','weights','chi','allowed'):
                np.testing.assert_array_equal(e[k],first[k])
    def test_absent_effects_identical_between_families(self):
        a=environment(791001,('linear','absent','ample'));b=environment(791001,('congestion','absent','ample'))
        np.testing.assert_array_equal(transition(a,10,np.full(SHAPE,.4)),transition(b,10,np.full(SHAPE,.4)))
    def test_distinct_topologies_equal_support_size(self):
        all_w=[]
        for top in SOURCES:
            W,K=templates(top);all_w.append(W)
            self.assertEqual(np.count_nonzero(W),16);self.assertEqual(np.count_nonzero(K),8)
            owner=np.arange(8)//2
            self.assertFalse(np.any(W[owner[:,None]==owner[None,:]]))
        self.assertFalse(np.array_equal(all_w[0],all_w[1]));self.assertFalse(np.array_equal(all_w[0],all_w[2]))
    def test_paired_lesions_preserve_parameters_other_than_route(self):
        full=build('full',self.selected,791001)
        noK=build('no_K',self.selected,791001);noW=build('no_W',self.selected,791001)
        np.testing.assert_array_equal(full.W,noK.W);np.testing.assert_array_equal(full.K,noW.K)
        self.assertFalse(np.any(noK.K));self.assertFalse(np.any(noW.W))
        self.assertEqual(full.config,noK.config)
    def test_generic_and_signed_equivalence_with_online_learning(self):
        for cell in [CELLS[0],CELLS[-1]]:
            env=environment(791001,cell)
            r=trial('full',self.selected,env)
            actions=np.array([f['action'] for f in r['frames']])
            for mode in ['generic_equivalent','signed_intensity']:
                q=trial(mode,self.selected,env)
                np.testing.assert_allclose([f['action'] for f in q['frames']],actions,atol=1e-12,rtol=0)
    def test_external_scores_recomputed_constraints(self):
        r=trial('full',self.selected,environment(791001,CELLS[-1]))
        calc=metrics(r['frames']);self.assertEqual(calc['loss'],r['metrics']['loss'])
        self.assertEqual(calc['violations'],0)
        with self.assertRaises(ValueError):metrics(r['frames'][:-1])
    def test_hidden_truth_keys_cannot_change_policy(self):
        env=environment(791001,CELLS[0]);a=build('full',self.selected,791001);b=build('full',self.selected,791001)
        o={'target':env['target'][0],'budget':8.}
        np.testing.assert_array_equal(a.act(o),b.act({**o,'true_gain':np.full(SHAPE,99),'true_topology':'anything'}))
    def test_final_requires_public_registration(self):
        with self.assertRaises(ValueError):verify_registration(None)
    def test_decision_all_clauses_and_failure_precedence(self):
        claims={k:True for k,_,_ in CONTRASTS};gates={k:True for k in ('cost','post_switch','constraints')}
        self.assertEqual(classify(False,claims,gates),'invalidate_study')
        self.assertEqual(classify(True,claims,gates),'support_configured_network_on_tested_tasks')
        for field in ['topology_rewired','topology_reverse','feature']:
            self.assertEqual(classify(True,{**claims,field:False},gates),'support_network_not_exclusive_polarity')
        self.assertEqual(classify(True,{**claims,'tension':False},gates),'network_benefit_without_tension_specific_evidence')
        self.assertEqual(classify(True,{**claims,'network':False},gates),'no_confirmed_network_advantage')
        self.assertEqual(classify(True,claims,{**gates,'cost':False}),'no_confirmed_network_advantage')
        with self.assertRaises(ValueError):classify(True,{},gates)


if __name__=='__main__':unittest.main()
