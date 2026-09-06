import unittest
from copy import deepcopy
import numpy as np
from integrated_polar import IntegratedAgent,AgentConfig
from integrated_polar.numerics import CoupledRLS,solve_plan,own_pair_mask
from integrated_polar.projection import project
from integrated_polar.layers import Content,ContentWorkspace,GatedMemory,CalibratedMonitor,SemanticAdapter,infer_effect_source
from integrated_polar.agent import IndependentResourceAgent,coordinate_offers
from gap_resolution.p0 import DiagnosticController
from study3.core import environment,trial,MODES,TUNED


def obs(n=1,target=.8,priority=1.,cue='A',event=True,**extra):
    r=np.broadcast_to(target,(3,n)).copy();w=np.broadcast_to(priority,(3,n)).copy()
    o={'state':np.full(n,.35),'cue':cue,'budget':2*n,'costs':np.ones(2*n)}
    if event:o['event']={'targets':r,'priorities':w,'identity':cue+'-1','version':1}
    o.update(extra);return o


def consume(a,o):
    u=a.act(o);y=.9*np.asarray(o['state'])+.03+a.model.predict(u)
    a.learn({'state':y,'token':deepcopy(a.pending['token'])})
    return u


class IntegratedGapTests(unittest.TestCase):
    def agent(self,mode='full',n=1,**kwargs):return IntegratedAgent(AgentConfig(tanks=n,mode=mode,**kwargs))
    def test_G01_internal_intention_survives_blocked_action(self):
        a=self.agent();u=consume(a,obs(allowed=np.array([False,True])))
        self.assertEqual(u[0],0);self.assertGreater(a.p[0],.01);self.assertGreater(a.intention[0],.1)
        self.assertNotEqual(a.p[0],u[0])
    def test_G01_internal_state_is_causal_not_only_telemetry(self):
        a,b=self.agent(),self.agent();a.p[:]=[.8,0.]
        ua,ub=a.act(obs(target=.36)),b.act(obs(target=.36))
        self.assertGreater(float(np.max(abs(ua-ub))),1e-5)
    def test_G02_opposite_deficits_are_not_merged(self):
        a,b=self.agent(),self.agent();a.act(obs(target=.6));b.act(obs(target=.1))
        ea=np.asarray(a.trace['planning']['signed_residual']);eb=np.asarray(b.trace['planning']['signed_residual'])
        self.assertGreater(ea[0,0],0);self.assertLess(eb[0,0],0)
        self.assertIn('coactivation',a.trace['planning'])
    def test_G03_receiving_poles_have_opposite_learned_sensitivities(self):
        a=self.agent();a.act(obs());K=np.asarray(a.trace['planning']['K_effective'])
        self.assertGreater(K[0,0],0);self.assertLess(K[1,0],0)
    def test_G04_G05_identifies_individual_directed_coefficients(self):
        rng=np.random.default_rng(791201);B=rng.uniform(-.2,.2,(3,6));m=CoupledRLS(3,6,ridge=1e-7,forgetting=1)
        for _ in range(150):
            u=rng.uniform(0,1,6);m.update(u,B@u)
        np.testing.assert_allclose(m.B,B,atol=1e-7)
    def test_G05_planning_lesion_retains_estimator_but_changes_decision(self):
        a,b=self.agent(n=2),self.agent('diagonal_plan',n=2)
        a.model.B[1,0]=.15;b.model.B=a.model.B.copy()
        ua,ub=a.act(obs(2,target=[.65,.25])),b.act(obs(2,target=[.65,.25]))
        np.testing.assert_array_equal(a.model.B,b.model.B)
        self.assertGreater(np.max(abs(ua-ub)),.01)
    def test_G06_priorities_change_actual_allocation(self):
        a,b=self.agent(n=2),self.agent(n=2)
        ua=a.act(obs(2,target=.75,priority=[9.,1.],budget=.5))
        ub=b.act(obs(2,target=.75,priority=[1.,9.],budget=.5))
        self.assertGreater(ua[0],ua[2]);self.assertGreater(ub[2],ub[0]);self.assertLessEqual(ua.sum(),.5+1e-10)
    def test_G06_weighted_projection_matches_analytic_solution(self):
        np.testing.assert_allclose(project(np.array([.8,.8]),np.ones(2),.6,weights=[9.,1.]),[.6,0],atol=1e-12)
    def test_G07_selective_content_cut_preserves_other_consumers(self):
        a,b=self.agent(),self.agent();b.workspace.cut.add('planner')
        ua,ub=a.act(obs(target=.2)),b.act(obs(target=.2))
        self.assertGreater(np.max(abs(ua-ub)),.01)
        self.assertIsNotNone(b.trace['reporter_content']);self.assertIn('memory',b.trace['C']['delivered']);self.assertNotIn('planner',b.trace['C']['delivered'])
    def test_G07_content_changes_not_budget_only(self):
        a,b=self.agent(n=2),self.agent(n=2)
        ua=a.act(obs(2,target=[.8,.2]));ub=b.act(obs(2,target=[.2,.8]))
        self.assertGreater(np.max(abs(ua-ub)),.1);self.assertEqual(a.trace['budget'],b.trace['budget'])
    def test_G08_cue_only_recall_erasure_and_gate_are_causal(self):
        a=self.agent();consume(a,obs(target=.2));b,c=deepcopy(a),deepcopy(a)
        for m in (a,b,c):m.p.fill(0);m.goals.reset(None)
        b.erase_memory('A')
        ua=a.act(obs(event=False));ub=b.act(obs(event=False));uc=c.act(obs(event=False,memory_gate=False))
        self.assertGreater(np.max(abs(ua-ub)),.01);np.testing.assert_allclose(ub,uc,atol=1e-12)
    def test_G08_reconsolidation_version_and_permission(self):
        a=self.agent();consume(a,obs());b=deepcopy(a)
        with self.assertRaises(PermissionError):b.memory.reconsolidate('A',[[.1]]*3,authorized=False)
        b.memory.reconsolidate('A',[[.1]]*3,authorized=True)
        for m in (a,b):m.goals.reset(None);m.p.fill(0)
        ua,ub=a.act(obs(event=False)),b.act(obs(event=False))
        self.assertGreater(ua[0],ub[0]);self.assertGreater(ub[1],ua[1])
        self.assertEqual(b.memory.records['A']['content'].version,2)
    def test_G08_private_record_not_released_without_permission(self):
        mem=GatedMemory();mem.store(Content('p','A',((.7,),)*3,((1.,),)*3,private=True))
        self.assertIsNone(mem.recall('A'));self.assertIsNotNone(mem.recall('A',permit_private=True))
    def test_G09_prohibition_overrides_requested_action_with_reason(self):
        a=self.agent();u=a.act(obs(target=.1,normative={'protected':np.array([True]),'approved_drain':False,'mandatory_min':[0,.4]}))
        self.assertEqual(u[1],0);rules=[r['rule'] for r in a.trace['E']['reasons']]
        self.assertIn('E-PROTECT',rules);self.assertIn('E-PRECEDENCE',rules)
    def test_G09_authorization_changes_action_not_just_flag(self):
        a,b=self.agent(),self.agent()
        ua=a.act(obs(target=.1,normative={'protected':np.array([True]),'approved_drain':False}))
        ub=b.act(obs(target=.1,normative={'protected':np.array([True]),'approved_drain':True}))
        self.assertEqual(ua[1],0);self.assertGreater(ub[1],.01)
    def test_G09_infeasible_obligation_is_explicit_halt(self):
        a=self.agent();u=a.act(obs(budget=.1,normative={'mandatory_min':[.3,0]}))
        self.assertTrue(a.trace['E']['halt']);np.testing.assert_array_equal(u,[0,0])
        self.assertTrue(any(r['rule']=='E-INFEASIBLE' for r in a.trace['E']['reasons']))
    def test_G10_forecast_changes_action_before_goal_arrives(self):
        a,b=self.agent(),self.agent('myopic');o=obs(target=[[.345],[.8],[.8]])
        ua,ub=a.act(o),b.act(o)
        self.assertGreater(ua[0],ub[0]+.01)
    def test_G10_commitment_persists_without_repeated_event(self):
        a=self.agent();consume(a,obs());a.memory.erase('A')
        a.act(obs(event=False))
        np.testing.assert_array_equal(a.goals.targets,[[.8]]*3)
    def test_G11_feedback_provenance_and_duplicate_guard(self):
        a=self.agent();a.act(obs());token=deepcopy(a.pending['token']);before=a.model.B.copy()
        with self.assertRaises(ValueError):a.learn({'state':[.5],'token':{**token,'actor':'other'}})
        np.testing.assert_array_equal(before,a.model.B)
        a.learn({'state':[.5],'token':token})
        with self.assertRaises(RuntimeError):a.learn({'state':[.5],'token':token})
    def test_G11_source_inference_without_source_labels_and_ambiguity(self):
        result=infer_effect_source([.2,-.1],{'A':[.2,-.1],'B':[-.1,.2]})
        self.assertEqual(result['source'],'A')
        self.assertIsNone(infer_effect_source([.2,-.1],{'A':[.2,-.1],'B':[.2,-.1]})['source'])
    def test_G11_empirical_monitor_and_behavioral_review(self):
        mon=CalibratedMonitor().fit(np.repeat([0.,1.],30),np.repeat([1,0],30))
        self.assertGreater(mon.probability(0),mon.probability(1))
        low=CalibratedMonitor.restore({'edges':[0.],'probabilities':[0.,0.],'base':0.})
        a=self.agent();b=IntegratedAgent(AgentConfig(tanks=1,meta_threshold=.55),low)
        ua,ub=a.act(obs()),b.act(obs())
        self.assertGreater(ua.sum(),0);self.assertEqual(ub.sum(),0);self.assertTrue(b.trace['requested_review'])
    def test_G12_independent_agents_and_bounded_coordination(self):
        a,b=IndependentResourceAgent('A'),IndependentResourceAgent('B')
        self.assertIsNot(a.controller.model,b.controller.model)
        offers=[a.offer(.2,.8,5),b.offer(.4,.5,1)]
        allocation=coordinate_offers(offers,.6);self.assertGreater(allocation['A'],allocation['B']);self.assertLessEqual(sum(allocation.values()),.6+1e-12)
        ua=a.act(obs(target=.8),allocation['A']);ub=b.act(obs(target=.1),allocation['B'])
        self.assertGreater(ua[0],ua[1]);self.assertGreater(ub[1],ub[0])
        self.assertLessEqual(ua.sum()+ub.sum(),.6+1e-10)
    def test_G13_typed_grammar_and_unknown_symbol_rejection(self):
        a=SemanticAdapter(2,3);c=a.parse('tank_1 = 0.7 @ 4; tank_2 = 0.2 @ 1','A')
        self.assertEqual(c.targets[0],(.7,.2));self.assertEqual(c.priorities[0],(4.,1.))
        for text in ['tank_3 = 0.2 @ 1','tank_1 = 0.7 @ 1; tank_1 = 0.2 @ 2','power = 0.8 @ 1']:
            with self.assertRaises(ValueError):a.parse(text,'A')
    def test_G13_channel_relabel_equivariance(self):
        B=np.array([[.1,-.1,.03,-.02],[.01,-.02,.1,-.1]]);p=np.array([.1,.2,.3,.1]);y=np.array([.2,.3]);r=np.array([[.7,.4]]*3);w=np.array([[2.,1.]]*3)
        u,_=solve_plan(B,y,r,w,.9,np.array([.03,.03]),p,np.ones(4),[1.]*3,np.ones(4,bool),iterations=48)
        perm=np.array([2,3,0,1]);out,_=solve_plan(B[[1,0]][:,perm],y[[1,0]],r[:,[1,0]],w[:,[1,0]],.9,np.array([.03,.03]),p[perm],np.ones(4),[1.]*3,np.ones(4,bool),iterations=48)
        np.testing.assert_allclose(out,u[:,perm],atol=1e-9)
    def test_G14_closed_loop_all_layers_and_finite_traces(self):
        import json
        a=self.agent();state=np.array([.35])
        for t in range(12):
            o=obs(target=.7);o['state']=state;o['event']['version']=t+1
            u=a.act(o);state=np.clip(.9*state+.03+np.array([[.12,-.1]])@u,0,1)
            a.learn({'state':state,'token':deepcopy(a.pending['token'])})
            json.dumps(a.trace,allow_nan=False);self.assertIsNotNone(a.trace['feedback'])
            self.assertLessEqual(u.sum(),o['budget']+1e-10)
    def test_G14_missing_state_and_hidden_truth_are_not_substituted(self):
        with self.assertRaises(ValueError):self.agent().act({'cue':'A'})
        a,b=self.agent(),self.agent();o=obs()
        np.testing.assert_array_equal(a.act(o),b.act({**o,'true_B':[[99,99]],'answer_key':[1]}))
    def test_G14_generic_matrix_formula_preserves_actions(self):
        a,b=self.agent(),self.agent('generic_equivalent')
        for t in range(4):
            o=obs(target=.65);o['event']['version']=t+1
            np.testing.assert_allclose(consume(a,o),consume(b,o),atol=1e-9,rtol=0)
    def test_G14_censored_feedback_does_not_update_invalid_row(self):
        a=self.agent();a.act(obs());B=a.model.B.copy();a.learn({'state':[1.],'valid_transition':np.array([False]),'token':deepcopy(a.pending['token'])})
        np.testing.assert_array_equal(B,a.model.B)
    def test_P0_zero_factors_reproduce_original_active_network(self):
        from study3.core import build
        pars={m:{'eta':1.,'a':0.,'b':-.2} for m in TUNED}
        a=DiagnosticController((0,0,0,0),791991);b=build('full',pars,791991)
        e=environment(791991,('linear','aligned','scarce'))
        from study3.core import transition
        for t in range(8):
            o={k:e[k][t] for k in ('target','weights','costs','allowed')};o.update(budget=float(e['budget'][t]),cue=str(t//16),observed=True)
            for m in (a,b):m.set_incompatibility(e['chi'][t])
            ua,ub=a.act(o),b.act(o);np.testing.assert_array_equal(ua,ub)
            for m,u in [(a,ua),(b,ub)]:m.learn({'effect':transition(e,t,u)})


if __name__=='__main__':unittest.main()
