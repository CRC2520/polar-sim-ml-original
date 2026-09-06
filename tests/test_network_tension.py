import json
import unittest
from dataclasses import replace
import numpy as np
from polar import ContextualPolarModel, ModelConfig, swap_poles
from network_tension import NetworkTensionModel


class NetworkTensionTests(unittest.TestCase):
    def setUp(self):
        self.cfg = ModelConfig(agents=1, types=4, seed=2026,
                               use_memory=False, use_self_model=False)
        self.shape = (1, 4, 2)
        self.W = np.zeros((8, 8)); self.K = np.zeros((8, 4))

    def obs(self, value=.2, **extra):
        return {"target": np.broadcast_to(value, self.shape).copy(),
                "budget": 8., **extra}

    def model(self, **extra):
        return NetworkTensionModel(self.cfg, **extra)

    def test_zero_graph_reproduces_original_with_learning(self):
        cfg = replace(self.cfg, use_memory=True, use_self_model=True)
        a, b = ContextualPolarModel(cfg), NetworkTensionModel(cfg)
        rng = np.random.default_rng(881)
        for t in range(12):
            o = self.obs(rng.uniform(0, 1, self.shape), observed=rng.random(self.shape)>.3,
                         cue=str(t%3), budget=2.5)
            aa, bb = a.act(o), b.act(o)
            np.testing.assert_array_equal(aa, bb)
            a.learn({"effect": .7*aa}); b.learn({"effect": .7*bb})

    def test_directed_state_edge_and_isolated_node(self):
        self.W[2,0] = .4
        a, b = self.model(state_coupling=self.W), self.model()
        initial = np.full(self.shape, .2); initial[0,0,0] = .8
        a.q = initial; b.q = initial
        delta = a.act(self.obs(initial)) - b.act(self.obs(initial))
        expected = np.zeros(self.shape); expected[0,1,0] = .65*.4*.8
        np.testing.assert_allclose(delta, expected, atol=1e-14)

    def test_tension_route_changes_recipient_with_identical_source_state(self):
        self.K[2,0] = .4
        chi = np.zeros((1,4)); chi[0,0] = 1
        a = self.model(tension_coupling=self.K, incompatibility=chi)
        b = self.model(tension_coupling=self.K)
        p = np.full(self.shape,.2); p[0,0]=[.8,.6]
        a.q=p; b.q=p
        delta=a.act(self.obs(p))-b.act(self.obs(p))
        self.assertAlmostEqual(delta[0,1,0],.65*.4*.8*.6)
        np.testing.assert_allclose(delta[0,[0,2,3]],0,atol=1e-14)

    def test_coactivation_not_necessarily_conflict_or_inactivity(self):
        a,b,c = self.model(),self.model(incompatibility=np.ones((1,4))),self.model()
        a.q=np.ones(self.shape); b.q=np.ones(self.shape)
        a.act(self.obs(1)); b.act(self.obs(1)); c.act(self.obs(0))
        for model,expected in [(a,0),(b,1),(c,0)]:
            np.testing.assert_allclose(model.last_trace['network']['tension'],expected)
        self.assertTrue(np.all(a.signed==c.signed))
        self.assertTrue(np.all(a.intensity!=c.intensity))

    def test_unmet_demand_tension_exists_without_coactivation(self):
        a=self.model(); a.act(self.obs(.8))
        np.testing.assert_allclose(a.last_trace['network']['tension'],.8)

    def test_inhibitory_tension_and_access_to_negative_pole(self):
        self.K[2,0]=-.4
        a=self.model(tension_coupling=self.K,incompatibility=np.ones((1,4)))
        a.q=np.full(self.shape,.5)
        out=a.act(self.obs(.5))
        self.assertLess(out[0,1,0],out[0,1,1])
        self.assertLess(a.signed[0,1],0)

    def test_multihop_delay_and_edge_lesion(self):
        self.W[2,0]=.3; self.W[4,2]=.4
        intact,lesion=self.model(state_coupling=self.W),self.model(state_coupling=self.W)
        intact.q=np.zeros(self.shape); lesion.q=np.zeros(self.shape)
        p=intact.q; p[0,0,0]=.5; intact.q=p
        first=intact.act(self.obs(0))-lesion.act(self.obs(0))
        self.assertGreater(first[0,1,0],0); self.assertEqual(first[0,2,0],0)
        for m in (intact,lesion): m.learn({"effect":m.last_action})
        second=intact.act(self.obs(0))-lesion.act(self.obs(0))
        self.assertGreater(second[0,2,0],0)
        np.testing.assert_allclose(second[0,3],0)

    def test_state_and_tension_routes_have_separate_lesions(self):
        self.W[2,0]=.2; self.K[3,0]=.3
        outputs=[]
        for W,K in [(self.W,self.K),(self.W,None),(None,self.K),(None,None)]:
            m=self.model(state_coupling=W,tension_coupling=K,incompatibility=np.ones((1,4)))
            m.q=np.full(self.shape,.5); outputs.append(m.act(self.obs(.5)))
        np.testing.assert_allclose(outputs[0]-outputs[3],
                                   outputs[1]+outputs[2]-2*outputs[3],atol=1e-14)

    def test_hard_constraints_after_strong_influences(self):
        self.W[2,0]=100; self.K[4,0]=100
        m=self.model(state_coupling=self.W,tension_coupling=self.K)
        m.q=np.ones(self.shape)
        mask=np.ones(self.shape,bool); mask[0,1,0]=False
        out=m.act(self.obs(.8,budget=.3,allowed=mask))
        self.assertEqual(out[0,1,0],0)
        self.assertTrue(np.all((out>=0)&(out<=1)))
        self.assertLessEqual(float(out.sum()),.3+1e-12)

    def test_trace_reconstructs_preprojection_and_feedback(self):
        self.W[2,0]=.2; self.K[3,0]=.3
        m=self.model(state_coupling=self.W,tension_coupling=self.K)
        action=m.act(self.obs(.7)); m.learn({"effect":action})
        n=m.last_trace['network']
        expected=np.asarray(n['base_proposal'])+np.asarray(n['eta'])*(np.asarray(n['state_term'])+np.asarray(n['tension_term']))
        np.testing.assert_allclose(expected,n['network_proposal'])
        self.assertIsNotNone(m.last_trace['feedback'])
        json.dumps(m.last_trace,allow_nan=False)

    def test_pole_relabel_transforms_routes_and_learned_state(self):
        rng=np.random.default_rng(14)
        cfg=replace(self.cfg,use_memory=True,use_self_model=True)
        self.W[2,0]=.2; self.W[7,2]=-.1; self.K[5,0]=.3
        a,b=[NetworkTensionModel(cfg,state_coupling=self.W,tension_coupling=self.K) for _ in range(2)]
        for _ in range(3):
            o=self.obs(rng.uniform(.1,.8,self.shape),cue='x')
            for m in (a,b):
                u=m.act(o);m.learn({'effect':u*.8})
        b.reverse_convention([0,2])
        for _ in range(4):
            o=self.obs(rng.uniform(.1,.8,self.shape),cue='x',observed=rng.random(self.shape)>.3)
            ob={k:swap_poles(v,[0,2]) if isinstance(v,np.ndarray) else v for k,v in o.items()}
            aa,bb=a.act(o),b.act(ob)
            np.testing.assert_allclose(bb,swap_poles(aa,[0,2]),atol=1e-13)
            a.learn({'effect':aa*.8});b.learn({'effect':bb*.8})

    def test_signed_intensity_is_equivalent_with_network(self):
        self.W[2,0]=.2;self.K[3,0]=.3
        a,b=[NetworkTensionModel(replace(self.cfg,representation=r),state_coupling=self.W,tension_coupling=self.K) for r in ('dual_pole','signed_intensity')]
        for _ in range(5):
            aa,bb=a.act(self.obs(.7)),b.act(self.obs(.7))
            np.testing.assert_allclose(aa,bb,atol=1e-13)
            a.learn({'effect':aa});b.learn({'effect':bb})

    def test_hidden_inputs_cannot_create_tension(self):
        self.K[2,0]=.5
        a,b=[self.model(tension_coupling=self.K) for _ in range(2)]
        mask=np.zeros(self.shape,bool)
        np.testing.assert_array_equal(a.act(self.obs(0,observed=mask)),b.act(self.obs(1,observed=mask)))

    def test_validation_and_feedback_order(self):
        with self.assertRaises(ValueError):self.model(state_coupling=np.eye(8))
        with self.assertRaises(ValueError):self.model(tension_coupling=np.full((8,4),np.nan))
        with self.assertRaises(ValueError):self.model(incompatibility=np.full((1,4),2))
        m=self.model();m.act(self.obs())
        with self.assertRaises(RuntimeError):m.act(self.obs())
        with self.assertRaises(RuntimeError):m.set_couplings()
        with self.assertRaises(RuntimeError):m.set_incompatibility()
        with self.assertRaises(RuntimeError):m.reverse_convention()


if __name__=='__main__':unittest.main()
