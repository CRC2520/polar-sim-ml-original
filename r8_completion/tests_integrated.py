"""Mechanistic tests of R8 integration, chronology, credit and observation limits."""
from pathlib import Path
import copy
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from r8_completion import integrated as i


class IntegratedContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = i.IntegratedAgent(940151)
        cls.trace,_ = i.rollout(cls.agent,0,learn=True,steps=100)
        cls.fit_data = i.fit_agent(cls.agent,[cls.trace])

    def test_targets_are_own_future_at_exact_offsets(self):
        sample = i.trajectory_samples(self.trace)
        for row in (0,7,32,70):
            t,g,a = sample["time_index"][row],sample["group_index"][row],sample["individual_index"][row]
            for j,horizon in enumerate(i.PROTOCOL["horizons"]):
                np.testing.assert_array_equal(sample["future"][row,j],self.trace["public_sequence"][t+horizon,g,a])
        self.assertLess(sample["time_index"].max()+max(i.PROTOCOL["horizons"]),len(self.trace["public_sequence"]))

    def test_integrated_physics_matches_preserved_engine_across_delay_domains(self):
        for domain in i.PROTOCOL["domains"]:
            fresh = i.IntegratedAgent(940157)
            trace,_ = i.rollout(fresh,912,domain,learn=False,steps=14)
            from dataclasses import replace
            engine = i.BridgeEngine(replace(i.config(domain),steps=14),[940157],.5,study="C",variant="full")
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary)/"engine.npz"
                engine.episode(912,learn=False,trace_path=path,trace_mode="full")
                with np.load(path,allow_pickle=False) as expected:
                    for left,right in (("action","action"),("energy_after","vitality_after"),
                                       ("resource_after_DIAGNOSTIC","resource_after"),("alive_after","alive_after")):
                        np.testing.assert_array_equal(trace[left],expected[right][0])

    def test_hidden_diagnostics_do_not_enter_training_features_or_targets(self):
        changed = {key:value.copy() for key,value in self.trace.items()}
        for key in changed:
            if "DIAGNOSTIC" in key:
                changed[key].fill(999.)
        original,other = i.trajectory_samples(self.trace),i.trajectory_samples(changed)
        for key in original:
            np.testing.assert_array_equal(original[key],other[key])

    def test_miscredit_keeps_actual_subsample_context_target_and_action_multisets(self):
        data = self.fit_data
        before = {key:value.copy() for key,value in data.items()}
        wrong,source_time = i.miscredit_sampled_actions(data)
        keys = np.stack((data["episode_index"],data["group_index"],data["individual_index"]),-1)
        for key in np.unique(keys,axis=0):
            ids = np.flatnonzero(np.all(keys==key,axis=1))
            np.testing.assert_array_equal(np.sort(data["action"][ids]),np.sort(wrong[ids]))
            for ix in ids:
                source = ids[data["time_index"][ids]==source_time[ix]][0]
                self.assertEqual(wrong[ix],data["action"][source])
        for key in data:
            np.testing.assert_array_equal(data[key],before[key])

    def test_credit_multisets_survive_the_actual_subsampling_branch(self):
        agent = copy.deepcopy(self.agent)
        with patch.dict(i.PROTOCOL,{"max_fit_samples":100}):
            data = i.fit_agent(agent,[self.trace])
        self.assertEqual(len(data["action"]),100)
        keys = np.stack((data["episode_index"],data["group_index"],data["individual_index"]),-1)
        for key in np.unique(keys,axis=0):
            ids = np.flatnonzero(np.all(keys==key,axis=1))
            np.testing.assert_array_equal(np.sort(data["action"][ids]),np.sort(data["miscredited_action"][ids]))

    def test_all_training_origins_are_factual_uniform_probes(self):
        self.assertTrue(self.fit_data["eligible"].all())
        self.assertTrue(self.fit_data["alive"].all())
        self.assertGreaterEqual(np.bincount(self.fit_data["action"],minlength=4).min(),3)

    def test_next_poles_equal_next_observed_poles(self):
        np.testing.assert_array_equal(self.trace["next_poles"][:-1],self.trace["poles"][1:])

    def test_tension_uses_previous_prediction_not_future(self):
        expected = np.mean(np.abs(self.trace["previous_prediction"]-self.trace["poles"]),axis=-1)
        expected += .2*self.trace["poles"][...,0]*self.trace["poles"][...,1]
        np.testing.assert_array_equal(expected,self.trace["tension"])
        np.testing.assert_array_equal(self.trace["previous_prediction"][1:],self.trace["issued_prediction"][:-1])

    def test_all_eight_pairs_allow_coactivation(self):
        rng = np.random.default_rng(940154)
        h = rng.random((200,16,4))
        h[...,2] = rng.integers(0,4,(200,16))
        h[:100,::3,2] = 0
        h[...,3] *= .3
        x = i.base_features(rng.uniform(.1,.9,200),rng.uniform(.2,.7,200),h)
        poles = i.encode_poles(x,h,np.zeros(200))
        coactive = (poles[...,0]>.02)&(poles[...,1]>.02)
        self.assertTrue(coactive.any(axis=0).all())
        self.assertFalse(np.allclose(poles.sum(-1),1.))

    def test_no_history_intervention_removes_lag_information(self):
        rng = np.random.default_rng(940155)
        h1,h2 = rng.random((16,16,4)),rng.random((16,16,4))
        r,e = np.full(16,.4),np.full(16,.6)
        np.testing.assert_array_equal(i.base_features(r,e,h1,True),i.base_features(r,e,h2,True))
        self.assertFalse(np.array_equal(i.base_features(r,e,h1),i.base_features(r,e,h2)))

    def test_no_ecology_retains_identical_viability_guard_and_network(self):
        x,poles,tau,state = (self.fit_data[key][:20] for key in ("features","poles","tension","state_index"))
        # Actor Q belongs to group/individual slots, so use a real [2,8] frame.
        x,poles,tau,state = (self.trace[key][2] for key in ("features","poles","tension","state_index"))
        full = self.agent.evaluate(x,poles,tau,state,"full",ecology_weight=8.,gate=1.)
        off = self.agent.evaluate(x,poles,tau,state,"noEcology",ecology_weight=8.,gate=1.)
        np.testing.assert_array_equal(full["feasible"],off["feasible"])
        np.testing.assert_array_equal(full["network_advantage"],off["network_advantage"])
        np.testing.assert_allclose(full["unguarded_scores"]-off["unguarded_scores"],8.*full["eco_advantage"],atol=1e-12)

    def test_network_coupling_enters_actual_actor_scores(self):
        x,poles,tau,state = (self.trace[key][2] for key in ("features","poles","tension","state_index"))
        full = self.agent.evaluate(x,poles,tau,state,"full",gate=1.)
        off = self.agent.evaluate(x,poles,tau,state,"noNetwork",gate=1.)
        self.assertGreater(np.max(np.abs(full["unguarded_scores"]-off["unguarded_scores"])),1e-8)
        np.testing.assert_allclose(full["unguarded_scores"]-off["unguarded_scores"],
            4.*(full["network_advantage"]-off["network_advantage"]),atol=1e-12)

    def test_native_evaluation_freezes_q_visits_predictor_and_graph(self):
        before = {key:self.agent.base.state[key].copy() for key in ("qt","qv","qg","visits")}
        head = self.agent.head.coef.copy()
        graph = self.agent.network.parameter_digest()
        i.rollout(self.agent,900,learn=False,steps=45)
        for key in before:
            np.testing.assert_array_equal(before[key],self.agent.base.state[key])
        np.testing.assert_array_equal(head,self.agent.head.coef)
        self.assertEqual(graph,self.agent.network.parameter_digest())

    def test_native_policy_respects_guard_while_diagnostic_probes_are_separate(self):
        native,_ = i.rollout(self.agent,905,learn=False,steps=45)
        selected = np.take_along_axis(native["feasible"],native["action"][...,None],axis=-1)[...,0]
        self.assertTrue(selected[native["alive_before"]].all())
        self.assertFalse(native["uniform_probe"].any())
        diagnostic,_ = i.rollout(self.agent,906,learn=False,steps=45,probe_rate=.15)
        self.assertTrue(diagnostic["uniform_probe"].any())

    def test_all_q_targets_use_the_same_successor_action(self):
        agent = i.IntegratedAgent(940156)
        heads = ([0.,1.,2.,3.],[3.,0.,1.,2.],[2.,3.,0.,1.])
        for key,value in zip(("qt","qv","qg"),heads):
            agent.base.state[key][:] = np.asarray(value)
        trace,_ = i.rollout(agent,910,learn=True,steps=1)
        expected = trace["rewards"][0].copy()
        successor = trace["successor_action"][0]
        for head,gamma in enumerate((.5,.97,.97)):
            expected[...,head] += gamma*np.asarray(heads[head])[successor]*trace["alive_after"][0]
        np.testing.assert_array_equal(expected,trace["td_targets"][0])
        self.assertTrue(np.all(successor==3))

    def test_checkpoint_roundtrip_replays_exactly_without_pickle(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"agent.npz"
            i.save_agent(self.agent,path)
            restored = i.load_agent(path)
            t1,r1 = i.rollout(self.agent,901,learn=False,steps=45)
            t2,r2 = i.rollout(restored,901,learn=False,steps=45)
            self.assertEqual(r1,r2)
            for key in t1:
                np.testing.assert_array_equal(t1[key],t2[key])


if __name__ == "__main__":
    unittest.main()
