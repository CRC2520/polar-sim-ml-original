import copy
import json
import unittest
import numpy as np

from study2.controllers import ControllerConfig, CoupledController, project_action


def observation(state=None, **changes):
    result = {"state": np.zeros(8) if state is None else np.asarray(state),
              "observed": np.ones(8, dtype=bool), "target": np.full(8, .3),
              "weights": np.ones(8), "costs": np.ones(8), "allowed": np.ones(8, dtype=bool),
              "budget": 5., "persistence": .55, "horizon": 3, "drift": np.zeros(8)}
    result.update(changes)
    return result


class ControllerTests(unittest.TestCase):
    def test_missing_nonfinite_and_uninitialized_hidden_state_are_explicit_errors(self):
        model = CoupledController()
        for obs in ({}, observation(state=np.full(8, np.nan)),
                    observation(observed=np.zeros(8, dtype=bool))):
            with self.assertRaises(ValueError):
                model.act(obs)
        with self.assertRaises(ValueError):
            model.act(observation(persistence=1))
        with self.assertRaises(ValueError):
            model.act(observation(costs=np.zeros(8)))

    def test_projection_obeys_budget_blocking_and_kkt_conditions(self):
        rng = np.random.default_rng(44)
        for _ in range(60):
            v, c = rng.normal(.8, .8, 8), rng.uniform(.3, 2, 8)
            allowed = rng.random(8) > .2
            budget = rng.uniform(.1, 3)
            action = project_action(v, c, budget, allowed)
            self.assertTrue(np.all((action >= 0) & (action <= 1)))
            self.assertTrue(np.all(action[~allowed] == 0))
            self.assertLessEqual(float(c @ action), budget + 1e-10)
            interior = allowed & (action > 1e-9) & (action < 1-1e-9)
            if np.sum(interior) > 1 and abs(c @ action - budget) < 1e-8:
                thresholds = (v[interior] - action[interior]) / c[interior]
                self.assertLess(np.ptp(thresholds), 1e-8)

    def test_feedback_pairs_with_own_action_and_prediction_precedes_learning(self):
        model = CoupledController()
        action = model.act(observation())
        expected_prediction = .45 * .5 * action
        np.testing.assert_allclose(model.last_trace["predicted_next_state"], expected_prediction)
        with self.assertRaises(RuntimeError):
            model.act(observation())
        model.learn({"state": np.ones(8)*.4, "observed": np.ones(8, dtype=bool)})
        np.testing.assert_allclose(model.last_trace["predicted_next_state"], expected_prediction)
        self.assertGreater(np.max(np.abs(model.B - .5*np.eye(8))), .01)
        with self.assertRaises(RuntimeError):
            model.learn({"state": np.ones(8), "observed": np.ones(8, dtype=bool)})
        json.dumps(model.last_trace, allow_nan=False)

    def test_hidden_truth_keys_and_masked_values_cannot_change_action_or_learning(self):
        first, second = CoupledController(), CoupledController()
        first.act(observation())
        second.act(observation(true_matrix=np.full((8,8), 999), answer_key="inaccessible"))
        mask = np.array([True, False]*4)
        first.learn({"state": np.where(mask,.2,0), "observed": mask})
        second.learn({"state": np.where(mask,.2,999), "observed": mask, "true_matrix": np.eye(8)*999})
        np.testing.assert_array_equal(first.B, second.B)
        a = first.act(observation(np.where(mask,.2,0), observed=mask))
        b = second.act(observation(np.where(mask,.2,-999), observed=mask, true_matrix=np.ones((8,8))))
        np.testing.assert_array_equal(a, b)
        self.assertIsNone(second.last_trace["observation"]["state"][1])

    def test_planning_lesion_changes_action_with_same_estimated_matrix_and_capacity(self):
        coupled = CoupledController()
        coupled.B[0,1] = .4
        coupled.B[1,0] = -.3
        lesion = copy.deepcopy(coupled)
        lesion.set_coupling_enabled(False)
        obs = observation(target=np.array([.1,.6,.2,.4,.3,.1,.4,.2]))
        a, b = coupled.act(obs), lesion.act(obs)
        self.assertGreater(np.linalg.norm(a-b), .1)
        np.testing.assert_array_equal(coupled.B, lesion.B)
        self.assertEqual(coupled.profile, lesion.profile)
        self.assertEqual(lesion.last_trace["planning_matrix"][0][1], 0)
        self.assertEqual(lesion.last_trace["estimated_matrix"][0][1], .4)

    def test_signed_intensity_recoding_preserves_full_closed_loop(self):
        first = CoupledController()
        second = CoupledController(ControllerConfig(mode="signed_intensity"))
        rng = np.random.default_rng(51)
        matrix = np.eye(8)*.7
        for pair in range(4):
            matrix[2*pair,2*pair+1] = -.2
            matrix[2*pair+1,2*pair] = .25
        states = [np.zeros(8), np.zeros(8)]
        for step in range(35):
            target, weights = rng.uniform(0,.5,8), rng.uniform(.4,2,8)
            actions = []
            for index, model in enumerate((first, second)):
                action = model.act(observation(states[index], target=target, weights=weights))
                states[index] = .55*states[index] + .45*matrix@action
                model.learn({"state": states[index], "observed": np.ones(8,dtype=bool)})
                actions.append(action)
            np.testing.assert_allclose(actions[0], actions[1], atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(first.B, second.B, atol=1e-12)

    def test_pole_relabel_conjugates_learned_relations_and_preserves_behavior(self):
        rng = np.random.default_rng(59)
        first = CoupledController()
        first.B[0,1], first.B[1,0], first.B[4,5] = .3,-.2,.4
        first.memory_state = rng.normal(size=8)
        first._remember_action(rng.uniform(0,1,8))
        second = copy.deepcopy(first)
        order = np.array([1,0,2,3,5,4,6,7])
        second.relabel(order)
        matrix = first.B.copy()
        state = rng.normal(size=8)
        for _ in range(8):
            obs = observation(state, target=rng.uniform(0,.5,8), weights=rng.uniform(.3,2,8),
                              costs=rng.uniform(.5,1.5,8), drift=rng.uniform(-.01,.01,8))
            alternate = {k:(v[order] if isinstance(v,np.ndarray) else v) for k,v in obs.items()}
            action, other_action = first.act(obs), second.act(alternate)
            np.testing.assert_allclose(action[order],other_action,atol=1e-11)
            state = .55*state + .45*matrix@action + obs["drift"]
            first.learn({"state":state,"observed":np.ones(8,dtype=bool)})
            second.learn({"state":state[order],"observed":np.ones(8,dtype=bool)})
            np.testing.assert_allclose(second.B,first.B[np.ix_(order,order)],atol=1e-11)

    def test_censored_inventory_rows_skip_identification_but_update_memory(self):
        model = CoupledController()
        action = model.act(observation(state_lower=np.zeros(8),state_upper=np.ones(8)*.2))
        self.assertTrue(np.all(np.asarray(model.last_trace["predicted_next_state"])<=.2))
        before = model.B.copy()
        valid = np.array([False,True]*4)
        model.learn({"state":np.where(valid,.1,.2),"observed":np.ones(8,dtype=bool),"transition_valid":valid})
        np.testing.assert_array_equal(model.B[~valid],before[~valid])
        np.testing.assert_array_equal(model.memory_state,np.where(valid,.1,.2))
        self.assertEqual(model.last_trace["feedback"]["update_counts"],[0,1]*4)

    def test_own_transition_identification_recovers_coupling_and_sign_shift(self):
        model = CoupledController(ControllerConfig(forgetting=.94))
        rng = np.random.default_rng(78)
        state = np.zeros(8)
        matrix = np.eye(8)*.7
        for pair in range(4):
            matrix[2*pair,2*pair+1] = .22
            matrix[2*pair+1,2*pair] = -.18
        for step in range(300):
            if step == 150:
                matrix[0,1] = -.25
            action = model.act(observation(state,target=rng.uniform(0,.7,8),horizon=1,budget=8))
            state = .55*state+.45*matrix@action
            model.learn({"state":state,"observed":np.ones(8,dtype=bool)})
            if step == 149:
                self.assertGreater(model.B[0,1],.15)
        self.assertLess(model.B[0,1],-.20)
        self.assertLess(np.max(np.abs(model.B-matrix)),.03)


if __name__ == "__main__":
    unittest.main()
