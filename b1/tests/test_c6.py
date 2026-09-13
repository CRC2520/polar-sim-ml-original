"""Exact conjugacy and corruption sensitivity on finite deterministic fixtures."""
from copy import deepcopy
from fractions import Fraction
import unittest

from b1.controllers import Controller, C6Controller, ConjugatedTask, encode_tree, decode_tree
from b1.controllers.c6 import encode_pair, decode_pair, feasible_pair
from b1.tasks import P5Task, P6Task, P7Task


class ConjugacyTests(unittest.TestCase):
    def test_four_regimes_and_intermediate_grid_exact(self):
        for a, b in [(0, 0), (1, 0), (0, 1), (1, 1), (Fraction(1, 2), 1)]:
            pair = encode_pair(a, b)
            self.assertEqual(decode_pair(pair), (a, b))
            self.assertTrue(feasible_pair(pair, (2, 2), admitted=True))
        self.assertFalse(feasible_pair(encode_pair(Fraction(1, 3), 0), (2, 2), admitted=True))
        self.assertTrue(feasible_pair(encode_pair(Fraction(1, 3), 0), (2, 2)))
        self.assertFalse(feasible_pair({"d": 0, "q": 3}, (2, 2)))
        self.assertFalse(feasible_pair({"d": 2, "q": 1}, (2, 2)))

    def test_transforms_parameters_memory_nested_actions_and_identity_fields(self):
        tree = {"parameters": {"A": Fraction(1, 3), "B": Fraction(4, 5)},
                "state": {"memory": [{"A": 0, "B": 1, "target_version": 7}]},
                "observation": {"stock": 8, "permissions": {"A": True, "B": False}},
                "tuple": (0, 1), "list": [0, 1]}
        encoded = encode_tree(tree)
        self.assertEqual(decode_tree(encoded), tree)
        encoded["items"][0][1]["coordinates"]["q"] += 1
        self.assertNotEqual(decode_tree(encoded), tree)  # Fault must be detectable.

    def test_full_controller_and_environment_transition_conjugacy(self):
        factories = {
            "P5": lambda i: P5Task(initial_reserve=0 if i == 0 else 8, cell_id=i),
            "P6": lambda i: P6Task(initial_mapping="q1", cell_id=i),
            "P7": lambda i: P7Task(initial_drifts=(1, 0), version_switch_epoch=1, cell_id=i),
        }
        for pilot, factory in factories.items():
            with self.subTest(pilot=pilot):
                original_tasks = [factory(i) for i in range(3)]
                recoded_tasks = [ConjugatedTask(deepcopy(t)) for t in original_tasks]
                original = Controller(pilot, "C1")
                recoded = C6Controller(deepcopy(original))
                for _ in range(4):
                    obs = [t.observe() for t in original_tasks]
                    encoded_obs = [t.observe_encoded() for t in recoded_tasks]
                    self.assertEqual([decode_tree(o) for o in encoded_obs], obs)
                    actions = original.act(obs)
                    transformed_actions = recoded.act_encoded(encode_tree(obs))
                    self.assertEqual(decode_tree(transformed_actions), actions)
                    self.assertEqual(decode_tree(recoded.state), original.__dict__)
                    for task, image_task, action in zip(original_tasks, recoded_tasks, actions):
                        event = task.step(action)
                        image_event = image_task.step_encoded(encode_tree(action))
                        self.assertEqual(decode_tree(image_event), event)
                        self.assertEqual(decode_tree(image_task._state), task.__dict__)
                self.assertEqual(recoded.last_decision_report["tolerance_absolute"], "0")
                self.assertGreater(recoded.last_decision_report["transformed_state_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
