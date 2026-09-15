"""B1-SC-D2 v1.0 QA suite. No scientific training or evaluation is performed."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from experiments.b1sc_d2_v1_0 import implementation as d2
from b1s.execution.core import EDGES, RoutingActor, model_digest
from b1s.execution.instrument import NativeEnv, runtime_lock


def write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def matched_actors(seed: int = 12345):
    torch.manual_seed(seed)
    s6 = RoutingActor(d2.S6_MASK)
    torch.manual_seed(seed + 1)
    g0 = RoutingActor("000000")
    g0.E.load_state_dict(s6.E.state_dict())
    g0.B.load_state_dict(s6.B.state_dict())
    g0.D.load_state_dict(s6.D.state_dict())
    with torch.no_grad():
        g0.log_std.copy_(s6.log_std)
    return s6, g0


class RuleTests(unittest.TestCase):
    def test_design_contract(self):
        self.assertEqual(d2.validate_design_contract()["status"], "PASS")

    def test_support_exact(self):
        self.assertEqual(tuple(EDGES), d2.EDGE_ORDER)
        self.assertEqual(d2.S6_MASK, "100110")
        self.assertEqual(d2.ACTIVE_EDGES, (0, 3, 4))
        self.assertEqual(d2.active_pairs(), ((0, 1), (1, 2), (2, 0)))
        self.assertEqual(d2.INACTIVE_CONTROL_EDGE, 1)

    def test_registry_exact_16(self):
        rows = d2.registry()
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({x["id"] for x in rows}), 16)
        self.assertEqual({x["condition"] for x in rows}, set(d2.CONDITIONS))
        self.assertTrue(all(x["native_steps"] == 1_048_576 for x in rows))

    def test_paired_seed_contract(self):
        self.assertEqual(d2.paired_seed_contract()["status"], "PASS")
        for block in range(8):
            on, off = d2.block_pair(block)
            for key in (
                "initial_seed", "policy_sampling_seed", "training_environment_seed_root",
                "diagnostic_panel_seed", "endpoint_panel_seed", "causal_panel_seed",
            ):
                self.assertEqual(on[key], off[key])

    def test_condition_not_allowed_in_paired_identity(self):
        with self.assertRaises(RuntimeError):
            d2.paired_seed_identity(0, "weights", condition="S6-ON")

    def test_seed_namespaces_disjoint(self):
        values = [d2.seed(split, {"purpose": "qa-collision", "i": i}) for split in d2.ALLOWED_SPLITS for i in range(64)]
        self.assertEqual(len(values), len(set(values)))
        with self.assertRaises(RuntimeError):
            d2.seed("final", {"i": 0})

    def test_training_and_endpoint_operators(self):
        self.assertEqual(d2.training_lesion("S6-ON"), ())
        self.assertEqual(d2.endpoint_lesion("S6-ON"), ())
        self.assertEqual(d2.training_lesion("S6-OFF-TRAIN"), (0, 3, 4))
        self.assertEqual(d2.endpoint_lesion("S6-OFF-TRAIN"), (0, 3, 4))
        self.assertEqual(d2.permanent_lesion(), (0, 3, 4))
        self.assertEqual(d2.inactive_control_lesion(), (1,))

    def test_replication_window_exact(self):
        self.assertEqual(d2.window_9_40_lesion(8), ())
        self.assertTrue(all(d2.window_9_40_lesion(i) == (0, 3, 4) for i in range(9, 41)))
        self.assertEqual(d2.window_9_40_lesion(41), ())

    def test_training_gate_6_of_8(self):
        self.assertTrue(d2.training_gate([True] * 6 + [False] * 2, True)["H_TRAIN_D2"])
        self.assertFalse(d2.training_gate([True] * 5 + [False] * 3, True)["H_TRAIN_D2"])
        self.assertFalse(d2.training_gate([True] * 8, False)["H_TRAIN_D2"])

    def test_online_gate_6_of_8_and_controls(self):
        self.assertTrue(d2.online_gate([True] * 6 + [False] * 2, True, True)["H_ONLINE_D2"])
        self.assertFalse(d2.online_gate([True] * 8, True, False)["H_ONLINE_D2"])
        self.assertFalse(d2.online_gate([True] * 5 + [False] * 3, True, True)["H_ONLINE_D2"])

    def test_practical_boundaries(self):
        self.assertFalse(d2.practical_advantage(d2.LOSS_MARGIN, -1.0))
        self.assertTrue(d2.practical_advantage(d2.LOSS_MARGIN + 1e-6, -1.0))
        self.assertTrue(d2.practical_advantage(-d2.LOSS_MARGIN, 1.0 + 1e-6))
        self.assertFalse(d2.practical_harm(d2.LOSS_MARGIN, -1.0))
        self.assertTrue(d2.practical_harm(d2.LOSS_MARGIN + 1e-6, -1.0))

    def test_adjudication_table(self):
        self.assertEqual(d2.adjudicate(True, False), "TRAINING_SCAFFOLD_SUPPORTED")
        self.assertEqual(d2.adjudicate(True, True), "TRAINING_AND_ONLINE_DEPENDENCE_SUPPORTED")
        self.assertEqual(d2.adjudicate(False, True), "ONLINE_DEPENDENCE_ONLY_SUPPORTED")
        self.assertEqual(d2.adjudicate(False, False), "NO_REPRODUCIBLE_S6_MECHANISM_UNDER_D2")
        self.assertEqual(d2.adjudicate(True, True, False), "INVALID_OR_INCONCLUSIVE_MECHANISTIC_ADJUDICATION")

    def test_checkpoints_frozen(self):
        self.assertEqual(d2.CHECKPOINTS, (262144, 524288, 786432, 1048576))
        self.assertEqual(d2.DIAGNOSTIC_EPISODES, 32)
        self.assertEqual(d2.ENDPOINT_EPISODES, 64)
        self.assertEqual(d2.CAUSAL_EPISODES, 64)

    def test_no_execution_boundary(self):
        design = d2.design()
        boundary = design["execution_boundary"]
        self.assertTrue(all(boundary[k] is False for k in (
            "execution_authorized", "workflow_installed", "start_request_exists",
            "training_started", "evaluation_started", "results_exist",
        )))
        self.assertFalse((d2.ROOT / "START_REQUEST.json").exists())


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(d2.seed("qa", {"purpose": "runtime-tests"}))

    def test_runtime_lock(self):
        lock = runtime_lock()
        self.assertEqual(lock["versions"]["torch"], "2.2.2+cpu")
        self.assertEqual(lock["versions"]["pettingzoo"], "1.24.3")

    def test_lambda_zero_received_route_exact_zero(self):
        actor = RoutingActor(d2.S6_MASK)
        z = torch.randn(4, 93)
        with torch.no_grad():
            _, _, trace = actor(z, d2.ACTIVE_EDGES, trace=True)
        self.assertEqual(float(trace["r"].abs().max()), 0.0)

    def test_off_forward_equals_g0_with_shared_direct_weights(self):
        s6, g0 = matched_actors()
        z = torch.randn(7, 93)
        with torch.no_grad():
            off, _ = s6(z, d2.ACTIVE_EDGES)
            ref, _ = g0(z)
        self.assertTrue(torch.allclose(off, ref, rtol=0, atol=d2.NUMERICAL_TOLERANCE))

    def test_off_gradients_ebd_equal_g0_and_a_zero(self):
        s6, g0 = matched_actors(22334)
        z = torch.randn(5, 93)
        s6.zero_grad(); g0.zero_grad()
        off, _ = s6(z, d2.ACTIVE_EDGES)
        ref, _ = g0(z)
        off.square().sum().backward()
        ref.square().sum().backward()
        for name in ("E", "B", "D"):
            a = getattr(s6, name)
            b = getattr(g0, name)
            self.assertTrue(torch.allclose(a.weight.grad, b.weight.grad, rtol=0, atol=d2.NUMERICAL_TOLERANCE), name + ".weight")
            self.assertTrue(torch.allclose(a.bias.grad, b.bias.grad, rtol=0, atol=d2.NUMERICAL_TOLERANCE), name + ".bias")
        agrad = s6.A.weight.grad
        self.assertTrue(agrad is None or float(agrad.abs().max()) == 0.0)
        abias = s6.A.bias.grad
        self.assertTrue(abias is None or float(abias.abs().max()) == 0.0)

    def test_on_off_initial_parameters_identical_when_seed_replayed(self):
        sd = d2.block_pair(3)[0]["initial_seed"]
        torch.manual_seed(sd)
        on = RoutingActor(d2.S6_MASK)
        torch.manual_seed(sd)
        off = RoutingActor(d2.S6_MASK)
        self.assertEqual(model_digest(on), model_digest(off))

    def test_permanent_lesion_only_changes_active_lambda(self):
        actor = RoutingActor(d2.S6_MASK)
        z = torch.randn(2, 93)
        with torch.no_grad():
            _, _, intact = actor(z, trace=True)
            _, _, lesion = actor(z, d2.permanent_lesion(), trace=True)
        changed = (intact["lambda"] - lesion["lambda"]).nonzero().flatten().tolist()
        self.assertEqual(changed, list(d2.ACTIVE_EDGES))
        self.assertTrue(torch.equal(intact["h"], lesion["h"]))
        self.assertTrue(torch.equal(intact["m"], lesion["m"]))

    def test_inactive_edge_control_exact(self):
        actor = RoutingActor(d2.S6_MASK)
        z = torch.randn(5, 93)
        with torch.no_grad():
            intact, _ = actor(z)
            ctl, _ = actor(z, d2.inactive_control_lesion())
        self.assertTrue(torch.equal(intact, ctl))

    def test_native_environment_smoke_only(self):
        env = NativeEnv()
        try:
            z, _ = env.reset(seed=d2.seed("qa", {"purpose": "native-smoke"}))
            self.assertEqual(z.shape, (93,))
            nxt, reward, term, trunc, audit = env.step(np.zeros(12, dtype=np.float32))
            self.assertEqual(nxt.shape, (93,))
            self.assertTrue(math.isfinite(reward))
            self.assertEqual(audit["requested"], audit["admitted"])
        finally:
            env.close()


def run_suite(case) -> dict:
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(case)
    result = unittest.TestResult()
    suite.run(result)
    return {
        "status": "PASS" if result.wasSuccessful() else "FAIL",
        "tests_run": result.testsRun,
        "failures": [str(x[0]) + ": " + x[1] for x in result.failures],
        "errors": [str(x[0]) + ": " + x[1] for x in result.errors],
        "skipped": [str(x) for x in result.skipped],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    rules = run_suite(RuleTests)
    runtime = run_suite(RuntimeTests)
    write(a.out / "RULE_TESTS.json", rules)
    write(a.out / "RUNTIME_TESTS.json", runtime)
    ok = rules["status"] == "PASS" and runtime["status"] == "PASS"
    validation = dict(
        status="D2_QA_PASS_NOT_SCIENTIFIC_EXECUTION" if ok else "D2_QA_FAIL",
        rule_tests=rules["tests_run"],
        runtime_tests=runtime["tests_run"],
        scientific_training_performed=False,
        scientific_evaluation_performed=False,
        qa_microfit_performed=False,
        start_request_present=(d2.ROOT / "START_REQUEST.json").exists(),
        implementation=d2.implementation_summary(),
        **d2.BOUNDARY,
    )
    write(a.out / "VALIDATION.json", validation)
    print(json.dumps(validation, indent=2, sort_keys=True))
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
