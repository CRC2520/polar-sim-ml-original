"""Acceptance tests for corrected legacy dynamics and auditability."""
import copy
import json
import math
import unittest

import torch

from engine_v2_corrected import (DEVICE, EngineConfig, HGI_graph, INC_weighted,
                                 SAT, TensionEngine, VERSION)


def passive_config(**kwargs):
    defaults = dict(N=8, K=2, steps=4, disable_modulation=True, no_homeostasis=True,
                    gamma_cons_start=0.0, gamma_cons_end=0.0, beta_mu=0.0,
                    hgi_gain=0.0, inc_gain=0.0)
    defaults.update(kwargs)
    return EngineConfig(**defaults)


class CorrectedEngineAcceptance(unittest.TestCase):
    def test_saturation_is_reachable_and_clipping_is_observable(self):
        engine = TensionEngine(passive_config())
        engine.tensions.zero_()
        _, first_flags = engine.step(torch.ones(8, device=DEVICE))
        self.assertFalse(first_flags["numeric_clip"])
        activation, flags = engine.step(torch.ones(8, device=DEVICE))
        self.assertTrue(flags["numeric_clip"])
        self.assertEqual(engine.ts["SAT"][-1], 1.0)
        self.assertLessEqual(activation.abs().max().item(), math.tanh(1.2) + 1e-6)
        self.assertEqual(SAT(activation, 0.85).item(), 0.0)
        with self.assertRaisesRegex(ValueError, "reachable"):
            TensionEngine(passive_config(saturation_threshold=0.85))

    def test_malformed_stimuli_and_missing_states_do_not_become_zero(self):
        engine = TensionEngine(passive_config())
        bad_inputs = [torch.zeros(7), torch.zeros(8, 1), torch.full((8,), float("nan")),
                      torch.full((8,), float("inf")), torch.full((8,), 1.01), [0.0] * 8]
        initial = engine.current_activation
        for value in bad_inputs:
            with self.subTest(value=str(value)):
                with self.assertRaises(ValueError):
                    engine.step(value)
                self.assertEqual(engine.t, 0)
                self.assertTrue(torch.equal(engine.current_activation, initial))
        del engine.tensions
        with self.assertRaises(AttributeError):
            _ = engine.current_activation

    def test_memory_timing_and_inc_can_be_reconstructed_from_trace(self):
        engine = TensionEngine(passive_config(memory_retention=0.75))
        engine.tensions.fill_(0.2)
        engine.A_prev.fill_(math.tanh(0.2))
        engine.mu_t.zero_()
        engine.step(torch.ones(8, device=DEVICE) * 0.3)
        ts = engine.ts
        before = torch.tensor(ts["A_before"][0])
        after = torch.tensor(ts["A"][0])
        old_mu = torch.tensor(ts["mu_before"][0])
        new_mu = torch.tensor(ts["mu_after"][0])
        self.assertTrue(torch.allclose(new_mu, 0.75 * old_mu + 0.25 * after))
        self.assertAlmostEqual(ts["INC"][0], INC_weighted(after, before, old_mu).item(), places=6)
        self.assertLess(ts["INC"][0], INC_weighted(after, before, new_mu).item() - 0.3)
        self.assertTrue(torch.allclose(torch.tensor(ts["delta"][0]),
                                      torch.tensor(ts["tension"][0]) - torch.tensor(ts["tension_before"][0])))

    def test_memory_intervention_changes_subsequent_unstimulated_state(self):
        cfg = passive_config(beta_mu=0.6)
        remember = TensionEngine(cfg)
        erase = TensionEngine(cfg)
        for engine in (remember, erase):
            engine.tensions.zero_()
            engine.mu_t.zero_()
            engine.A_prev.zero_()
            engine.step(torch.ones(8, device=DEVICE) * 0.5)
        self.assertTrue(torch.equal(remember.current_activation, erase.current_activation))
        erase.mu_t.zero_()
        remember.step()
        erase.step()
        self.assertFalse(torch.equal(remember.current_activation, erase.current_activation))
        self.assertEqual(sum(abs(x) for x in remember.ts["stimulus"][-1]), 0.0)

    def test_modulation_override_guards_and_numeric_clip_are_distinct(self):
        engine = TensionEngine(passive_config(disable_modulation=False, ethics_tau_ovr=2.0))
        engine.tensions.copy_(torch.tensor([0.8, -0.8] * 4, device=DEVICE))
        _, flags = engine.step()
        self.assertTrue(flags["continuous_modulation"])
        self.assertFalse(flags["override"])
        self.assertFalse(flags["numeric_clip"])
        self.assertFalse(flags["hgi_guard"])
        override = TensionEngine(passive_config(disable_modulation=False, ethics_tau_ovr=0.0))
        override.tensions.fill_(1.1)
        _, flags = override.step()
        self.assertTrue(flags["override"])
        self.assertTrue(flags["override_effect"])
        self.assertFalse(flags["numeric_clip"])
        self.assertLessEqual(override.tensions.abs().max().item(), 1.0)
        guard = TensionEngine(passive_config(hgi_floor=1.0, hgi_gain=0.1))
        guard.tensions.copy_(torch.tensor([0.8, -0.8] * 4, device=DEVICE))
        _, flags = guard.step()
        self.assertTrue(flags["hgi_guard"])
        self.assertFalse(flags["continuous_modulation"])
        self.assertFalse(flags["override"])

    def test_no_ethics_compatibility_alias_disables_modulation_only(self):
        cfg = passive_config(disable_modulation=False, no_ethics=True, hgi_gain=0.1, hgi_floor=1.0)
        engine = TensionEngine(cfg)
        _, flags = engine.step()
        self.assertEqual(engine.ts["alpha"][-1], 1.0)
        self.assertFalse(flags["override"])
        self.assertTrue(flags["hgi_guard"])
        self.assertGreater(engine.ts["risk"][-1], 0.0)

    def test_seeded_replay_and_complete_json_payload(self):
        cfg = EngineConfig(N=8, K=2, steps=5, seed=44, learn_M=True, max_grad_norm=0.01)
        first, second = TensionEngine(cfg), TensionEngine(cfg)
        schedule = [None, torch.arange(8, device=DEVICE) / 10, None, None, None]
        for stimulus in schedule:
            first.step(stimulus)
            second.step(stimulus)
        self.assertEqual(first.ts, second.ts)
        payload = first.export_payload()
        parsed = json.loads(json.dumps(payload, allow_nan=False))
        self.assertEqual(parsed["metadata"]["engine_version"], VERSION)
        self.assertEqual(parsed["metadata"]["seed"], 44)
        self.assertEqual(len(parsed["timeseries"]["stimulus"]), 5)
        self.assertTrue(all(v is not None for v in first.ts["learning_loss"]))
        self.assertGreater(first.M.detach().norm().item(), 0.0)
        self.assertTrue(torch.isfinite(first.M).all())
        self.assertLessEqual(first.M.grad.norm().item(), 0.01001)
        self.assertLessEqual(first.M.detach().abs().max().item(), cfg.max_abs_M)
        # Export and public properties must not let the caller mutate live state.
        parsed["timeseries"]["A"][0][0] = 999
        clone = first.A
        clone.zero_()
        self.assertFalse(torch.equal(first.A, clone))

    def test_unit_count_is_not_polarity_type_count(self):
        engine = TensionEngine(EngineConfig(N=16, K=2))
        self.assertEqual(len(engine.nodes), 16)
        self.assertEqual(len(set(engine.nodes)), 8)
        with self.assertRaisesRegex(ValueError, "exactly N"):
            TensionEngine(EngineConfig(N=16, K=2), nodes=engine.nodes[:8])
        with self.assertRaisesRegex(ValueError, "exactly N"):
            TensionEngine(EngineConfig(N=8, K=2), nodes=[""] * 8)

    def test_descriptors_do_not_certify_function(self):
        engine = TensionEngine(passive_config())
        zero = torch.zeros(8, device=DEVICE)
        uniform = torch.full((8,), 0.8, device=DEVICE)
        self.assertEqual(HGI_graph(zero, engine.mask).item(), 1.0)
        self.assertEqual(INC_weighted(zero, zero, zero).item(), 0.5)
        self.assertEqual(HGI_graph(uniform, engine.mask).item(), 1.0)
        self.assertAlmostEqual(INC_weighted(uniform, uniform, uniform).item(), 1.0, places=6)
        self.assertEqual(HGI_graph(uniform, torch.zeros_like(engine.mask)).item(), 1.0)


if __name__ == "__main__":
    unittest.main()
