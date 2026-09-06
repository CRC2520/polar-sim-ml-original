"""Causal smoke/regression tests for previously nonfunctional interventions."""
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import unittest
import tempfile
import torch

from engine_v2_corrected import EngineConfig
from p6_stimulus_mapping import POLARITY_POLES, polarity_evidence, stim_from_text
from p7_learnedM import run_one
from p_mini_iacl import MiniIACL
from p_mini_trauma import PersistentPerturbation, PersistentPerturbationCfg
from report_utils import recovery_from_traces, save_timeseries, summarize_trace, compress_trace, load_json

torch.set_num_threads(1)


class CorrectedExperimentsTest(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.scratch.cleanup()

    def test_all_poles_route_opposite_and_both_are_recorded(self):
        for index, (positive, negative) in enumerate(POLARITY_POLES):
            a, b = stim_from_text(positive, 16), stim_from_text(negative, 16)
            assert torch.equal(a, -b)
            assert a[index].item() == .5 and b[index].item() == -.5
            assert torch.count_nonzero(a).item() == 2
        assert polarity_evidence("Libertad vs Orden")[5] == (True, True)
        assert torch.count_nonzero(stim_from_text("Libertad vs Orden", 16)) == 0
        with self.assertRaisesRegex(ValueError, "No recognized"):
            stim_from_text("unknown token", 16)

    def test_iacl_requires_real_full_state(self):
        group = MiniIACL(num_agents=2, N=8, K=2, steps=2)
        with self.assertRaisesRegex(ValueError, "state API"):
            group._current_A(SimpleNamespace(N=8))
        with self.assertRaisesRegex(ValueError, "shape"):
            group._current_A(SimpleNamespace(N=8, current_activation=lambda: np.zeros(3)))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            group._current_A(SimpleNamespace(N=8, current_activation=lambda: np.full(8, np.nan)))

    def test_iacl_nonzero_synchronous_coupling_changes_trajectory(self):
        tmp_path = Path(self.scratch.name)
        group = MiniIACL(num_agents=3, N=16, K=4, steps=4, gamma_group_start=.2, gamma_group_end=.2)
        initial = np.stack([group._current_A(e) for e in group.agents])
        trace = group.run(str(tmp_path / "coupled"), print_console=False)
        control = MiniIACL(num_agents=3, N=16, K=4, steps=4, gamma_group_start=0., gamma_group_end=0.)
        control.run(str(tmp_path / "control"), print_console=False)
        np.testing.assert_allclose(trace["A_before"][0], initial)
        expected = .2 * (initial.mean(axis=0) - initial)
        np.testing.assert_allclose(trace["stimulus"][0], expected)
        assert trace["coupling_rms"][0] > 0
        assert all(engine.t == 4 for engine in group.agents)  # no hidden warm-up
        assert not np.allclose(trace["A"][-1], control.ts["A"][-1])

    def test_p7_injects_documented_pulse_and_matches_ablation_schedule(self):
        tmp_path = Path(self.scratch.name)
        traces = []
        for name, toggle in (("base", {}), ("no_modulation", {"no_ethics": True}), ("no_homeo", {"no_homeostasis": True})):
            cfg = EngineConfig(N=16, K=4, steps=12, stim_step=8, stim_scale=.6, learn_M=True, **toggle)
            run_one(cfg, str(tmp_path / name), name, verbose=False, plots=False)
            payload = json.loads((tmp_path / name / "p7_timeseries.json").read_text())
            traces.append(payload["timeseries"])
            stimulus = np.asarray(traces[-1]["stimulus"])
            assert np.flatnonzero(np.any(stimulus != 0, axis=1)).tolist() == [8]
            np.testing.assert_allclose(stimulus[8, [5, 13]], [.6, .6])
            control = json.loads((tmp_path / name / "p7_control_timeseries.json").read_text())["timeseries"]
            np.testing.assert_allclose(traces[-1]["A_before"][0], control["A_before"][0])
            assert not np.allclose(traces[-1]["A"][8], control["A"][8])
        for trace in traces[1:]:
            np.testing.assert_allclose(trace["stimulus"], traces[0]["stimulus"])

    def test_reporting_preserves_vectors_and_refuses_unequal_columns(self):
        tmp_path = Path(self.scratch.name)
        ts = {"step": [0, 1], "A": [[.1, .2], [.3, .4]], "HGI": [None, .5], "seed": 9}
        save_timeseries(str(tmp_path), "sample", ts, {"config": {"N": 2}})
        with open(tmp_path / "sample_timeseries.csv") as stream:
            rows = list(csv.DictReader(stream))
        assert json.loads(rows[0]["A"]) == [.1, .2]
        assert rows[0]["HGI"] == "null"
        payload = json.loads((tmp_path / "sample_timeseries.json").read_text())
        assert payload["metadata"]["seed"] == 9 and payload["timeseries"]["A"] == ts["A"]
        with self.assertRaisesRegex(ValueError, "expected 2"):
            save_timeseries(str(tmp_path), "bad", {"step": [0, 1], "HGI": [.1]})

    def test_gzip_preserves_full_trace_and_removes_plain_copy(self):
        path = Path(self.scratch.name) / "trace.json"
        payload = {"timeseries": {"step": [0], "A": [[.1, -.1]]}}
        path.write_text(json.dumps(payload))
        compressed = compress_trace(path)
        assert not path.exists()
        assert load_json(compressed) == payload
        first_bytes = compressed.read_bytes()
        path.write_text(json.dumps(payload))
        assert compress_trace(path).read_bytes() == first_bytes

    def test_missing_interventions_are_null_and_percentages_have_correct_units(self):
        row = summarize_trace({"step": [0, 1], "HGI": [.8, .9], "INC": [.4, .5]}, "incomplete")
        assert row["Override_pct"] is None and row["Continuous_modulation_pct"] is None
        assert row["Recovery"] is None and row["Recovery_status"] == "unavailable"
        row = summarize_trace({"step": [0, 1], "alpha": [.9, 1.], "override": [False, True]}, "flags")
        assert row["Continuous_modulation_pct"] == 50 and row["Override_pct"] == 50

    def test_recovery_is_sustained_and_censoring_not_zero(self):
        control = {"step": list(range(7)), "A": [[0.]] * 7}
        trace = {"step": list(range(7)), "stimulus": [[0.], [1.], [0.], [0.], [0.], [0.], [0.]],
                 "A": [[0.], [1.], [.01], [.3], [.02], [.01], [0.]]}
        result = recovery_from_traces(trace, control)
        assert result["status"] == "recovered" and result["steps"] == 3
        trace["A"][-1] = [.2]
        result = recovery_from_traces(trace, control)
        assert result["status"] == "censored" and result["steps"] is None
        trace["stimulus"] = [[0.]] * 7
        assert recovery_from_traces(trace)["status"] == "not_applicable"

    def test_persistent_perturbation_changes_external_input_not_memory(self):
        base = PersistentPerturbationCfg(N=16, K=4, steps=28)
        treated = PersistentPerturbation(base)
        untreated = PersistentPerturbation(PersistentPerturbationCfg(N=16, K=4, steps=28, attenuate_external_input=False))
        np.testing.assert_allclose(treated.stimulus_at(10)[0], untreated.stimulus_at(10)[0])
        assert np.linalg.norm(treated.stimulus_at(20)[0]) < np.linalg.norm(untreated.stimulus_at(20)[0])
        assert np.linalg.norm(treated.stimulus_at(27)[0]) > 0  # input persists through final sample

if __name__ == "__main__":
    unittest.main()
