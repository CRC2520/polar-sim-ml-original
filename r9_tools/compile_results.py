"""Compile complete R9 evidence outside the frozen scientific source tree.

Default mode requires all 80 final seeds, 15 variants and four domains. The
explicit --pilot-fixture mode accepts only the three development seeds and two
ecologies, and never computes primary tests. No simulation is executed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from r8_completion.io import atomic_json, atomic_text, write_manifest
from r9_completion.config import DOMAINS, FINAL_SEEDS, KINDS, PILOT_SEEDS, PROTOCOL, VARIANTS
from r9_completion.statistics import compile_from_metrics, episode_endpoints, paired_mean_interval


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def _json_default(value):
    if isinstance(value,np.ndarray):
        return value.tolist()
    if isinstance(value,np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def assert_same(expected,actual,context):
    if isinstance(expected,dict):
        if set(expected) != set(actual):
            raise AssertionError(f"{context}: dictionary keys differ")
        for key in expected:
            assert_same(expected[key],actual[key],context+"/"+key)
    else:
        np.testing.assert_array_equal(expected,actual,err_msg=context)


def check_manifest(folder):
    path = folder/"COMPLETE.json"
    manifest = read_json(path)
    if manifest.get("complete") is not True:
        raise ValueError(f"Incomplete seed: {folder}")
    records,members,npz_files = {},0,0
    for entry in manifest["files"]:
        artifact = (folder/entry["path"]).resolve()
        if not artifact.is_relative_to(folder.resolve()) or artifact in records:
            raise ValueError(f"Invalid/duplicate artifact path: {artifact}")
        if artifact.stat().st_size != entry["bytes"] or sha256(artifact) != entry["sha256"]:
            raise ValueError(f"Artifact SHA256/size mismatch: {artifact}")
        if artifact.suffix == ".npz":
            with zipfile.ZipFile(artifact) as archive:
                if archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist())):
                    raise ValueError(f"CRC/member validation failed: {artifact}")
            with np.load(artifact,allow_pickle=False) as arrays:
                observed = {}
                for key in arrays.files:
                    value = arrays[key]
                    if value.dtype.hasobject:
                        raise TypeError(f"Object dtype forbidden: {artifact}/{key}")
                    observed[key] = {"shape":list(value.shape),"dtype":str(value.dtype)}
                assert_same(entry["arrays"],observed,f"NPZ metadata:{artifact}")
                members += len(observed)
            npz_files += 1
        records[artifact] = entry
    return records,{"manifest_sha256":sha256(path),"files_sha256":len(records),
                    "npz_files_crc":npz_files,"npz_members_safe_load":members}


def event_metrics(path,trace):
    counts = Counter({name:0 for name in (
        "goal_formed_events","goal_revised_events","goal_abandoned_events","goal_persisted_events",
        "memory_events","memory_updates_applied","memory_updates_changed","memory_updates_suppressed",
        "memory_formations_applied","memory_reconsolidations","memory_reactivations",
        "memory_horizon1_events","memory_horizon4_events","memory_horizon12_events")})
    memory_by_step = np.zeros(PROTOCOL["steps"],dtype=int)
    goals_by_step = np.zeros(PROTOCOL["steps"],dtype=int)
    if path.exists():
        seen = set()
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path,"rt",encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                step = row["step"]
                if step in seen or not 0 <= step < PROTOCOL["steps"]:
                    raise ValueError(f"Duplicate/out-of-range event step: {path}")
                seen.add(step)
                memory_by_step[step] = len(row["memory"])
                goals_by_step[step] = len(row["goals"])
                for event in row["goals"]:
                    key = event["type"]+"_events"
                    if key not in counts:
                        raise ValueError(f"Unknown goal event {key}")
                    counts[key] += 1
                for event in row["memory"]:
                    counts["memory_events"] += 1
                    counts[f"memory_horizon{event['horizon']}_events"] += 1
                    if event["destination_time"] != step or event["destination_time"]-event["origin_time"] != event["horizon"]:
                        raise ValueError(f"Noncausal memory event timing: {path}")
                    applied = bool(event["update_applied"])
                    changed = not np.array_equal(event["before"],event["after"])
                    counts["memory_updates_applied"] += int(applied)
                    counts["memory_updates_changed"] += int(applied and changed)
                    counts["memory_updates_suppressed"] += int(not applied)
                    counts["memory_formations_applied"] += int(applied and event["count_before"] == 0)
                    counts["memory_reconsolidations"] += int(event["reconsolidated"])
                    counts["memory_reactivations"] += int(event["recalled"])
    np.testing.assert_array_equal(memory_by_step,trace["workspace_update_count"],err_msg=f"Memory event completeness:{path}")
    np.testing.assert_array_equal(goals_by_step,trace["goal_event_count"],err_msg=f"Goal event completeness:{path}")
    return dict(counts)


def load_evaluation(folder,records,seed,domain,variant,summary):
    path = folder/"evaluation"/domain/variant
    for filename in ("metrics.json","trace.npz"):
        if (path/filename).resolve() not in records:
            raise ValueError(f"Unmanifested evaluation file: {path/filename}")
    stored = read_json(path/"metrics.json")
    metadata = stored["metadata"]
    for key,value in (("seed",seed),("domain",domain),("variant",variant),("measured_steps",320),
                      ("epsilon",0.),("learn",False)):
        assert_same(value,metadata[key],f"Evaluation metadata:{path}/{key}")
    if metadata["policy_parameter_sha256_before"] != metadata["policy_parameter_sha256_after"]:
        raise ValueError(f"Learned parameters changed in evaluation: {path}")
    with np.load(path/"trace.npz",allow_pickle=False) as data:
        trace = {key:data[key].copy() for key in data.files}
    if any(value.shape[0] != 320 or not np.isfinite(value).all() for value in trace.values()):
        raise ValueError(f"Incomplete or nonfinite trace: {path}")
    endpoints = episode_endpoints(trace["reward"],trace["alive"],horizon=320)
    reconstructed = {
        "reward_mean":endpoints["reward_mean"],"alive_fraction":endpoints["alive_fraction"],
        "reserve_mean":float(trace["reserve"].mean()),"service_mean":float(trace["service"].mean()),
        "constraint_fraction":float(trace["constraint"].mean()),
        "gate_active_fraction":float((trace["gate"]>0).mean()),
        "fallback_fraction":float(trace["fallback"].mean()),
        "mask_violations":int(trace["mask_violation"].sum()),"probes":int(trace["probe"].sum()),
        "observed_updates":int(trace["workspace_update_count"].sum()),
        "goal_events":int(trace["goal_event_count"].sum()),"all_finite":True,
    }
    assert_same(stored["metrics"],reconstructed,f"Metrics reconstructed:{path}")
    assert_same(summary,reconstructed,f"Summary/evaluation consistency:{path}")
    if reconstructed["mask_violations"] or reconstructed["probes"]:
        raise ValueError(f"Native mask violation or exploration: {path}")
    derived = {key:value for key,value in reconstructed.items() if key != "all_finite"}
    derived.update(terminal_survival=float(endpoints["terminal_alive"]),
        gate_mean=float(trace["gate"].mean()),gate_std=float(trace["gate"].std()),
        gate_switches=int(np.count_nonzero(np.diff(trace["gate"]))),
        goal_weight_viability_mean=float(trace["goals"][:,0].mean()),
        goal_weight_resource_mean=float(trace["goals"][:,1].mean()),
        goal_weight_task_mean=float(trace["goals"][:,2].mean()),
        memory_energy_prediction_mean=float(trace["memory_energy"].mean()),
        memory_resource_prediction_mean=float(trace["memory_resource"].mean()))
    event_candidates = [candidate for candidate in
        (path/"workspace_events.jsonl",path/"workspace_events.jsonl.gz") if candidate.is_file()]
    if len(event_candidates) != 1:
        raise ValueError(f"Require exactly one plain or gzip workspace event stream: {path}")
    events_path = event_candidates[0]
    if events_path.resolve() not in records:
        raise ValueError(f"Unmanifested workspace events: {events_path}")
    derived.update(event_metrics(events_path,trace))
    diagnostic = {"seed":seed,"domain":domain,"variant":variant,
        "gate_zero_all_steps":bool(np.all(trace["gate"]==0)),
        "gate_constant_all_steps":bool(np.ptp(trace["gate"])==0),
        "gate_unique_values":np.unique(trace["gate"]).tolist(),
        "gate_modulated":bool(np.ptp(trace["gate"])>0),
        "environment_tape_sha256":metadata["environment_tape_sha256"]}
    return reconstructed,derived,trace,diagnostic


def load_checkpoint(folder,records,seed,kind):
    path = folder/"checkpoints"/f"{kind}.json"
    if path.resolve() not in records:
        raise ValueError(f"Unmanifested checkpoint: {path}")
    checkpoint = read_json(path)
    if (checkpoint["seed"],checkpoint["kind"]) != (seed,kind):
        raise ValueError(f"Checkpoint identity mismatch: {path}")
    selection = checkpoint["selection_record"]
    candidates = selection["candidates"]
    if set(candidates) != set(PROTOCOL["selection_candidates"]):
        raise ValueError(f"Incomplete candidate budget: {path}")
    for label,count in (("training",8*320),("calibration",2*320),("selection",7*320)):
        if selection[f"total_{label}_transitions"] != count:
            raise ValueError(f"Budget mismatch:{path}/{label}")
    if not selection["candidate_workspace_updates_not_retained"]:
        raise ValueError("Candidate validation memory leaked into selected checkpoint")
    selection_tapes,selection_parameter_hashes = set(),set()
    for phase,episodes in (("training",[f"episode_{i:02d}" for i in range(8)]),
                           ("calibration",[f"episode_{i:02d}" for i in range(2)]),
                           ("selection",list(PROTOCOL["selection_candidates"]))):
        actual_episodes = {p.name for p in (folder/phase/kind).iterdir() if p.is_dir()}
        if actual_episodes != set(episodes):
            raise ValueError(f"Acquisition episode coverage differs: {path}/{phase}")
        for episode in episodes:
            episode_path = folder/phase/kind/episode
            for filename in ("trace.npz","metrics.json"):
                if (episode_path/filename).resolve() not in records:
                    raise ValueError(f"Unmanifested acquisition artifact: {episode_path/filename}")
            info = read_json(episode_path/"metrics.json")
            metadata = info["metadata"]
            expected = {"seed":seed,"kind":kind,"domain":"ecology_train","measured_steps":320,
                        "learn":phase=="training","epsilon":0. if phase=="selection" else .2}
            for key,value in expected.items():
                assert_same(value,metadata[key],f"Acquisition metadata:{episode_path}/{key}")
            if phase != "training" and metadata["policy_parameter_sha256_before"] != metadata["policy_parameter_sha256_after"]:
                raise ValueError(f"Parameters changed during frozen acquisition:{episode_path}")
            if phase == "selection":
                assert_same(candidates[episode],info["metrics"],f"Candidate metrics:{episode_path}")
                selection_tapes.add(metadata["environment_tape_sha256"])
                selection_parameter_hashes.add(metadata["policy_parameter_sha256_before"])
    if len(selection_tapes) != 1 or len(selection_parameter_hashes) != 1:
        raise ValueError(f"Candidate comparisons do not share a tape/checkpoint:{path}")
    def selected(names):
        viable = [name for name in names if candidates[name]["alive_fraction"] >= .80]
        available = viable or list(names)
        if viable:
            return max(available,key=lambda name:(candidates[name]["reward_mean"],-names.index(name)))
        return max(available,key=lambda name:(candidates[name]["alive_fraction"],candidates[name]["reward_mean"],-names.index(name)))
    expected = selected(("off","context_zero","context_small","context_large"))
    constant = selected(("off","constant_quarter","constant_half","constant_one"))
    if (selection["selected"],checkpoint["gate_mode"]) != (expected,expected):
        raise ValueError(f"Context selection mismatch:{path}")
    if (selection["constant_selected"],checkpoint["constant_mode"]) != (constant,constant):
        raise ValueError(f"Constant selection mismatch:{path}")
    diag = checkpoint["model"]["fit_diagnostics"]
    slopes = np.asarray(checkpoint["model"]["coef_"],dtype=float)
    intercepts = np.asarray(checkpoint["model"]["intercept_"],dtype=float)
    allowed = np.asarray(checkpoint["model"]["allowed_support"],dtype=bool)
    if slopes.shape != (4,48,4) or intercepts.shape != (4,4) or allowed.shape != slopes.shape:
        raise ValueError(f"Wrong allocated predictive shapes:{path}")
    actual_support = slopes != 0
    assert_same(int(actual_support.sum()),diag["active_slope_coefficients"],f"Actual support:{path}")
    assert_same(actual_support.sum(axis=1).tolist(),diag["active_by_action_output"],f"Support by output:{path}")
    assert_same(int(allowed.sum()),diag["allowed_slope_coefficients"],f"Allowed support:{path}")
    if np.any(slopes[~allowed] != 0):
        raise ValueError(f"Coefficient outside declared support:{path}")
    expected_budget = {"samples":2560,"allocated_coefficients_including_intercepts":784,
                       "gradient_steps_per_action":80,"gradient_steps_total":320}
    for key,value in expected_budget.items():
        if diag[key] != value:
            raise ValueError(f"Model fitting budget mismatch:{path}/{key}")
    counters = checkpoint["workspace"]["counters"]
    numeric = {key:diag[key] for key in ("allocated_coefficients_including_intercepts",
        "allowed_slope_coefficients","active_slope_coefficients","feature_rank_centered",
        "features_below_scale_floor","final_objective_direct","samples","gradient_steps_total")}
    numeric["checkpoint_memory_entries"] = len(checkpoint["workspace"]["memory"])
    numeric.update({"checkpoint_"+key:value for key,value in counters.items()})
    return {"seed":seed,"kind":kind,"selected_gate":expected,"selected_constant":constant,
        "training_arrays_sha256":diag["training_arrays_sha256"],"numeric":numeric,
        "training_steps":2560,"calibration_steps":640,"selection_steps":2240,
        "full_candidate_count":4,"constant_candidate_count":4,"unique_selection_episodes":7,
        "candidate_workspace_updates_retained":False,
        "active_by_action_output":diag["active_by_action_output"],
        "gate_coefficients":checkpoint["gate_coef"],"empirical_margin":checkpoint["margin"],
        "causal_graph_identified":diag["causal_graph_identified"]}


def describe(values,n):
    values = np.asarray(values,dtype=float)
    ci = paired_mean_interval(values,confidence=.95,resamples=20000,bootstrap_seed=953900,expected_n=n)
    return {**ci,"median":float(np.median(values)),"minimum":float(values.min()),
            "maximum":float(values.max()),"by_seed":values.tolist()}


def csv_text(rows):
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output.getvalue()


def compile_data(input_dir,output_dir,*,pilot_fixture=False):
    input_dir,output_dir = Path(input_dir),Path(output_dir)
    seeds = tuple(PILOT_SEEDS if pilot_fixture else FINAL_SEEDS)
    domains = tuple(DOMAINS[:2] if pilot_fixture else DOMAINS)
    phase = "pilot" if pilot_fixture else "final"
    expected_folders = {f"seed_{seed}" for seed in seeds}
    actual_folders = {path.name for path in input_dir.glob("seed_*") if path.is_dir()}
    if actual_folders != expected_folders:
        raise ValueError(f"Require exact complete seed set; missing={sorted(expected_folders-actual_folders)}, extra={sorted(actual_folders-expected_folders)}")
    if any(not (input_dir/name/"COMPLETE.json").is_file() for name in expected_folders):
        raise ValueError("Every expected seed must have COMPLETE.json before compilation")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite compiled evidence: {output_dir}")
    source = {"status":"development fixture, no confirmatory tests"}
    if not pilot_fixture:
        from r9_completion.provenance import verify
        source = verify()
    data,derived,checkpoint_rows,gate_rows,equivalence,integrity = {},{},[],[],[],{}
    flat_seed_rows = []
    for index,seed in enumerate(seeds):
        folder = input_dir/f"seed_{seed}"
        records,checks = check_manifest(folder)
        integrity[str(seed)] = checks
        if (folder/"summary.json").resolve() not in records:
            raise ValueError(f"Unmanifested summary:{folder}")
        summary = read_json(folder/"summary.json")
        if (summary["seed"],summary["phase"]) != (seed,phase):
            raise ValueError(f"Wrong phase/seed identity:{folder}")
        if not pilot_fixture:
            assert_same(source,summary["source"],f"Prospective freeze:{folder}")
        assert_same(list(KINDS),summary["trained_kinds"],"Trained kinds")
        assert_same(list(VARIANTS),summary["evaluated_variants"],"Evaluated variants")
        assert_same(list(domains),summary["evaluated_domains"],"Evaluated domains")
        if set(summary["metrics"]) != set(domains):
            raise ValueError(f"Missing or extra evaluation domain:{folder}")
        for kind in KINDS:
            checkpoint_rows.append(load_checkpoint(folder,records,seed,kind))
        data[seed],derived[seed] = {},{}
        for domain in domains:
            if set(summary["metrics"][domain]) != set(VARIANTS):
                raise ValueError(f"Missing or extra variant:{folder}/{domain}")
            data[seed][domain],derived[seed][domain] = {},{}
            traces,diagnostics = {},{}
            for variant in VARIANTS:
                metrics,extra,trace,diag = load_evaluation(folder,records,seed,domain,variant,summary["metrics"][domain][variant])
                data[seed][domain][variant],derived[seed][domain][variant] = metrics,extra
                traces[variant],diagnostics[variant] = trace,diag
                gate_rows.append(diag)
                flat_seed_rows.append({"seed":seed,"domain":domain,"variant":variant,**extra})
            tapes = {d["environment_tape_sha256"] for d in diagnostics.values()}
            if len(tapes) != 1:
                raise ValueError(f"Unpaired environment tapes:{folder}/{domain}")
            full,lesion = traces["full"],traces["noCross"]
            off = diagnostics["full"]["gate_zero_all_steps"]
            exact_members = {key:bool(np.array_equal(full[key],lesion[key])) for key in full}
            equivalence.append({"seed":seed,"domain":domain,"full_gate_zero_all_steps":off,
                "noCross_trace_exact":bool(all(exact_members.values())),
                "unequal_trace_members":[key for key,value in exact_members.items() if not value],
                "expected_off_gate_equivalence_pass":not off or bool(all(exact_members.values()))})
        if (index+1)%10 == 0 or index+1 == len(seeds):
            print(json.dumps({"validated_seeds":index+1,"planned_seeds":len(seeds),"phase":phase}),flush=True)
    confirmation = None if pilot_fixture else compile_from_metrics(data)
    descriptive,contrasts,adverse = {},{},{}
    mean_rows,difference_rows = [],[]
    for domain in domains:
        descriptive[domain],contrasts[domain],adverse[domain] = {},{},{}
        for variant in VARIANTS:
            descriptive[domain][variant],contrasts[domain][variant] = {},{}
            metrics = tuple(derived[seeds[0]][domain][variant])
            for metric in metrics:
                values = np.asarray([derived[seed][domain][variant][metric] for seed in seeds],dtype=float)
                baseline = np.asarray([derived[seed][domain]["full"][metric] for seed in seeds],dtype=float)
                stats,delta = describe(values,len(seeds)),describe(values-baseline,len(seeds))
                descriptive[domain][variant][metric] = stats
                contrasts[domain][variant][metric] = delta
                mean_rows.append({"domain":domain,"variant":variant,"metric":metric,"n":len(seeds),
                    "mean":stats["mean"],"lower95":stats["lower"],"upper95":stats["upper"],
                    "median":stats["median"],"minimum":stats["minimum"],"maximum":stats["maximum"]})
                difference_rows.append({"domain":domain,"contrast":variant+"_minus_full","metric":metric,"n":len(seeds),
                    "mean_difference":delta["mean"],"lower95":delta["lower"],"upper95":delta["upper"],
                    "negative_seeds":int(np.count_nonzero(values<baseline)),"zero_seeds":int(np.count_nonzero(values==baseline)),
                    "positive_seeds":int(np.count_nonzero(values>baseline))})
            va = np.asarray([derived[s][domain][variant]["alive_fraction"] for s in seeds])
            fa = np.asarray([derived[s][domain]["full"]["alive_fraction"] for s in seeds])
            vr = np.asarray([derived[s][domain][variant]["reward_mean"] for s in seeds])
            fr = np.asarray([derived[s][domain]["full"]["reward_mean"] for s in seeds])
            adverse[domain][variant] = {"variant_below_alive_floor_seeds":int((va<.8).sum()),
                "full_below_alive_floor_seeds":int((fa<.8).sum()),
                "full_alive_loss_over_tolerance_vs_variant":int((fa-va < -.005).sum()),
                "variant_alive_loss_over_tolerance_vs_full":int((va-fa < -.005).sum()),
                "full_reward_lower_than_variant_seeds":int((fr<vr).sum()),
                "variant_reward_lower_than_full_seeds":int((vr<fr).sum()),
                "interpretation":"Descriptive adverse-outcome counts; no additional hypothesis tests"}
    kinds = {}
    for kind in KINDS:
        rows = [row for row in checkpoint_rows if row["kind"]==kind]
        numeric_names = tuple(rows[0]["numeric"])
        kinds[kind] = {"gate_selection_counts":dict(Counter(row["selected_gate"] for row in rows)),
            "constant_selection_counts":dict(Counter(row["selected_constant"] for row in rows)),
            "numeric":{name:describe([row["numeric"][name] for row in rows],len(seeds)) for name in numeric_names},
            "budget":{"training_steps":2560,"calibration_steps":640,"selection_steps":2240,
                "full_candidate_count":4,"constant_candidate_count":4,"unique_selection_episodes":7},
            "interpretation":"Same allocated coefficient and update budgets; not equal effective rank or function class"}
    gate_summary = {}
    for domain in domains:
        for variant in VARIANTS:
            rows = [r for r in gate_rows if r["domain"]==domain and r["variant"]==variant]
            gate_summary[domain+"/"+variant] = {"zero_all_steps_seeds":sum(r["gate_zero_all_steps"] for r in rows),
                "constant_all_steps_seeds":sum(r["gate_constant_all_steps"] for r in rows),
                "modulated_seeds":sum(r["gate_modulated"] for r in rows)}
    off_rows = [row for row in equivalence if row["full_gate_zero_all_steps"]]
    diagnostics = {"gates_by_seed":gate_rows,"gate_summary":gate_summary,"noCross_equivalence":equivalence,
        "off_gate_native_episodes":len(off_rows),"off_gate_exact_equivalent_episodes":sum(r["noCross_trace_exact"] for r in off_rows),
        "unexpected_off_gate_equivalence_failures":[r for r in off_rows if not r["noCross_trace_exact"]],
        "interpretation":"Gate-off equivalence is a structural negative control, not evidence of useful coupling; a selected contextual mode can remain unmodulated. Goal persistence events are separate from revisions; memory attempts are separate from effective writes."}
    totals = {key:sum(r[key] for r in integrity.values()) for key in ("files_sha256","npz_files_crc","npz_members_safe_load")}
    if not pilot_fixture:
        after = verify()
        assert_same(source,after,"Frozen source after compilation")
    metadata = {"schema":"polar-r9-results-compiler-v1","phase":phase,"pilot_fixture":pilot_fixture,
        "seed_order":list(seeds),"n":len(seeds),"domains":list(domains),"variants":list(VARIANTS),
        "native_evaluations":len(seeds)*len(domains)*len(VARIANTS),"source":source,
        "compiler_sha256":sha256(__file__),"integrity_totals":totals,"integrity_by_seed":integrity,
        "primary_tests_computed":not pilot_fixture,"contrast_orientation":"variant minus full",
        "descriptive_intervals":"95% paired-seed percentile bootstrap, 20000 draws, seed953900; marginal, not multiplicity adjusted",
        "limits":["No consciousness endpoint or psychological-catalog validation.",
            "Predictive support is not an environmental causal graph.","No equivalence or mechanism usefulness follows from a nonsignificant primary test.",
            "Goal/memory events are designed computational operations; counts alone do not establish utility.",
            "Transfer families are internal synthetic generators, not external replication."]}
    files = []
    if confirmation is not None:
        files.append(atomic_json(output_dir/"R9_CONFIRMATION.json",confirmation))
        primary_rows = [{key:value for key,value in row.items() if key!="seed_success"} for row in confirmation["rows"]]
        files.append(atomic_text(output_dir/"R9_PRIMARY.csv",csv_text(primary_rows)))
    files.extend([
        atomic_json(output_dir/"R9_DESCRIPTIVES.json",{"metadata":metadata,"native":descriptive,"variant_minus_full":contrasts,"adverse":adverse}),
        atomic_json(output_dir/"R9_DIAGNOSTICS.json",diagnostics),
        atomic_json(output_dir/"R9_SELECTION_SUPPORT_BUDGET.json",{"by_kind":kinds,"by_seed":checkpoint_rows}),
        atomic_json(output_dir/"R9_COMPILATION_METADATA.json",metadata),
        atomic_text(output_dir/"R9_NATIVE_BY_SEED.csv",csv_text(flat_seed_rows)),
        atomic_text(output_dir/"R9_NATIVE_DESCRIPTIVES.csv",csv_text(mean_rows)),
        atomic_text(output_dir/"R9_PAIRED_DIFFERENCES.csv",csv_text(difference_rows)),
    ])
    write_manifest(output_dir/"COMPLETE.json",files,{"compiler_sha256":sha256(__file__),"phase":phase,
        "n":len(seeds),"primary_tests_computed":not pilot_fixture,"all_seed_artifacts_verified":True})
    return {"output":str(output_dir),"phase":phase,"n":len(seeds),"evaluations":metadata["native_evaluations"],
        "primary_tests_computed":not pilot_fixture,"integrity":totals,
        "off_gate_equivalence_failures":len(diagnostics["unexpected_off_gate_equivalence_failures"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--pilot-fixture",action="store_true",
        help="Only three declared pilots and two ecologies; suppress every primary test")
    args = parser.parse_args()
    print(json.dumps(compile_data(args.input,args.output_dir,pilot_fixture=args.pilot_fixture)),flush=True)


if __name__ == "__main__":
    main()
