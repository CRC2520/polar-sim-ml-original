"""Trace-first outcomes, integrity verification and predeclared inference."""
import csv
import gzip
import hashlib
import json
import math
import io
import tarfile
from pathlib import Path
import numpy as np
from .environments import transition_details


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    return value


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(plain(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(payload, compresslevel=9, mtime=0))
    else:
        path.write_text(json.dumps(plain(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def read_json(path):
    path = Path(path)
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(data, parse_constant=reject_constant)


def source_hashes(root):
    root = Path(root)
    sources = sorted(list((root/"study2").glob("*.py")) +
                     [root/"scripts/run_study2.py", root/"scripts/regenerate_study2.py",
                      root/"docs/study2_protocol.json", root/"docs/STUDY2_PROTOCOL.md", root/"docs/STUDY2_CONTROLLER_SPEC.md"] +
                     list((root/"tests").glob("test_study2*.py")))
    return {str(p.relative_to(root)): sha256(p) for p in sources}


def pack_artifacts(directory, paths, chunk_bytes=6_000_000):
    """Pack already-compressed, individually hashed records into small tar parts.

    Deterministic headers; no extraction is needed during regeneration. Packing
    changes transport only, preserving every logical record and its SHA256.
    """
    directory = Path(directory)
    archives, index = [], {}
    archive = None
    archive_path = None
    used = 0
    part = 0
    try:
        for relative in sorted(set(paths)):
            source = directory/relative
            data = source.read_bytes()
            size = 512 + ((len(data)+511)//512)*512
            if len(data) > chunk_bytes-10240:
                raise ValueError("An individual compressed trace exceeds the archive size limit")
            if archive is None or used+size+10240 > chunk_bytes:
                if archive is not None:
                    archive.close()
                    archives.append(dict(path=str(archive_path.relative_to(directory)), sha256=sha256(archive_path), bytes=archive_path.stat().st_size))
                archive_path = directory/f"archives/part-{part:03d}.tar"
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                archive = tarfile.open(archive_path, "w", format=tarfile.USTAR_FORMAT)
                part += 1
                used = 0
            info = tarfile.TarInfo(relative)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(data))
            used += size
            index[relative] = dict(archive=str(archive_path.relative_to(directory)), member=relative,
                                   sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
        if archive is not None:
            archive.close()
            archive = None
            archives.append(dict(path=str(archive_path.relative_to(directory)), sha256=sha256(archive_path), bytes=archive_path.stat().st_size))
        # Only remove redundant loose files after every archive has closed.
        for relative in index:
            (directory/relative).unlink()
        return dict(archives=archives, artifacts=index)
    finally:
        if archive is not None:
            archive.close()


class ArtifactReader:
    def __init__(self, directory, manifest):
        self.directory, self.manifest, self.handles = Path(directory), manifest, {}
        for item in manifest.get("archives", []):
            path = self.directory/item["path"]
            if sha256(path) != item["sha256"] or path.stat().st_size != item["bytes"]:
                raise ValueError("Archive integrity mismatch")
            self.handles[item["path"]] = tarfile.open(path, "r")

    def read(self, relative, expected_sha):
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Unsafe logical artifact path")
        if "artifacts" in self.manifest:
            item = self.manifest["artifacts"][relative]
            if item["member"] != relative or item["sha256"] != expected_sha:
                raise ValueError("Artifact index disagrees with trial manifest")
            stream = self.handles[item["archive"]].extractfile(item["member"])
            if stream is None:
                raise ValueError("Missing archive member")
            data = stream.read()
            if len(data) != item["bytes"]:
                raise ValueError("Artifact byte count mismatch")
        else:
            data = (self.directory/relative).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha:
            raise ValueError("Artifact SHA256 mismatch")
        payload = gzip.decompress(data) if relative.endswith(".gz") else data
        return json.loads(payload, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Nonfinite JSON {x}")))

    def close(self):
        for handle in self.handles.values():
            handle.close()


def recovery(losses, start, end, threshold, sustained):
    for index in range(start, end-sustained+1):
        if np.all(np.asarray(losses[index:index+sustained]) <= threshold):
            return {"start": start, "end": end, "status": "recovered", "steps": index-start}
    return {"start": start, "end": end, "status": "right_censored", "steps": None}


def summarize_trial(environment, trace, protocol):
    frames, records = environment["frames"], trace["records"]
    if len(records) != protocol["steps"] or len(frames) != len(records):
        raise ValueError("Complete exact-length environment and controller traces required")
    state = np.asarray(environment["initial_state"], float)
    losses, costs, violations, stockouts, overflows, censored = [], [], [], [], [], []
    action_seconds = learning_seconds = 0.
    for t, (frame, record) in enumerate(zip(frames, records)):
        if record["step"] != t or frame["step"] != t:
            raise ValueError("Missing, duplicate or out-of-order transition")
        action = np.asarray(record["action"], float)
        before, after = np.asarray(record["state_before"], float), np.asarray(record["state_after"], float)
        if action.shape != (8,) or before.shape != (8,) or after.shape != (8,) or not np.isfinite([action, before, after]).all():
            raise ValueError("Finite eight-channel complete states/actions required")
        if not np.array_equal(before, state):
            raise ValueError("Non-contiguous persistent state trace")
        expected = transition_details(frame, state, action)
        if not np.allclose(after, expected["state"], rtol=0, atol=1e-12):
            raise ValueError("Saved transition disagrees with environment truth")
        if not np.array_equal(np.asarray(record["transition_valid"], bool), expected["transition_valid"]):
            raise ValueError("Saved censoring mask disagrees with physical transition")
        weight = np.asarray(frame["weights"], float)
        loss = float(np.sum(weight*(after-np.asarray(frame["target"]))**2)/np.sum(weight))
        cost = float(np.dot(frame["costs"], action))
        allowed = np.asarray(frame["allowed"], bool)
        violation = bool(np.any(action < -1e-10) or np.any(action > 1+1e-10) or cost > frame["budget"]+1e-9 or np.any(np.abs(action[~allowed]) > 1e-10))
        losses.append(loss); costs.append(cost); violations.append(violation)
        if expected["stockout"] is not None:
            stockouts.append(float(np.mean(expected["stockout"])))
            overflows.append(float(np.mean(expected["overflow"])))
            censored.append(float(np.mean(~expected["transition_valid"])))
        for name in ("action_seconds", "learning_seconds"):
            if not math.isfinite(record[name]) or record[name] < 0:
                raise ValueError("Nonnegative measured walltime required")
        action_seconds += record["action_seconds"]
        learning_seconds += record["learning_seconds"]
        state = after
    boundaries = [protocol["phase_length"]*i for i in (1, 2, 3)]
    rr = protocol["recovery_rule"]
    recoveries = [recovery(losses, start, boundaries[i+1] if i+1 < len(boundaries) else len(losses), rr["tracking_loss_max"], rr["sustained_steps"])
                  for i, start in enumerate(boundaries)]
    recovered = [r["steps"] for r in recoveries if r["steps"] is not None]
    recovery_fraction = len(recovered)/len(recoveries)
    mean_loss = float(np.mean(losses))
    rule = protocol["success_rule"]
    passed = bool(mean_loss <= rule["mean_tracking_loss_max"] and sum(violations) <= rule["hard_violations_max"] and recovery_fraction >= rule["recovery_fraction_min"])
    phase = protocol["phase_length"]
    initial_tail = float(np.mean(losses[phase-8:phase]))
    return_tail = float(np.mean(losses[-8:]))
    return dict(controller=trace["controller"], seed=trace["seed"], family=trace["family"], regime=trace["regime"],
                mean_tracking_loss=mean_loss, tracking_rmse=float(np.sqrt(mean_loss)), mean_action_cost=float(np.mean(costs)),
                hard_violations=int(sum(violations)), tracking_pass=passed, failure_rate=float(not passed),
                recovery_fraction=recovery_fraction, mean_recovery_steps=float(np.mean(recovered)) if recovered else None,
                recovery_censored=len(recoveries)-len(recovered), recoveries=recoveries,
                initial_tail_loss=initial_tail, return_tail_loss=return_tail, return_loss_change=return_tail-initial_tail,
                mean_stockout=float(np.mean(stockouts)) if stockouts else None, mean_overflow=float(np.mean(overflows)) if overflows else None,
                censored_transition_fraction=float(np.mean(censored)) if censored else None,
                action_seconds=action_seconds, learning_seconds=learning_seconds,
                controller_seconds=action_seconds+learning_seconds, losses=losses, action_costs=costs,
                profile=trace["profile"])


def seed_metric(summaries, controller, metric, families=None, regimes=None):
    rows = [r for r in summaries if r["controller"] == controller and (families is None or r["family"] in families) and (regimes is None or r["regime"] in regimes)]
    seeds = sorted({r["seed"] for r in rows})
    values = []
    expected_cells = {(r["family"], r["regime"]) for r in rows}
    for seed in seeds:
        group = [r for r in rows if r["seed"] == seed]
        if {(r["family"], r["regime"]) for r in group} != expected_cells or len(group) != len(expected_cells):
            raise ValueError("Missing or duplicate seed cells")
        if any(r[metric] is None for r in group):
            raise ValueError(f"Missing {metric}; select applicable family rather than treating missing as zero")
        values.append(float(np.mean([r[metric] for r in group])))
    if not values:
        raise ValueError("No matched seed observations")
    return seeds, np.asarray(values)


def contrast(summaries, a, b, metric, protocol, families=None, regimes=None, ratio=False):
    seeds_a, av = seed_metric(summaries, a, metric, families, regimes)
    seeds_b, bv = seed_metric(summaries, b, metric, families, regimes)
    if seeds_a != seeds_b:
        raise ValueError("Unpaired inferential seed sets")
    rng = np.random.default_rng(protocol["bootstrap_seed"])
    indices = rng.integers(0, len(av), (protocol["bootstrap_draws"], len(av)))
    if ratio:
        if float(np.mean(bv)) <= 0 or np.any(np.mean(bv[indices], axis=1) <= 0):
            raise ValueError("Undefined ratio denominator")
        effect = float(np.mean(av)/np.mean(bv))
        draws = np.mean(av[indices], axis=1)/np.mean(bv[indices], axis=1)
    else:
        effect = float(np.mean(av-bv))
        draws = np.mean((av-bv)[indices], axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return dict(a=a, b=b, metric=metric, ratio=ratio, n=len(av), effect=effect, ci95=[float(low), float(high)],
                paired_seed_sd=float(np.std(av-bv, ddof=1)) if len(av)>1 else None,
                families=families, regimes=regimes, a_mean=float(np.mean(av)), b_mean=float(np.mean(bv)))


def sample_size(summaries, protocol):
    seeds, av = seed_metric(summaries, "paired", "mean_tracking_loss")
    seeds_b, bv = seed_metric(summaries, "paired_lesion", "mean_tracking_loss")
    if seeds != protocol["pilot_seeds"] or seeds_b != seeds:
        raise ValueError("Sample-size calculation requires exactly the predeclared pilot seeds")
    sd = float(np.std(av-bv, ddof=1))
    unbounded = max(2, int(math.ceil((1.96*sd/protocol["precision_halfwidth_target"])**2)))
    chosen = min(protocol["final_n_max"], max(protocol["final_n_min"], unbounded))
    return dict(pilot_seeds=seeds, pilot_sd=sd, unbounded_n=unbounded, chosen_n=chosen,
                precision_target=protocol["precision_halfwidth_target"], pilot_precision_qualification=unbounded <= protocol["final_n_max"],
                estimated_halfwidth_at_chosen_n=1.96*sd/np.sqrt(chosen),
                final_seeds=list(range(protocol["final_seed_start"], protocol["final_seed_start"]+chosen)),
                note="Precision planning from pilot variance, not a power calculation; small-pilot SD is uncertain.")


def analyze(summaries, protocol, split, equivalence_error):
    primary = contrast(summaries, "paired", "paired_lesion", "mean_tracking_loss", protocol)
    comparisons = {b: contrast(summaries, "paired", b, "mean_tracking_loss", protocol) for b in protocol["controllers"] if b != "paired"}
    regimes = {r: contrast(summaries, "paired", "paired_lesion", "mean_tracking_loss", protocol, regimes=[r]) for r in protocol["regimes"]}
    families = {f: contrast(summaries, "paired", "paired_lesion", "mean_tracking_loss", protocol, families=[f]) for f in protocol["families"]}
    guards = dict(
        failures=contrast(summaries, "paired", "paired_lesion", "failure_rate", protocol),
        action_cost=contrast(summaries, "paired", "paired_lesion", "mean_action_cost", protocol, ratio=True),
        return_loss=contrast(summaries, "paired", "paired_lesion", "return_loss_change", protocol),
        stockout=contrast(summaries, "paired", "paired_lesion", "mean_stockout", protocol, families=["cooperative_inventory"]),
        overflow=contrast(summaries, "paired", "paired_lesion", "mean_overflow", protocol, families=["cooperative_inventory"]),
    )
    paired_rows = [r for r in summaries if r["controller"] == "paired"]
    lesion_rows = [r for r in summaries if r["controller"] == "paired_lesion"]
    runtime_ratio = sum(r["controller_seconds"] for r in paired_rows)/sum(r["controller_seconds"] for r in lesion_rows)
    limits = protocol["guardrails"]
    guard_pass = dict(
        failures=guards["failures"]["ci95"][1] <= limits["failure_rate_noninferiority_margin"],
        action_cost=guards["action_cost"]["ci95"][1] <= limits["action_cost_ratio_max"],
        return_loss=guards["return_loss"]["ci95"][1] <= limits["return_loss_difference_max"],
        stockout=guards["stockout"]["ci95"][1] <= limits["inventory_stockout_difference_max"],
        overflow=guards["overflow"]["ci95"][1] <= limits["inventory_overflow_difference_max"],
        hard_constraints=sum(r["hard_violations"] for r in paired_rows) <= limits["hard_violations_max"],
        runtime=runtime_ratio <= limits["runtime_ratio_max"],
    )
    dense_transfer = contrast(summaries, "paired", "dense", "mean_tracking_loss", protocol, regimes=["diagonal", "misaligned"])
    negative = {c: sum(not r["tracking_pass"] for r in summaries if r["controller"] == c) for c in ("zero", "hold")}
    valid = all(v > 0 for v in negative.values()) and equivalence_error <= protocol["signed_equivalence_tolerance"]
    practical = primary["ci95"][1] < -protocol["minimum_practical_improvement"]
    distinctive = comparisons["shuffled"]["ci95"][1] < 0 and dense_transfer["ci95"][1] <= protocol["dense_transfer_noninferiority_margin"]
    if not valid:
        decision = "evaluator_or_equivalence_invalid"
    elif practical and all(guard_pass.values()) and distinctive:
        decision = "continue_bounded_contextual_coupling_research"
    elif regimes["paired"]["ci95"][1] < -protocol["minimum_practical_improvement"]:
        decision = "pivot_to_conditional_structural_prior"
    else:
        decision = "suspend_exclusive_polar_advantage_claim"
    # Return context minus original context is not, alone, a memory lesion or a
    # claim of preserved general competence; maintain that limit in exports.
    aggregates = {}
    for controller in protocol["controllers"]:
        rows = [r for r in summaries if r["controller"] == controller]
        aggregates[controller] = dict(n_trials=len(rows), passes=sum(r["tracking_pass"] for r in rows),
            mean_tracking_loss=float(np.mean([r["mean_tracking_loss"] for r in rows])),
            mean_action_cost=float(np.mean([r["mean_action_cost"] for r in rows])),
            recovery_fraction=float(np.mean([r["recovery_fraction"] for r in rows])),
            return_loss_change=float(np.mean([r["return_loss_change"] for r in rows])),
            mean_controller_seconds=float(np.mean([r["controller_seconds"] for r in rows])),
            hard_violations=sum(r["hard_violations"] for r in rows))
    return dict(split=split, primary=primary, comparisons=comparisons, regimes=regimes, families=families,
                guards=guards, guard_pass=guard_pass, runtime_ratio=runtime_ratio, dense_transfer=dense_transfer,
                negative_control_failures=negative, max_signed_action_difference=equivalence_error,
                practical_primary_benefit=practical, distinctive_secondary_requirement=distinctive,
                decision=decision if split == "final" else "pilot_not_confirmatory", proposed_final_rule_outcome=decision,
                aggregates=aggregates, scope="Internally designed computational tasks only; no exclusive polar ontology or consciousness conclusion.")


def regenerate(manifest_path, output_dir):
    manifest_path, output_dir = Path(manifest_path), Path(output_dir)
    manifest = read_json(manifest_path)
    root = Path(__file__).resolve().parents[1]
    if manifest["source_sha256"] != source_hashes(root):
        raise ValueError("Current source differs from source hashes recorded before this run")
    protocol = manifest["protocol"]
    if protocol != read_json(root/"docs/study2_protocol.json"):
        raise ValueError("Manifest protocol differs from current frozen protocol")
    seeds = manifest["seeds"]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Duplicate seed declaration")
    if manifest["split"] == "final":
        if set(seeds) & set(protocol["pilot_seeds"]):
            raise ValueError("Pilot/final leakage")
        if seeds != manifest["freeze"]["sample_size"]["final_seeds"]:
            raise ValueError("Final seed set differs from frozen sample size")
        import re
        if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("registration_sha") or ""):
            raise ValueError("Final run lacks prospective registration commit")
        if manifest["freeze"]["source_sha256"] != manifest["source_sha256"] or manifest["freeze"]["protocol_sha256"] != sha256(root/"docs/study2_protocol.json") or not manifest["freeze"].get("reviewed_by"):
            raise ValueError("Final evidence differs from reviewed freeze")
    expected = {(s, f, r, c) for s in seeds for f in protocol["families"] for r in protocol["regimes"] for c in protocol["controllers"]}
    declared = [(x["seed"], x["family"], x["regime"], x["controller"]) for x in manifest["trials"]]
    if set(declared) != expected or len(declared) != len(expected):
        raise ValueError("Manifest has missing, additional or duplicate trials")
    environment_cache = {}
    reader = ArtifactReader(manifest_path.parent, manifest)
    summaries, action_cache = [], {}
    for item in manifest["trials"]:
        if item["environment_path"] not in environment_cache:
            environment_cache[item["environment_path"]] = reader.read(item["environment_path"], item["environment_sha256"])
        environment, trace = environment_cache[item["environment_path"]], reader.read(item["trace_path"], item["trace_sha256"])
        for key in ("seed", "family", "regime", "controller"):
            if trace[key] != item[key]:
                raise ValueError("Trace identity mismatch")
        if environment["config"]["seed"] != item["seed"] or environment["config"]["family"] != item["family"] or environment["config"]["regime"] != item["regime"]:
            raise ValueError("Environment identity mismatch")
        summaries.append(summarize_trial(environment, trace, protocol))
        if item["controller"] in ("paired", "signed_intensity"):
            action_cache[(item["seed"], item["family"], item["regime"], item["controller"])] = np.asarray([r["action"] for r in trace["records"]])
    reader.close()
    equivalence_error = max(float(np.max(np.abs(action_cache[(s,f,r,"paired")]-action_cache[(s,f,r,"signed_intensity")])) )
                            for s in seeds for f in protocol["families"] for r in protocol["regimes"])
    result = analyze(summaries, protocol, manifest["split"], equivalence_error)
    result["registration_sha"] = manifest.get("registration_sha")
    result["manifest_sha256"] = sha256(manifest_path)
    result["trace_count"] = len(summaries)
    if manifest["split"] == "pilot":
        result["sample_size"] = sample_size(summaries, protocol)
    elif "freeze" in manifest:
        result["sample_size"] = manifest["freeze"]["sample_size"]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir/"trial_summaries.json", summaries)
    write_json(output_dir/"decision.json", result)
    with (output_dir/"trial_summaries.csv").open("w", newline="") as stream:
        fields = [k for k in summaries[0] if k not in ("losses", "action_costs", "recoveries", "profile")]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in summaries)
    export(result, summaries, output_dir)
    return result


def export(result, summaries, output_dir):
    output_dir = Path(output_dir)
    primary = result["primary"]
    lines = ["# Study 2: contextual coupling", "", f"Split: **{result['split']}**. {result['trace_count']} complete controller trials.", "",
             "Primary unit: seed, equally averaging the six task-by-regime cells; lower loss is better.", "",
             f"Paired minus planning lesion: **{primary['effect']:.6f}**, 95% paired-seed interval **[{primary['ci95'][0]:.6f}, {primary['ci95'][1]:.6f}]**, n={primary['n']} seeds.", "",
             f"Predeclared decision: **{result['decision']}**.", "", "| Controller | Mean loss | Tracking passes | Mean action cost | Recovery fraction | Seconds/trial |", "|---|---:|---:|---:|---:|---:|"]
    tex = [r"\begin{table}[htbp]", r"\centering\small", r"\begin{tabular}{lrrrr}", r"\toprule", r"Controller & Tracking loss & Passes & Action cost & Recovery \\", r"\midrule"]
    for name, row in result["aggregates"].items():
        lines.append(f"| {name} | {row['mean_tracking_loss']:.6f} | {row['passes']}/{row['n_trials']} | {row['mean_action_cost']:.4f} | {row['recovery_fraction']:.3f} | {row['mean_controller_seconds']:.4f} |")
        escaped = name.replace("_", r"\_")
        tex.append(f"{escaped} & {row['mean_tracking_loss']:.6f} & {row['passes']}/{row['n_trials']} & {row['mean_action_cost']:.3f} & {row['recovery_fraction']:.3f} " + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\caption{Study 2 " + result["split"] + r" evaluation. Tracking passes use the frozen loss, recovery and constraint rule; inventory consequences are evaluated separately.}", r"\label{tab:study2}", r"\end{table}"]
    lines += ["", "## Regime dependence", "", "| Regime | Paired minus lesion | 95% interval |", "|---|---:|---|"]
    for name, c in result["regimes"].items():
        lines.append(f"| {name} | {c['effect']:.6f} | [{c['ci95'][0]:.6f}, {c['ci95'][1]:.6f}] |")
    lines += ["", "## Guardrails", ""]
    for name, passed in result["guard_pass"].items():
        lines.append(f"- {name}: {'met' if passed else 'not met'}")
    lines += ["", f"Observed paired/lesion controller walltime ratio: {result['runtime_ratio']:.3f}; this is machine- and implementation-dependent, not a FLOP count.",
              f"Maximum paired/signed-intensity action difference: {result['max_signed_action_difference']:.3g}.",
              "", "Both environments are internally designed. Inventory adds finite-capacity clipping and stockouts; the shared planner still uses an affine horizon approximation, a documented model mismatch.",
              "Changing the within-pair planner affects subsequent actions and therefore subsequent learning data; it does not preserve identical fitted weights throughout each paired run.",
              "Return-to-initial-context performance measures reacquisition in this task, not general lifelong capacity retention. No consciousness indicator is inferred from these outcomes."]
    if not result["sample_size"]["pilot_precision_qualification"]:
        lines += ["", "The pilot-derived unbounded precision sample size exceeded the predeclared cap. This study does not meet its pilot precision design qualification."]
    (output_dir/"results.md").write_text("\n".join(lines)+"\n")
    (output_dir/"results.tex").write_text("\n".join(tex)+"\n")
    macro = [f"\\newcommand{{\\StudyTwoEffect}}{{{primary['effect']:.6f}}}",
             f"\\newcommand{{\\StudyTwoLow}}{{{primary['ci95'][0]:.6f}}}",
             f"\\newcommand{{\\StudyTwoHigh}}{{{primary['ci95'][1]:.6f}}}",
             f"\\newcommand{{\\StudyTwoSeeds}}{{{primary['n']}}}",
             f"\\newcommand{{\\StudyTwoTrials}}{{{result['trace_count']}}}"]
    (output_dir/"result_macros.tex").write_text("\n".join(macro)+"\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
    names = list(result["aggregates"])
    axes[0].barh(names, [result["aggregates"][n]["mean_tracking_loss"] for n in names], color=["#135c85" if n == "paired" else "#92afbe" for n in names])
    axes[0].invert_yaxis(); axes[0].set_xlabel("Mean weighted state tracking loss")
    regimes = list(result["regimes"])
    effects = [result["regimes"][r]["effect"] for r in regimes]
    low = [e-result["regimes"][r]["ci95"][0] for r,e in zip(regimes,effects)]
    high = [result["regimes"][r]["ci95"][1]-e for r,e in zip(regimes,effects)]
    axes[1].errorbar(effects, regimes, xerr=[low,high], fmt="o", color="#135c85", capsize=4)
    axes[1].axvline(0, color="black", lw=.7); axes[1].axvline(-.002, color="#af6316", ls="--", lw=.8)
    axes[1].set_xlabel("Paired − lesion loss; 95% paired-seed CI")
    fig.savefig(output_dir/"evaluation_summary.pdf")
    fig.savefig(output_dir/"evaluation_summary.png", dpi=150)
    plt.close(fig)
