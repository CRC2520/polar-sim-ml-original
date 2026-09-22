#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def source_blob(path):
    b = Path(path).read_bytes()
    return hashlib.sha1(("blob " + str(len(b))).encode() + bytes([0]) + b).hexdigest()

def med(records, arch, key, task=None):
    vals = []
    for r in records:
        q = r["architectures"][arch]
        if task is not None:
            q = q[task]
        vals.append(float(q[key]))
    return float(np.median(vals))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--freeze", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()

    r = json.loads(Path(a.results).read_text())
    f = json.loads(Path(a.freeze).read_text())
    blob = source_blob(a.source)

    assert f["status"] == "FROZEN_BEFORE_CONFIRMATORY_SEEDS"
    assert blob == f["source_git_blob_sha"], (blob, f["source_git_blob_sha"])
    assert r["mode"] == "confirm"
    assert r["seeds"] == f["confirmatory_seeds"]
    assert r["confirmatory_architectures"] == f["architectures"]
    assert r["strict_heldout_architectures"] == f["strict_heldout_architectures"]
    assert r["confirmatory_families"] == f["confirmatory_families"]

    rec = r["records"]
    archs = f["architectures"]
    noncurrent = [x for x in archs if x != "CURRENT_MLP"]
    taskkeys = [
        "loss_contextual", "loss_always", "loss_random", "loss_blocked",
        "contextual_vs_always", "contextual_vs_random", "contextual_vs_blocked",
        "contextual_ratio_to_always", "contextual_ratio_to_blocked",
        "history_contribution_energy", "contextual_gain_vs_current",
    ]
    m = {}
    for arch in archs:
        m[arch] = {
            "selective": {k: med(rec, arch, k, "selective") for k in taskkeys},
            "uniform": {k: med(rec, arch, k, "uniform") for k in taskkeys},
            "access_specificity_interaction": med(rec, arch, "access_specificity_interaction"),
            "selective_contextual_advantage_min": med(rec, arch, "selective_contextual_advantage_min"),
            "uniform_always_advantage": med(rec, arch, "uniform_always_advantage"),
        }

    cap = f["capability"]
    crit = f["criteria"]
    sg = f["seed_guard"]

    def capable(q):
        return (
            q["selective"]["contextual_gain_vs_current"] >= cap["selective_contextual_gain_min"]
            and q["selective"]["history_contribution_energy"] >= cap["history_energy_min"]
            and q["uniform"]["history_contribution_energy"] >= cap["history_energy_min"]
        )

    def pattern(q):
        return (
            capable(q)
            and q["selective"]["contextual_vs_always"] >= crit["selective_vs_always_min"]
            and q["selective"]["contextual_vs_random"] >= crit["selective_vs_random_min"]
            and q["selective"]["contextual_vs_blocked"] >= crit["selective_vs_blocked_min"]
            and q["uniform_always_advantage"] >= crit["uniform_always_advantage_min"]
            and q["access_specificity_interaction"] >= crit["interaction_min"]
        )

    arch_eval = {}
    seed_counts = {}
    for arch in archs:
        arch_eval[arch] = {
            "capable": capable(m[arch]),
            "necessity_pattern": pattern(m[arch]),
        }
        if arch == "CURRENT_MLP":
            continue
        n = 0
        for rr in rec:
            q = rr["architectures"][arch]
            ok = (
                q["selective"]["contextual_gain_vs_current"] >= sg["selective_contextual_gain_min"]
                and q["selective"]["history_contribution_energy"] >= sg["history_energy_min"]
                and q["selective"]["contextual_vs_always"] >= sg["selective_vs_always_min"]
                and q["selective"]["contextual_vs_random"] >= sg["selective_vs_random_min"]
                and q["selective"]["contextual_vs_blocked"] >= sg["selective_vs_blocked_min"]
                and q["uniform_always_advantage"] >= sg["uniform_always_advantage_min"]
                and q["access_specificity_interaction"] >= sg["interaction_min"]
            )
            if ok:
                n += 1
        seed_counts[arch] = n

    current_null = (
        abs(m["CURRENT_MLP"]["selective"]["contextual_vs_always"]) <= f["current_null_abs_max"]
        and abs(m["CURRENT_MLP"]["selective"]["contextual_vs_random"]) <= f["current_null_abs_max"]
        and abs(m["CURRENT_MLP"]["selective"]["contextual_vs_blocked"]) <= f["current_null_abs_max"]
        and abs(m["CURRENT_MLP"]["access_specificity_interaction"]) <= f["current_null_abs_max"]
    )

    capable_archs = [x for x in noncurrent if arch_eval[x]["capable"]]
    passing = [
        x for x in noncurrent
        if arch_eval[x]["necessity_pattern"] and seed_counts[x] >= f["seed_guard_required"]
    ]
    heldout_passing = [x for x in f["strict_heldout_architectures"] if x in passing]

    # A direct weakening result requires enough capable systems but failure of
    # the intervention pattern in all/most systems, including held-outs.
    if (
        len(passing) >= f["minimum_passing_architectures"]
        and len(heldout_passing) >= f["minimum_heldout_passing_architectures"]
        and current_null
    ):
        resolution = "R35_A_CAUSAL_NECESSITY_SUPPORTED_INTERNAL"
    elif (
        len(capable_archs) >= f["minimum_capable_architectures"]
        and len(passing) <= f["max_passing_for_weakening"]
        and current_null
    ):
        resolution = "R35_A_CORE_NECESSITY_WEAKENED"
    else:
        resolution = "R35_INCONCLUSIVE_CAPABILITY_OR_HETEROGENEITY"

    out = {
        "campaign": f["campaign"],
        "source_git_blob_sha": blob,
        "confirmatory_seeds": f["confirmatory_seeds"],
        "architecture_medians": m,
        "architecture_adjudication": arch_eval,
        "capable_architectures": capable_archs,
        "necessity_pattern_architectures": passing,
        "heldout_necessity_pattern_architectures": heldout_passing,
        "seed_guard_counts": seed_counts,
        "current_negative_control_pass": bool(current_null),
        "resolution": resolution,
        "necessity_supported": resolution == "R35_A_CAUSAL_NECESSITY_SUPPORTED_INTERNAL",
        "core_necessity_weakened": resolution == "R35_A_CORE_NECESSITY_WEAKENED",
        "boundaries": f["fixed_boundaries"],
    }
    Path(a.output).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
