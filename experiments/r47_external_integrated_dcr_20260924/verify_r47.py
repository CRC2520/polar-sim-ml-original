#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEV_SEEDS = list(range(2186001, 2186017))
CONF_SEEDS = list(range(2187001, 2187033))
ISO_TOL = 1e-12

MEDIAN = {
    "full_completion": 0.95,
    "full_return": 0.30,
    "d_effect": 0.35,
    "c_effect": 0.50,
    "r_effect": 0.25,
}
SEED = {
    "full_completion": 0.90,
    "full_return": 0.20,
    "d_effect": 0.20,
    "c_effect": 0.30,
    "r_effect": 0.15,
}


def require(v: bool, name: str) -> None:
    if not v:
        raise AssertionError(name)
    print("PASS:", name)


def recompute_summary_pass(s: dict, required: int) -> bool:
    return bool(
        s["median_full_completion"] >= MEDIAN["full_completion"]
        and s["median_full_return"] >= MEDIAN["full_return"]
        and s["median_d_effect"] >= MEDIAN["d_effect"]
        and s["median_c_effect"] >= MEDIAN["c_effect"]
        and s["median_r_effect"] >= MEDIAN["r_effect"]
        and s["median_iso_completion_gap"] <= ISO_TOL
        and s["median_iso_return_gap"] <= ISO_TOL
        and s["seed_guard_count"] >= required
    )


def recompute_record_guard(r: dict) -> bool:
    return bool(
        r["full_completion"] >= SEED["full_completion"]
        and r["full_return"] >= SEED["full_return"]
        and r["d_effect"] >= SEED["d_effect"]
        and r["c_effect"] >= SEED["c_effect"]
        and r["r_effect"] >= SEED["r_effect"]
        and r["iso_completion_gap"] <= ISO_TOL
        and r["iso_return_gap"] <= ISO_TOL
    )


def verify_common(d: dict, seeds: list[int], required: int) -> None:
    require(d["campaign"] == "R47 external integrated D+C+R", "campaign identity")
    require(d["seeds"] == seeds, "frozen seed set")
    require(d["episodes_per_seed"] == 4, "four paired episodes per seed")
    require(
        d["source"]["commit"] == "e397e5eac9965f9963d18c9f455cd1983bca14fb",
        "exact external POPGym commit",
    )
    require(d["source"]["environment"] == "ConcentrationHard", "external task identity")
    require(len(d["records"]) == len(seeds), "one record per statistical seed")
    require(all(recompute_record_guard(r) == bool(r["seed_guard"]) for r in d["records"]),
            "all seed guards recompute")
    require(
        d["summary"]["seed_guard_count"] == sum(bool(r["seed_guard"]) for r in d["records"]),
        "seed guard count recomputes",
    )
    require(d["summary"]["seed_guard_required"] == required, "frozen seed guard requirement")
    require(d["summary"]["pass"] == recompute_summary_pass(d["summary"], required),
            "summary decision recomputes")
    require(d["summary"]["median_iso_completion_gap"] <= ISO_TOL, "completion iso guard")
    require(d["summary"]["median_iso_return_gap"] <= ISO_TOL, "return iso guard")


def verify_development(path: str) -> bool:
    d = json.loads(Path(path).read_text())
    require(d["mode"] == "development", "development mode")
    verify_common(d, DEV_SEEDS, 13)
    expected = (
        "R47_DEVELOPMENT_AUTHORIZE_CONFIRM"
        if d["summary"]["pass"]
        else "R47_DEVELOPMENT_FAIL_NO_CONFIRM"
    )
    require(d["resolution"] == expected, "development resolution")
    require(d["authorize_confirm"] == d["summary"]["pass"], "development authorization")
    require(d["boundaries"]["E6b"] == "OPEN", "E6b open")
    require(d["boundaries"]["consciousness"] == "NOT_ESTABLISHED", "consciousness boundary")
    return bool(d["authorize_confirm"])


def verify_confirmatory(path: str, authorized: bool) -> None:
    require(authorized, "confirmation only after development authorization")
    d = json.loads(Path(path).read_text())
    require(d["mode"] == "confirmatory", "confirmatory mode")
    verify_common(d, CONF_SEEDS, 28)
    expected = (
        "R47_EXTERNAL_INTEGRATED_DCR_PASS_SAME_PROGRAM"
        if d["summary"]["pass"]
        else "R47_EXTERNAL_INTEGRATED_DCR_FAIL"
    )
    require(d["resolution"] == expected, "final R47 resolution")
    if d["summary"]["pass"]:
        require(
            d["boundaries"]["single_task_external_DCR_conjunction"] == "SUPPORTED_SAME_PROGRAM",
            "single-task D+C+R bounded support",
        )
        require(
            d["boundaries"]["external_online_R_acquisition"] == "SUPPORTED_BOUNDED_TASK",
            "bounded online R acquisition support",
        )
    require(d["boundaries"]["neural_R_learnability_from_reward"] == "NOT_TESTED",
            "neural reward-learnability not inferred")
    require(d["boundaries"]["global_minimality"] == "OPEN", "global minimality open")
    require(d["boundaries"]["POLAR_superiority"] == "NOT_ESTABLISHED", "no POLAR superiority")
    require(d["boundaries"]["E6b"] == "OPEN", "E6b open")
    require(d["boundaries"]["E7"] == "OPEN", "E7 open")
    require(d["boundaries"]["AGI_ASI"] == "NOT_ESTABLISHED", "AGI/ASI not established")
    require(d["boundaries"]["consciousness"] == "NOT_ESTABLISHED", "consciousness not established")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--development", required=True)
    p.add_argument("--confirmatory")
    a = p.parse_args()
    auth = verify_development(a.development)
    if a.confirmatory:
        verify_confirmatory(a.confirmatory, auth)
    print("R47 integrity verification COMPLETE")


if __name__ == "__main__":
    main()
