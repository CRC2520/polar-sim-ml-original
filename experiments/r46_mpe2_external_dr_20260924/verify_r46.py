#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEV_SEEDS = list(range(2176001, 2176017))
REF_CONFIRM_SEEDS = list(range(2177001, 2177033))
CRYPTO_CONFIRM_SEEDS = list(range(2177101, 2177133))
ISO_TOL = 1e-12


def require(value: bool, label: str) -> None:
    if not value:
        raise AssertionError(label)
    print("PASS:", label)


def verify_development(path: str) -> bool:
    d = json.loads(Path(path).read_text())
    require(d["campaign"] == "R46 external D/R causal transport", "campaign identity")
    require(d["mode"] == "development", "development mode")
    require(d["seeds"] == DEV_SEEDS, "development seeds frozen")
    s = d["summary"]
    authorize = bool(
        s["median_full_score"] >= 0.70
        and s["median_d_effect"] >= 0.20
        and s["median_r_binding_effect"] >= 0.20
        and s["median_r_rebind_effect"] >= 0.10
        and s["median_iso_gap"] <= ISO_TOL
        and s["median_relation_change_fraction"] >= 0.50
        and s["seed_guard_count"] >= 13
    ) if "median_full_score" in s else False
    require(bool(d["authorize_confirm"]) == authorize, "development gate recomputes")
    expected = "R46_DEVELOPMENT_AUTHORIZE_CONFIRM" if authorize else "R46_REFERENCE_DEV_FAIL_NO_CONFIRM"
    require(d["resolution"] == expected, "development resolution")
    require(d["boundaries"]["E6b"] == "OPEN", "E6b remains open")
    require(d["boundaries"]["external_R_learnability"] == "NOT_TESTED", "external R learnability not inferred")
    return authorize


def verify_confirmatory(path: str, development_authorized: bool) -> None:
    require(development_authorized, "confirmation opened only after development PASS")
    d = json.loads(Path(path).read_text())
    require(d["campaign"] == "R46 external D/R causal transport", "campaign identity")
    require(d["mode"] == "confirmatory", "confirmatory mode")
    require(d["reference"]["seeds"] == REF_CONFIRM_SEEDS, "reference confirmation seeds frozen")
    require(d["crypto"]["seeds"] == CRYPTO_CONFIRM_SEEDS, "crypto confirmation seeds frozen")

    r = d["reference"]["summary"]
    r_pass = bool(
        r["median_full_score"] >= 0.70
        and r["median_d_effect"] >= 0.20
        and r["median_r_binding_effect"] >= 0.20
        and r["median_r_rebind_effect"] >= 0.10
        and r["median_iso_gap"] <= ISO_TOL
        and r["median_relation_change_fraction"] >= 0.50
        and r["seed_guard_count"] >= 28
    ) if "median_full_score" in r else False
    require(bool(r["pass"]) == r_pass, "reference confirmation rule recomputes")

    c = d["crypto"]["summary"]
    c_pass = bool(
        c["median_full_reward"] >= 0.65
        and c["median_d_effect"] >= 0.40
        and c["median_r_direct_effect"] >= 0.40
        and c["median_r_rebind_effect"] >= 0.30
        and c["median_iso_gap"] <= ISO_TOL
        and c["seed_guard_count"] >= 26
    )
    require(bool(c["pass"]) == c_pass, "crypto confirmation rule recomputes")

    expected = (
        "R46_EXTERNAL_D_R_TRANSPORT_PASS_SAME_PROGRAM"
        if r_pass and c_pass
        else "R46_EXTERNAL_D_R_TRANSPORT_PARTIAL"
        if r_pass or c_pass
        else "R46_EXTERNAL_D_R_TRANSPORT_FAIL"
    )
    require(d["resolution"] == expected, "final R46 resolution")
    require(d["boundaries"]["E6b"] == "OPEN", "E6b remains open")
    require(d["boundaries"]["E7"] == "OPEN", "E7 remains open")
    require(d["boundaries"]["POLAR_superiority"] == "NOT_ESTABLISHED", "no implementation superiority inference")
    require(d["boundaries"]["consciousness"] == "NOT_ESTABLISHED", "no consciousness inference")
    require(d["boundaries"]["single_task_external_DCR_conjunction"] == "NOT_ESTABLISHED", "no single-task external D+C+R overclaim")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--development", required=True)
    p.add_argument("--confirmatory")
    a = p.parse_args()
    authorized = verify_development(a.development)
    if a.confirmatory:
        verify_confirmatory(a.confirmatory, authorized)
    print("R46 integrity verification COMPLETE")


if __name__ == "__main__":
    main()
