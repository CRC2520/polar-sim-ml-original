#!/usr/bin/env python3
"""Integrity verifier for frozen R45 aggregate results.

This script does not run or alter the experiment. It checks that an exported
development/confirmatory adjudication matches the preregistered R45 design.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEV_TRAIN_SEEDS = [2166001, 2166002, 2166003]
CONF_TRAIN_SEEDS = [2167001, 2167002, 2167003]
COMPETENCE_SCORE = 0.90
HISTORY_BENEFIT = 0.15
CORE_MARGIN = -0.05
ISO_TOL = 1e-12


def require(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print("PASS:", msg)


def verify_checkpoint_summary(s):
    require(int(s["n_eval_seeds"]) == 12, "12 held-out evaluation seeds")
    require("median_lstm_intact_score" in s, "intact LSTM score present")
    require("median_history_benefit" in s, "history-benefit endpoint present")
    expected = (
        float(s["median_lstm_intact_score"]) >= COMPETENCE_SCORE
        and float(s["median_history_benefit"]) >= HISTORY_BENEFIT
    )
    require(bool(s["eligible"]) == expected, "eligibility follows frozen thresholds")


def verify_development(path):
    d = json.loads(Path(path).read_text())
    require(d["campaign"] == "R45 official POPGym LSTM reproduction", "campaign identity")
    require(d["phase"] == "development", "development phase")
    cps = d["checkpoints"]
    seeds = [int(x["train_seed"]) for x in cps]
    require(seeds == DEV_TRAIN_SEEDS, "development training seeds unchanged")
    for cp in cps:
        verify_checkpoint_summary(cp["summary"])
    eligible = sum(bool(x["summary"]["eligible"]) for x in cps)
    require(int(d["eligible_checkpoints"]) == eligible, "eligible checkpoint count")
    require(int(d["required_eligible_checkpoints"]) == 2, "2/3 frozen development gate")
    authorize = eligible >= 2
    require(bool(d["authorize_confirm"]) == authorize, "confirmation authorization rule")
    expected_resolution = (
        "R45_DEVELOPMENT_AUTHORIZE_CONFIRM"
        if authorize
        else "R45_OFFICIAL_LSTM_REPRODUCTION_DEV_FAIL_NO_CONFIRM"
    )
    require(d["resolution"] == expected_resolution, "development resolution label")
    require(d["boundaries"]["POLAR_superiority"] == "NOT_ESTABLISHED", "no superiority inference")
    require(d["boundaries"]["E6b"] == "OPEN", "E6b remains open")
    return authorize


def verify_confirmatory(path):
    d = json.loads(Path(path).read_text())
    require(d["campaign"] == "R45 official POPGym LSTM reproduction", "campaign identity")
    require(d["phase"] == "confirmatory", "confirmatory phase")
    require(set(d["levels"]) == {"Medium", "Hard"}, "Medium and Hard only")
    all_pass = True
    for level in ("Medium", "Hard"):
        s = d["levels"][level]
        cps = s["checkpoints"]
        seeds = [int(x["train_seed"]) for x in cps]
        require(seeds == CONF_TRAIN_SEEDS, f"{level} confirmatory training seeds unchanged")
        for cp in cps:
            verify_checkpoint_summary(cp["summary"])
        eligible = sum(bool(x["summary"]["eligible"]) for x in cps)
        require(int(s["eligible_checkpoints"]) == eligible, f"{level} eligible checkpoint count")
        require(int(s["required_eligible_checkpoints"]) == 2, f"{level} 2/3 competence gate")
        require(abs(float(s["noninferiority_margin"]) - CORE_MARGIN) <= 1e-15,
                f"{level} frozen -0.05 compatibility margin")
        expected = (
            eligible >= 2
            and float(s["median_core_score"]) >= COMPETENCE_SCORE
            and float(s["median_iso_gap"]) <= ISO_TOL
            and float(s["median_core_minus_lstm"]) >= CORE_MARGIN
        )
        require(bool(s["pass"]) == expected, f"{level} frozen confirmatory rule")
        all_pass = all_pass and expected
    expected_resolution = (
        "R45_DIRECT_REPRODUCED_CHECKPOINT_COMPATIBILITY_PASS"
        if all_pass
        else "R45_DIRECT_REPRODUCED_CHECKPOINT_COMPATIBILITY_FAIL"
    )
    require(d["resolution"] == expected_resolution, "confirmatory resolution label")
    require(d["boundaries"]["POLAR_superiority"] == "NOT_ESTABLISHED", "no superiority inference")
    require(d["boundaries"]["external_D_or_R"] == "NOT_TESTED", "external D/R not tested")
    require(d["boundaries"]["E6b"] == "OPEN", "E6b remains open")
    require(d["boundaries"]["consciousness"] == "NOT_ESTABLISHED", "consciousness not established")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--development", required=True)
    p.add_argument("--confirmatory")
    args = p.parse_args()
    authorize = verify_development(args.development)
    if args.confirmatory:
        require(authorize, "confirmatory result only after authorized development")
        verify_confirmatory(args.confirmatory)
    print("R45 integrity verification COMPLETE")


if __name__ == "__main__":
    main()
