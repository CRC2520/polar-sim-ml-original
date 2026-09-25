#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from one.api import ONE

PAPER_BWM_COMMIT="118fc36cb3602934466ad2c6087c2b3b441f9f1f"
MODEL_SPEC_COMMIT="f03ca1192fa385d16af96e3ce2422000de06f4e3"
FREEZE="2023_12_bwm_release"
N_DEV=8
N_CONFIRM=16
N_RESERVE=24
MIN_TRIALS=300
MIN_GOOD_UNITS=20

def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def main():
    from brainwidemap import bwm_query
    from brainwidemap.bwm_loading import download_aggregate_tables

    one=ONE(base_url="https://openalyx.internationalbrainlab.org", username="intbrainlab", password="international", silent=True)
    bwm=bwm_query(one=one, freeze=FREEZE).copy()

    clusters_path=download_aggregate_tables(one, type="clusters", tag="2024_Q2_IBL_et_al_BWM")
    trials_path=download_aggregate_tables(one, type="trials", tag="2024_Q2_IBL_et_al_BWM")
    clusters=pd.read_parquet(clusters_path)
    trials=pd.read_parquet(trials_path)

    required_bwm={"pid","eid","probe_name","subject","lab"}
    missing=required_bwm-set(bwm.columns)
    if missing:
        raise RuntimeError(f"bwm_query missing columns: {sorted(missing)}")
    if "pid" not in clusters.columns:
        raise RuntimeError("aggregate clusters table has no pid; selection freeze aborted")
    if "label" not in clusters.columns:
        raise RuntimeError("aggregate clusters table has no label; selection freeze aborted")
    if "eid" not in trials.columns:
        raise RuntimeError("aggregate trials table has no eid; selection freeze aborted")

    # Metadata-only counts. No spike times/firing rates/decoder outcomes are loaded.
    good=clusters.loc[clusters["label"]>=1].groupby("pid").size().rename("n_good_units")
    tcount=trials.groupby("eid").size().rename("n_trials")

    df=bwm.merge(good,left_on="pid",right_index=True,how="left")
    df=df.merge(tcount,left_on="eid",right_index=True,how="left")
    df["n_good_units"]=df["n_good_units"].fillna(0).astype(int)
    df["n_trials"]=df["n_trials"].fillna(0).astype(int)
    df=df[(df["n_good_units"]>=MIN_GOOD_UNITS)&(df["n_trials"]>=MIN_TRIALS)].copy()

    # One insertion per session.
    df["insertion_hash"]=df["pid"].map(lambda x:h("POLAR_E7_INSERTION_V1|"+str(x)))
    df=df.sort_values(["eid","insertion_hash"]).groupby("eid",as_index=False).first()

    # One session per subject.
    df["session_hash"]=df["eid"].map(lambda x:h("POLAR_E7_SESSION_V1|"+str(x)))
    df=df.sort_values(["subject","session_hash"]).groupby("subject",as_index=False).first()

    # Final deterministic panel ranking.
    df["panel_hash"]=[
        h(f"POLAR_E7_PANEL_V1|{sub}|{eid}|{pid}")
        for sub,eid,pid in zip(df["subject"],df["eid"],df["pid"])
    ]
    df=df.sort_values("panel_hash").reset_index(drop=True)

    need=N_DEV+N_CONFIRM+N_RESERVE
    if len(df)<N_DEV+N_CONFIRM:
        raise RuntimeError(f"only {len(df)} eligible subject-level insertions; need at least {N_DEV+N_CONFIRM}")

    rows=[]
    for i,row in df.head(need).iterrows():
        if i<N_DEV:
            phase="development"
        elif i<N_DEV+N_CONFIRM:
            phase="confirmatory"
        else:
            phase="reserve"
        rows.append({
            "rank":int(i+1),
            "phase":phase,
            "subject":str(row["subject"]),
            "lab":str(row["lab"]),
            "eid":str(row["eid"]),
            "pid":str(row["pid"]),
            "probe_name":str(row["probe_name"]),
            "n_trials":int(row["n_trials"]),
            "n_good_units":int(row["n_good_units"]),
            "panel_hash":str(row["panel_hash"]),
        })

    out={
        "campaign":"E7 prospective IBL biological correspondence",
        "mode":"metadata_only_panel_freeze",
        "neural_outcomes_inspected":False,
        "model_spec_commit":MODEL_SPEC_COMMIT,
        "paper_bwm_commit":PAPER_BWM_COMMIT,
        "bwm_freeze":FREEZE,
        "selection_rules":{
            "min_trials":MIN_TRIALS,
            "min_good_units":MIN_GOOD_UNITS,
            "one_insertion_per_session":True,
            "one_session_per_subject":True,
            "development_n":N_DEV,
            "confirmatory_n":N_CONFIRM,
            "reserve_n":N_RESERVE,
        },
        "eligible_subject_level_insertions":int(len(df)),
        "panel":rows,
    }
    Path("e7_panel").mkdir(exist_ok=True)
    Path("e7_panel/E7_PANEL_FREEZE.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    pd.DataFrame(rows).to_csv("e7_panel/E7_PANEL_FREEZE.csv",index=False)
    print(json.dumps({
        "eligible_subject_level_insertions":len(df),
        "development":[r["pid"] for r in rows if r["phase"]=="development"],
        "confirmatory":[r["pid"] for r in rows if r["phase"]=="confirmatory"],
        "reserve_count":sum(r["phase"]=="reserve" for r in rows),
        "neural_outcomes_inspected":False,
    },indent=2))

if __name__=="__main__":
    main()
