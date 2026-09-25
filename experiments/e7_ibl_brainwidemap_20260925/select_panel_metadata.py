#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from one.api import ONE
from brainwidemap.bwm_loading import download_aggregate_tables

RELEASE_TAG="Brainwidemap"
AGGREGATE_TAG="2026_Q2_IBL_et_al_BWM"
BASE_URL="https://openalyx.internationalbrainlab.org"
REQUIRED_TRIAL_DATASETS=(
    "trials.probabilityLeft",
    "trials.choice",
    "trials.feedbackType",
    "trials.contrastLeft",
    "trials.contrastRight",
    "trials.stimOn_times",
)

def h(text:str)->str:
    return hashlib.sha256(text.encode()).hexdigest()

def dataset_stems(one:ONE,eid:str)->set[str]:
    names=one.list_datasets(eid, details=False)
    out=set()
    for x in names:
        name=Path(str(x)).name
        # Strip namespace and extension while keeping object.attribute.
        parts=name.split(".")
        if len(parts)>=2:
            obj=parts[-3] if len(parts)>=3 and parts[-1] in {"npy","pqt","csv","ssv","tsv"} else parts[-2]
            attr=parts[-2] if len(parts)>=3 and parts[-1] in {"npy","pqt","csv","ssv","tsv"} else parts[-1]
            if obj.startswith("_ibl_"):
                obj=obj[len("_ibl_"):]
            out.add(f"{obj}.{attr}")
        out.add(name)
    return out

def required_trials_available(stems:set[str])->bool:
    # ONE lists may preserve namespace; accept suffix matches.
    for req in REQUIRED_TRIAL_DATASETS:
        obj,attr=req.split(".",1)
        ok=any(
            (req in s)
            or (f"{obj}.{attr}" in s)
            or (f"_ibl_{obj}.{attr}" in s)
            for s in stems
        )
        if not ok:
            return False
    return True

def locate_cluster_keys(df:pd.DataFrame):
    cols=set(df.columns)
    pid_col=next((x for x in ("pid","probe_id","insertion_id") if x in cols),None)
    eid_col=next((x for x in ("eid","session","session_id") if x in cols),None)
    probe_col=next((x for x in ("probe_name","pname","probe") if x in cols),None)
    if "label" not in cols:
        raise RuntimeError(f"aggregate cluster table lacks label column: {sorted(cols)}")
    if pid_col is None and not (eid_col and probe_col):
        raise RuntimeError(
            "cannot map aggregate cluster rows to insertion without pid or eid+probe_name; "
            f"columns={sorted(cols)}"
        )
    return pid_col,eid_col,probe_col

def good_count(df,pid,eid,probe,pid_col,eid_col,probe_col)->int:
    q=df
    if pid_col:
        q=q[q[pid_col].astype(str)==str(pid)]
    else:
        q=q[(q[eid_col].astype(str)==str(eid)) & (q[probe_col].astype(str)==str(probe))]
    return int((q["label"].astype(float)>=1.0).sum())

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",required=True)
    ap.add_argument("--cache-dir",default=".e7_one_cache")
    a=ap.parse_args()

    one=ONE(
        base_url=BASE_URL,
        password="international",
        cache_dir=a.cache_dir,
        silent=True,
    )

    django=f"datasets__tags__name,{RELEASE_TAG}"
    insertions=list(one.alyx.rest("insertions","list",django=django))
    if not insertions:
        raise RuntimeError("Brainwidemap tag query returned no insertions")

    clusters_path=download_aggregate_tables(
        one,
        type="clusters",
        tag=AGGREGATE_TAG,
        overwrite=False,
    )
    if clusters_path is None:
        raise RuntimeError("failed to obtain metadata-only cluster aggregate")
    clusters=pd.read_parquet(clusters_path)
    pid_col,eid_col,probe_col=locate_cluster_keys(clusters)

    candidates=[]
    session_dataset_cache={}
    session_meta_cache={}

    for ins in insertions:
        pid=str(ins["id"])
        eid=str(ins["session"])
        probe=str(ins.get("name") or "")
        si=ins.get("session_info") or {}

        if eid not in session_meta_cache:
            sm=one.alyx.rest("sessions","read",id=eid)
            session_meta_cache[eid]=sm
        sm=session_meta_cache[eid]

        subject=str(si.get("subject") or sm.get("subject") or "")
        lab=str(si.get("lab") or sm.get("lab") or "")
        n_trials=si.get("n_trials")
        if n_trials is None:
            n_trials=sm.get("n_trials")
        try:
            n_trials=int(n_trials)
        except (TypeError,ValueError):
            n_trials=-1

        if eid not in session_dataset_cache:
            stems=dataset_stems(one,eid)
            session_dataset_cache[eid]=stems
        trials_ok=required_trials_available(session_dataset_cache[eid])

        n_good=good_count(
            clusters,pid,eid,probe,pid_col,eid_col,probe_col
        )

        candidate={
            "subject":subject,
            "lab":lab,
            "eid":eid,
            "pid":pid,
            "probe_name":probe,
            "n_trials_metadata":n_trials,
            "n_good_units_metadata":n_good,
            "required_trial_datasets_available":bool(trials_ok),
            "eligible":bool(n_trials>=300 and n_good>=20 and trials_ok),
            "insertion_hash":h(f"POLAR_E7_INSERTION_V1|{pid}"),
            "session_hash":h(f"POLAR_E7_SESSION_V1|{eid}"),
            "panel_hash":h(f"POLAR_E7_PANEL_V1|{subject}|{eid}|{pid}"),
        }
        candidates.append(candidate)

    eligible=[x for x in candidates if x["eligible"]]

    # One insertion per session by preregistered insertion hash.
    by_eid={}
    for x in eligible:
        by_eid.setdefault(x["eid"],[]).append(x)
    session_unique=[
        min(xs,key=lambda z:z["insertion_hash"])
        for xs in by_eid.values()
    ]

    # One session per subject by preregistered session hash.
    by_subject={}
    for x in session_unique:
        by_subject.setdefault(x["subject"],[]).append(x)
    subject_unique=[
        min(xs,key=lambda z:z["session_hash"])
        for xs in by_subject.values()
    ]

    ranked=sorted(subject_unique,key=lambda z:z["panel_hash"])
    if len(ranked)<24:
        raise RuntimeError(f"only {len(ranked)} eligible unique subjects; need >=24")

    development=ranked[:8]
    confirmatory=ranked[8:24]
    reserve=ranked[24:]

    out={
        "campaign":"E7 prospective D/C/R correspondence on IBL Brain-Wide Map",
        "phase":"metadata_only_panel_freeze",
        "neural_data_opened":False,
        "release_tag":RELEASE_TAG,
        "aggregate_qc_metadata_tag":AGGREGATE_TAG,
        "one_base_url":BASE_URL,
        "selection_contract":{
            "min_trials":300,
            "min_good_units":20,
            "one_insertion_per_session":"lowest SHA256 POLAR_E7_INSERTION_V1|pid",
            "one_session_per_subject":"lowest SHA256 POLAR_E7_SESSION_V1|eid",
            "panel_rank":"SHA256 POLAR_E7_PANEL_V1|subject|eid|pid",
            "development_n":8,
            "confirmatory_n":16,
        },
        "aggregate_cluster_schema":{
            "columns":[str(x) for x in clusters.columns],
            "pid_column":pid_col,
            "eid_column":eid_col,
            "probe_column":probe_col,
            "rows":int(len(clusters)),
        },
        "counts":{
            "tagged_insertions":len(candidates),
            "eligible_insertions":len(eligible),
            "unique_sessions":len(session_unique),
            "unique_subjects":len(ranked),
        },
        "development":development,
        "confirmatory":confirmatory,
        "reserve":reserve,
        "all_candidates":candidates,
        "forbidden_outputs":[
            "spike_times","firing_rates","decoder_scores","D_endpoint","C_endpoint","R_endpoint"
        ],
    }
    Path(a.output).parent.mkdir(parents=True,exist_ok=True)
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps({
        "counts":out["counts"],
        "development":[{k:x[k] for k in ("subject","eid","pid","probe_name","n_trials_metadata","n_good_units_metadata","panel_hash")} for x in development],
        "confirmatory":[{k:x[k] for k in ("subject","eid","pid","probe_name","n_trials_metadata","n_good_units_metadata","panel_hash")} for x in confirmatory],
        "neural_data_opened":False,
    },indent=2,sort_keys=True))

if __name__=="__main__":
    main()
