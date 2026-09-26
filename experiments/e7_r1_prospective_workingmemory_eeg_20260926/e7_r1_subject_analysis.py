#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from urllib.request import urlopen

import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_ID="on004117"
DATASET_VERSION="v1.0.0"
BASE_URL=f"https://data.nemar.org/{DATASET_ID}/{DATASET_VERSION}"
METADATA_COMMIT="7f607906883e117658534aa2879d59a7a7b5d145"

RUN_MAP={
"001":[1,2,3,4],"002":[1,2,3,4],"003":[1,2,3,4],"004":[1,2,3,4],
"005":[1,2,3,4],"006":[1,2,3,4],"007":[1,2,3,4],"008":[1,2,3,4],
"009":[1,2,3,4],"010":[1,2,3,4],"011":[1,2,3,4],"012":[1,2,3],
"014":[1,2,3],"015":[1,2,3],"016":[1,2,3],"017":[1,2,3],
"018":[1,2,3],"019":[1,2,3],"020":[1,2,3],"021":[1,2,3],
"022":[1,2,3,4,5,6],"023":[1,2,3,4,5],"024":[1,2,3]
}
DEV=["001","002","003","004","005","006","007","008"]
CONF=["009","010","011","012","014","015","016","017","018","019","020","021","022","023","024"]

BANDS={"theta":(4.0,7.0),"alpha":(8.0,12.0),"beta":(13.0,30.0)}
D_ACC=0.58; D_SPEC=0.04
C_ACC=0.40; C_SPEC=0.04
R_ACC=0.58; R_SPEC=0.04

def stable_seed(subject: str, base: int) -> int:
    raw=hashlib.sha256(f"E7R1|{base}|{subject}".encode()).digest()
    return (base + int.from_bytes(raw[:4],"little")) & 0xffffffff

def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True,exist_ok=True)
    with urlopen(url,timeout=180) as r, dest.open("wb") as f:
        shutil.copyfileobj(r,f,length=1024*1024)

def fetch_run(subject: str, run: int, root: Path):
    sub=f"sub-{subject}"
    stem=f"{sub}_ses-01_task-WorkingMemory_run-{run}"
    eegdir=root/sub/"ses-01"/"eeg"
    names={
        "set":f"{stem}_eeg.set",
        "fdt":f"{stem}_eeg.fdt",
        "events":f"{stem}_events.tsv",
        "channels":f"{stem}_channels.tsv",
    }
    paths={k:eegdir/v for k,v in names.items()}
    rel=f"{sub}/ses-01/eeg"
    for k,name in names.items():
        url=f"{BASE_URL}/{rel}/{name}"
        print(f"DOWNLOAD {sub} run={run} {k}",flush=True)
        download(url,paths[k])
    return paths

def artifact_ok(epoch: np.ndarray) -> bool:
    ptp=np.ptp(epoch,axis=-1)
    too_high=np.sum(ptp>500e-6)
    return bool(too_high <= 0.10*len(ptp) and float(np.max(ptp)) <= 1500e-6)

def feature_vector(epoch: np.ndarray, times: np.ndarray, lo: float, hi: float, sfreq: float) -> np.ndarray:
    mask=(times>=lo)&(times<=hi)
    x=epoch[:,mask]
    idx=np.where(mask)[0]
    bins=np.array_split(np.arange(x.shape[1]),5)
    erp=np.concatenate([x[:,b].mean(axis=1) for b in bins],axis=0)
    n=x.shape[1]
    win=np.hanning(n)
    xf=np.fft.rfft(x*win[None,:],axis=-1)
    power=(np.abs(xf)**2)/(np.sum(win**2)*sfreq)
    freqs=np.fft.rfftfreq(n,1.0/sfreq)
    bp=[]
    for flo,fhi in BANDS.values():
        fm=(freqs>=flo)&(freqs<=fhi)
        bp.append(np.log(np.mean(power[:,fm],axis=1)+1e-20))
    return np.concatenate([erp]+bp,axis=0)

def extract_epoch(raw, onset: float, tmin: float, tmax: float):
    sf=float(raw.info["sfreq"])
    start=int(round((onset+tmin)*sf))
    stop=int(round((onset+tmax)*sf))+1
    if start<0 or stop>raw.n_times:
        return None,None
    data=raw.get_data(start=start,stop=stop)
    times=np.arange(data.shape[1])/sf+tmin
    base=(times>=-0.20)&(times<=0.0)
    if not np.any(base):
        return None,None
    data=data-data[:,base].mean(axis=1,keepdims=True)
    if not artifact_ok(data):
        return None,None
    return data,times

def cv_score(X,y,groups):
    X=np.asarray(X,float); y=np.asarray(y); groups=np.asarray(groups)
    n_groups=len(np.unique(groups))
    n_splits=min(4,n_groups)
    if n_splits<3: raise RuntimeError("fewer than 3 runs for CV")
    cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=7107)
    scores=[]
    for tr,te in cv.split(X,y,groups):
        pipe=Pipeline([
            ("scale",StandardScaler()),
            ("clf",LogisticRegression(C=0.1,penalty="l2",solver="lbfgs",max_iter=3000))
        ])
        pipe.fit(X[tr],y[tr])
        pred=pipe.predict(X[te])
        scores.append(balanced_accuracy_score(y[te],pred))
    return float(np.mean(scores))

def permute_within(y, strata, seed):
    y=np.asarray(y).copy(); strata=np.asarray(strata)
    rng=np.random.default_rng(seed)
    for s in np.unique(strata):
        idx=np.where(strata==s)[0]
        y[idx]=rng.permutation(y[idx])
    return y

def analyze_subject(subject: str, outpath: Path, cache: Path):
    XD=[]; yD=[]; gD=[]; sD=[]
    XC=[]; yC=[]; gC=[]
    XR=[]; yR=[]; gR=[]; sR=[]
    raw_sfreqs=[]; eeg_counts=[]

    for run in RUN_MAP[subject]:
        paths=fetch_run(subject,run,cache)
        ch=pd.read_csv(paths["channels"],sep="\t")
        eeg_names=ch.loc[ch["type"].str.upper()=="EEG","name"].astype(str).tolist()
        ev=pd.read_csv(paths["events"],sep="\t")
        raw=mne.io.read_raw_eeglab(paths["set"],preload=True,verbose="ERROR")
        picks=[x for x in eeg_names if x in raw.ch_names]
        if len(picks)<50:
            raise RuntimeError(f"{subject} run {run}: only {len(picks)} EEG channels matched")
        raw.pick(picks)
        raw_sfreqs.append(float(raw.info["sfreq"]))
        eeg_counts.append(len(raw.ch_names))
        raw.filter(1.0,35.0,verbose="ERROR")
        raw.resample(128.0,verbose="ERROR")
        raw.set_eeg_reference("average",projection=False,verbose="ERROR")
        sf=float(raw.info["sfreq"])

        # D: remember vs ignore encoding letters
        drows=ev[(ev["event_type"]=="show_letter") & ev["task_role"].isin(["to_remember","to_ignore"])]
        for _,row in drows.iterrows():
            ep,t=extract_epoch(raw,float(row["onset"]),-0.20,1.05)
            if ep is None: continue
            XD.append(feature_vector(ep,t,0.10,1.00,sf))
            yD.append(1 if row["task_role"]=="to_remember" else 0)
            gD.append(run)
            sD.append(f"{run}|{int(row['memory_cond'])}")

        # C: maintenance load 3/5/7
        crows=ev[(ev["event_type"]=="show_dash") & (ev["task_role"]=="work_memory")]
        for _,row in crows.iterrows():
            ep,t=extract_epoch(raw,float(row["onset"]),-0.20,1.55)
            if ep is None: continue
            XC.append(feature_vector(ep,t,0.40,1.40,sf))
            yC.append(int(row["memory_cond"]))
            gC.append(run)

        # R: target vs not shown probe
        rrows=ev[(ev["event_type"]=="show_letter") & ev["task_role"].isin(["probe_target","probe_not_shown"])]
        for _,row in rrows.iterrows():
            ep,t=extract_epoch(raw,float(row["onset"]),-0.20,0.85)
            if ep is None: continue
            XR.append(feature_vector(ep,t,0.10,0.80,sf))
            yR.append(1 if row["task_role"]=="probe_target" else 0)
            gR.append(run)
            sR.append(f"{run}|{int(row['memory_cond'])}")

        raw.close()
        # immediately delete run bytes
        for p in paths.values():
            try: p.unlink()
            except FileNotFoundError: pass

    yD=np.asarray(yD,int); yC=np.asarray(yC,int); yR=np.asarray(yR,int)
    counts={
        "D_total":int(len(yD)),"D_remember":int(np.sum(yD==1)),"D_ignore":int(np.sum(yD==0)),
        "C_total":int(len(yC)),"C_3":int(np.sum(yC==3)),"C_5":int(np.sum(yC==5)),"C_7":int(np.sum(yC==7)),
        "R_total":int(len(yR)),"R_in":int(np.sum(yR==1)),"R_out":int(np.sum(yR==0)),
    }
    valid=bool(
        counts["D_total"]>=300 and counts["D_remember"]>=100 and counts["D_ignore"]>=100 and
        counts["C_total"]>=60 and counts["C_3"]>=15 and counts["C_5"]>=15 and counts["C_7"]>=15 and
        counts["R_total"]>=60 and counts["R_in"]>=25 and counts["R_out"]>=25
    )
    result={
        "subject":subject,
        "dataset":DATASET_ID,"dataset_version":DATASET_VERSION,
        "metadata_repo_commit":METADATA_COMMIT,
        "runs":RUN_MAP[subject],
        "raw_sfreqs":raw_sfreqs,
        "eeg_channel_counts":eeg_counts,
        "counts":counts,
        "scientifically_valid":valid,
        "thresholds":{"D_accuracy":D_ACC,"D_specificity":D_SPEC,"C_accuracy":C_ACC,"C_specificity":C_SPEC,"R_accuracy":R_ACC,"R_specificity":R_SPEC}
    }

    if valid:
        d_obs=cv_score(XD,yD,gD)
        d_sh=cv_score(XD,permute_within(yD,sD,stable_seed(subject,7111)),gD)
        c_obs=cv_score(XC,yC,gC)
        c_sh=cv_score(XC,permute_within(yC,gC,stable_seed(subject,7113)),gC)
        r_obs=cv_score(XR,yR,gR)
        r_sh=cv_score(XR,permute_within(yR,sR,stable_seed(subject,7117)),gR)

        Ds=d_obs-d_sh; Cs=c_obs-c_sh; Rs=r_obs-r_sh
        Dp=bool(d_obs>=D_ACC and Ds>=D_SPEC)
        Cp=bool(c_obs>=C_ACC and Cs>=C_SPEC)
        Rp=bool(r_obs>=R_ACC and Rs>=R_SPEC)
        result.update({
            "D":{"accuracy":d_obs,"shuffle_accuracy":d_sh,"specificity":Ds,"pass":Dp},
            "C":{"accuracy":c_obs,"shuffle_accuracy":c_sh,"specificity":Cs,"pass":Cp},
            "R":{"accuracy":r_obs,"shuffle_accuracy":r_sh,"specificity":Rs,"pass":Rp},
            "joint_pass":bool(Dp and Cp and Rp)
        })
    else:
        result.update({"D":None,"C":None,"R":None,"joint_pass":False})

    outpath.parent.mkdir(parents=True,exist_ok=True)
    outpath.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    shutil.rmtree(cache,ignore_errors=True)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--subject",required=True,choices=sorted(RUN_MAP))
    p.add_argument("--output",required=True)
    p.add_argument("--cache",required=True)
    a=p.parse_args()
    analyze_subject(a.subject,Path(a.output),Path(a.cache))

if __name__=="__main__":
    main()
