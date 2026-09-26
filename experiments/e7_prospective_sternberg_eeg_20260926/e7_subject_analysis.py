#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlopen

import mne
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATASET_ID = "on005095"
DATASET_VERSION = "v1.0.0"
BASE_URL = f"https://data.nemar.org/{DATASET_ID}/{DATASET_VERSION}"
METADATA_COMMIT = "e7246fbd0c0372161e8dbfbf19cdd387442472c7"

DEV_SUBJECTS = [f"{x:02d}" for x in range(4,20)]
CONF_SUBJECTS = [f"{x:02d}" for x in range(21,53)]

TRIAL_CODES = {
    31:(3,0), 32:(3,1),
    61:(6,0), 62:(6,1),
    91:(9,0), 92:(9,1),
    121:(12,0), 122:(12,1),
    151:(15,0), 152:(15,1),
}

PHASE_WINDOWS = {
    "encoding": (0.20,1.40),
    "retention": (1.80,3.30),
    "retrieval": (3.55,4.25),
}
BANDS = {"theta":(4.0,7.0),"alpha":(8.0,12.0),"beta":(13.0,30.0)}
REJECT_PTP_V = 200e-6

D_ACC=0.28; D_SPEC=0.04
C_ACC=0.25; C_SPEC=0.03
R_ACC=0.58; R_SPEC=0.04


def stable_seed(subject: str, base: int) -> int:
    raw=hashlib.sha256(f"E7|{base}|{subject}".encode()).digest()
    return (base + int.from_bytes(raw[:4],"little")) & 0xffffffff


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True,exist_ok=True)
    with urlopen(url, timeout=120) as r, dest.open("wb") as f:
        shutil.copyfileobj(r,f,length=1024*1024)


def subject_paths(subject: str, root: Path) -> Dict[str,Path]:
    sub=f"sub-{subject}"
    stem=f"{sub}_ses-01_task-STERNBERG"
    eegdir=root/sub/"ses-01"/"eeg"
    return {
        "vhdr":eegdir/f"{stem}_eeg.vhdr",
        "vmrk":eegdir/f"{stem}_eeg.vmrk",
        "eeg":eegdir/f"{stem}_eeg.eeg",
        "events":eegdir/f"{stem}_events.tsv",
        "channels":eegdir/f"{stem}_channels.tsv",
        "eegjson":eegdir/f"{stem}_eeg.json",
    }


def fetch_subject(subject: str, root: Path) -> Dict[str,Path]:
    paths=subject_paths(subject,root)
    sub=f"sub-{subject}"
    stem=f"{sub}_ses-01_task-STERNBERG"
    relbase=f"{sub}/ses-01/eeg"
    names={
        "vhdr":f"{stem}_eeg.vhdr",
        "vmrk":f"{stem}_eeg.vmrk",
        "eeg":f"{stem}_eeg.eeg",
        "events":f"{stem}_events.tsv",
        "channels":f"{stem}_channels.tsv",
        "eegjson":f"{stem}_eeg.json",
    }
    for key,name in names.items():
        url=f"{BASE_URL}/{relbase}/{name}"
        print(f"DOWNLOAD {subject} {key} {url}",flush=True)
        download(url,paths[key])
    return paths


def parse_events(path: Path) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
    df=pd.read_csv(path,sep="\t")
    # BIDS export duplicates markers as duration 0 and .001; one unique
    # sample/value pair is retained.
    df=df[["sample","value"]].drop_duplicates()
    df["value"]=pd.to_numeric(df["value"],errors="coerce")
    df=df.dropna()
    df["value"]=df["value"].astype(int)
    df=df[df["value"].isin(TRIAL_CODES)]
    df=df.sort_values("sample")
    samples=df["sample"].to_numpy(dtype=int)
    codes=df["value"].to_numpy(dtype=int)
    set_sizes=np.asarray([TRIAL_CODES[int(v)][0] for v in codes],dtype=int)
    relations=np.asarray([TRIAL_CODES[int(v)][1] for v in codes],dtype=int)
    return samples,set_sizes,relations


def phase_features(data: np.ndarray, times: np.ndarray, lo: float, hi: float, sfreq: float) -> np.ndarray:
    mask=(times>=lo)&(times<=hi)
    x=data[:,:,mask]
    n=x.shape[-1]
    win=np.hanning(n).astype(np.float64)
    xf=np.fft.rfft(x*win[None,None,:],axis=-1)
    power=(np.abs(xf)**2)/(np.sum(win**2)*sfreq)
    freqs=np.fft.rfftfreq(n,1.0/sfreq)
    feats=[]
    for _,(flo,fhi) in BANDS.items():
        fm=(freqs>=flo)&(freqs<=fhi)
        bp=np.mean(power[:,:,fm],axis=-1)
        feats.append(np.log(bp+1e-20))
    # trials x (channels * bands)
    return np.concatenate(feats,axis=1)


def cv_score(X: np.ndarray, y: np.ndarray, strat: np.ndarray) -> float:
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=7007)
    scores=[]
    for tr,te in cv.split(X,strat):
        pipe=Pipeline([
            ("scale",StandardScaler()),
            ("clf",LogisticRegression(C=1.0,penalty="l2",solver="lbfgs",max_iter=2000)),
        ])
        pipe.fit(X[tr],y[tr])
        pred=pipe.predict(X[te])
        scores.append(balanced_accuracy_score(y[te],pred))
    return float(np.mean(scores))


def shuffled_set_size(y: np.ndarray, subject: str) -> np.ndarray:
    rng=np.random.default_rng(stable_seed(subject,7011))
    return rng.permutation(y)


def shuffled_relation(y: np.ndarray, set_sizes: np.ndarray, subject: str) -> np.ndarray:
    rng=np.random.default_rng(stable_seed(subject,7013))
    out=y.copy()
    for ss in sorted(set(set_sizes.tolist())):
        idx=np.where(set_sizes==ss)[0]
        out[idx]=rng.permutation(out[idx])
    return out


def analyze_subject(subject: str, output: Path, cache: Path) -> dict:
    paths=fetch_subject(subject,cache)
    samples,set_sizes,relations=parse_events(paths["events"])
    if len(samples) < 150:
        raise RuntimeError(f"{subject}: only {len(samples)} trial events")

    raw=mne.io.read_raw_brainvision(paths["vhdr"],preload=True,verbose="ERROR")
    raw.pick("eeg")
    sfreq0=float(raw.info["sfreq"])
    # markers use original sample indices from events.tsv
    events=np.c_[samples,np.zeros(len(samples),dtype=int),np.arange(1,len(samples)+1)]
    raw.notch_filter(freqs=[50.0],verbose="ERROR")
    raw.filter(l_freq=1.0,h_freq=35.0,verbose="ERROR")
    raw.set_eeg_reference("average",projection=False,verbose="ERROR")
    raw,events_rs=raw.resample(128.0,events=events,verbose="ERROR")

    epochs=mne.Epochs(
        raw,events_rs,event_id=None,tmin=-0.20,tmax=5.0,baseline=(-0.20,0.0),
        preload=True,reject=None,flat=None,detrend=None,verbose="ERROR"
    )
    data=epochs.get_data(copy=True)
    times=epochs.times.copy()
    sfreq=float(epochs.info["sfreq"])

    ptp=np.ptp(data,axis=-1).max(axis=1)
    keep=ptp<=REJECT_PTP_V
    data=data[keep]
    set_sizes=set_sizes[keep]
    relations=relations[keep]

    counts={str(ss):int(np.sum(set_sizes==ss)) for ss in [3,6,9,12,15]}
    valid=bool(len(data)>=120 and all(counts[str(ss)]>=15 for ss in [3,6,9,12,15]))

    result={
        "subject":subject,
        "dataset":DATASET_ID,
        "dataset_version":DATASET_VERSION,
        "metadata_repo_commit":METADATA_COMMIT,
        "raw_sfreq":sfreq0,
        "analysis_sfreq":sfreq,
        "n_eeg_channels":int(data.shape[1]) if len(data) else int(len(raw.ch_names)),
        "trial_events_total":int(len(samples)),
        "trials_retained":int(len(data)),
        "trials_rejected":int(len(samples)-len(data)),
        "trials_by_set_size":counts,
        "scientifically_valid":valid,
        "thresholds":{
            "D_accuracy":D_ACC,"D_specificity":D_SPEC,
            "C_accuracy":C_ACC,"C_specificity":C_SPEC,
            "R_accuracy":R_ACC,"R_specificity":R_SPEC,
        },
    }

    if valid:
        enc=phase_features(data,times,*PHASE_WINDOWS["encoding"],sfreq)
        ret=phase_features(data,times,*PHASE_WINDOWS["retention"],sfreq)
        qry=phase_features(data,times,*PHASE_WINDOWS["retrieval"],sfreq)

        y_set=set_sizes.copy()
        y_rel=relations.copy()
        strat_set=y_set
        # relation CV preserves both relation and set-size strata
        ss_index={v:i for i,v in enumerate([3,6,9,12,15])}
        strat_rel=np.asarray([2*ss_index[int(s)]+int(r) for s,r in zip(y_set,y_rel)],dtype=int)

        d_obs=cv_score(enc,y_set,strat_set)
        d_sh=cv_score(enc,shuffled_set_size(y_set,subject),strat_set)
        c_obs=cv_score(ret,y_set,strat_set)
        c_sh=cv_score(ret,shuffled_set_size(y_set,subject),strat_set)
        r_obs=cv_score(qry,y_rel,strat_rel)
        r_sh_y=shuffled_relation(y_rel,y_set,subject)
        r_sh=cv_score(qry,r_sh_y,strat_rel)

        D_spec=d_obs-d_sh
        C_spec=c_obs-c_sh
        R_spec=r_obs-r_sh
        D_pass=bool(d_obs>=D_ACC and D_spec>=D_SPEC)
        C_pass=bool(c_obs>=C_ACC and C_spec>=C_SPEC)
        R_pass=bool(r_obs>=R_ACC and R_spec>=R_SPEC)

        result.update({
            "D":{"accuracy":d_obs,"shuffle_accuracy":d_sh,"specificity":D_spec,"pass":D_pass},
            "C":{"accuracy":c_obs,"shuffle_accuracy":c_sh,"specificity":C_spec,"pass":C_pass},
            "R":{"accuracy":r_obs,"shuffle_accuracy":r_sh,"specificity":R_spec,"pass":R_pass},
            "joint_pass":bool(D_pass and C_pass and R_pass),
        })
    else:
        result.update({"D":None,"C":None,"R":None,"joint_pass":False})

    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps(result,indent=2,sort_keys=True))
    raw.close()
    shutil.rmtree(cache,ignore_errors=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--subject",required=True)
    p.add_argument("--output",required=True)
    p.add_argument("--cache",required=True)
    a=p.parse_args()
    analyze_subject(a.subject,Path(a.output),Path(a.cache))

if __name__=="__main__":
    main()
