#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

def git_blob(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(("blob "+str(len(b))).encode()+b"\0"+b).hexdigest()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--results",required=True)
    p.add_argument("--freeze",required=True)
    p.add_argument("--source",required=True)
    p.add_argument("--protocol",required=True)
    p.add_argument("--output",required=True)
    a=p.parse_args()

    r=json.loads(Path(a.results).read_text())
    f=json.loads(Path(a.freeze).read_text())
    sb=git_blob(a.source); pb=git_blob(a.protocol)

    assert f["status"]=="FROZEN_BEFORE_CONFIRMATORY_SEEDS"
    assert sb==f["source_git_blob_sha"],(sb,f["source_git_blob_sha"])
    assert pb==f["protocol_git_blob_sha"],(pb,f["protocol_git_blob_sha"])
    assert r["mode"]=="confirm"
    assert r["confirmatory_seeds"]==f["confirmatory_seeds"]
    assert r["confirmatory_families"]==f["confirmatory_families"]
    assert r["criteria"]==f["criteria"]

    fam={}
    for name in f["confirmatory_families"]:
        s=r["summary"][name]
        fam[name]={
            "native_pass_count":int(s["native_pass_count"]),
            "instrument_valid_count":int(s["instrument_valid_count"]),
            "median_action_agreement":float(s["median_action_agreement"]),
            "median_score_gap":float(s["median_score_gap"]),
            "native_medians":s["native_medians"],
            "pass":bool(
                s["native_pass_count"]>=f["criteria"]["family_pass_required"]
                and s["instrument_valid_count"]==f["criteria"]["family_denominator"]
                and s["median_action_agreement"]>=f["criteria"]["median_action_agreement_min"]
                and s["median_score_gap"]<=f["criteria"]["median_score_gap_max"]
            )
        }

    strong=all(x["pass"] for x in fam.values())
    expected="R42_INDEPENDENT_IMPLEMENTATION_TRANSPORT_PASS_INTERNAL" if strong else "R42_INDEPENDENT_IMPLEMENTATION_TRANSPORT_FAIL"
    # Cross-check the scientific runner's own frozen resolution.
    assert r["resolution"]==expected,(r["resolution"],expected)

    out={
        "campaign":f["campaign"],
        "resolution":expected,
        "strong_pass":strong,
        "source_git_blob_sha":sb,
        "protocol_git_blob_sha":pb,
        "confirmatory_seeds":f["confirmatory_seeds"],
        "families":fam,
        "criteria":f["criteria"],
        "boundaries":f["fixed_boundaries"]
    }
    Path(a.output).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))

if __name__=="__main__":
    main()
