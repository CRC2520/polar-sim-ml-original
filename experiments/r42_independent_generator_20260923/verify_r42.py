#!/usr/bin/env python3
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
src=ROOT/"r42_independent.py"
freeze=json.loads((ROOT/"R42_CONFIRM_FREEZE.json").read_text())
res=json.loads((ROOT/"RESULTS_EXTRACT.json").read_text())
prov=json.loads((ROOT/"PROVENANCE_R42.json").read_text())
summary=(ROOT/"R42_RESULT_SUMMARY.md").read_text()
checks=[]

def blob(path):
    b=path.read_bytes()
    return hashlib.sha1(("blob "+str(len(b))).encode()+bytes([0])+b).hexdigest()

def check(v,name):
    if not v: raise ValueError(name)
    checks.append(name)

check(freeze["status"]=="FROZEN_BEFORE_CONFIRMATORY_SEEDS","freeze status")
check(blob(src)==freeze["source_git_blob_sha"],"source matches frozen blob")
check(blob(src)==res["frozen_source_git_blob_sha"],"extract source identity")
check(blob(src)==prov["source_git_blob_sha"],"provenance source identity")
check(res["resolution"]=="R42_INDEPENDENT_IMPLEMENTATION_TRANSPORT_PASS_INTERNAL","frozen resolution")
check(res["confirmatory"]["families"]["ring_signed"]["native_pass_count"]==16,"ring 16/16")
check(res["confirmatory"]["families"]["lowrank_skew"]["native_pass_count"]==14,"lowrank 14/16")
check(res["confirmatory"]["families"]["ring_signed"]["median_action_agreement"]==1.0,"ring rotation exact action agreement")
check(res["confirmatory"]["families"]["lowrank_skew"]["median_action_agreement"]==1.0,"lowrank rotation exact action agreement")
check(res["confirmatory"]["families"]["ring_signed"]["median_score_gap"]==0.0,"ring zero score gap")
check(res["confirmatory"]["families"]["lowrank_skew"]["median_score_gap"]==0.0,"lowrank zero score gap")
fails=res["confirmatory"]["families"]["lowrank_skew"]["failed_seeds"]
check([x["seed"] for x in fails]==[2137010,2137014],"adverse seeds preserved")
check(fails[0]["failed_component"]=="R" and fails[0]["R_dz"]<0.20,"R adverse retained")
check(fails[1]["failed_component"]=="C" and fails[1]["C_dz"]<0.20,"C adverse retained")
check(res["boundaries"]["E6b"]=="OPEN_SAME_PROGRAM","E6b remains open")
check(res["boundaries"]["implementation_privilege"]=="NOT_SUPPORTED","implementation privilege unsupported")
check(res["boundaries"]["consciousness"]=="NOT_ESTABLISHED","consciousness boundary")
check(prov["source_changed_after_opening"] is False,"source unchanged after opening")
check(prov["thresholds_changed_after_opening"] is False,"thresholds unchanged after opening")
check(prov["independent_replication"] is False,"not independent replication")
check("14/16" in summary and "16/16" in summary,"summary family counts")
check("E6b remains OPEN" in summary,"summary E6b boundary")

print(json.dumps({"status":"PASS","checks_total":len(checks),"checks":checks},indent=2))
