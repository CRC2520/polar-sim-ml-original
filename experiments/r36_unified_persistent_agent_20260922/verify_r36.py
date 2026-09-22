#!/usr/bin/env python3
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
freeze=json.loads((ROOT/"R36_CONFIRM_FREEZE.json").read_text())
adj=json.loads((ROOT/"R36_CONFIRM_ADJUDICATION.json").read_text())
prov=json.loads((ROOT/"PROVENANCE_R36.json").read_text())
summary=(ROOT/"R36_RESULT_SUMMARY.md").read_text()
checks=[]

def blob(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(("blob "+str(len(b))).encode()+bytes([0])+b).hexdigest()

def check(v,name):
    if not v: raise ValueError(name)
    checks.append(name)

check(freeze["status"]=="FROZEN_BEFORE_CONFIRMATORY_SEEDS","freeze status")
check(blob(ROOT/"r36_unified_persistent_agent.py")==freeze["source_git_blob_sha"],"source frozen identity")
check(blob(ROOT/"adjudicate_r36.py")==freeze["adjudicator_git_blob_sha"],"adjudicator frozen identity")
check(adj["resolution"]=="R36_FAIL","frozen FAIL retained")
check(adj["strong_pass"] is False,"strong pass false")
check(adj["joint_pass_count"]==4,"joint pass 4/12")
check(adj["joint_required"]==9,"joint requirement 9/12")
check(adj["component_counts"]["D"]==12,"D 12/12")
check(adj["component_counts"]["C"]==11,"C 11/12")
check(adj["component_counts"]["R"]==11,"R 11/12")
check(adj["component_counts"]["metacog_calibration"]==7,"metacog calibration 7/12")
check(adj["component_counts"]["metacog_function"]==8,"metacog function 8/12")
check(adj["component_counts"]["planning"]==9,"planning 9/12")
check(all(adj["median_checks"].values()),"all median checks pass")
check(prov["confirmatory"]["run_id"]==35738004777,"confirmatory run provenance")
check(prov["confirmatory"]["artifact_id"]==10697874816,"artifact provenance")
check(prov["thresholds_changed_after_confirmatory_opening"] is False,"thresholds unchanged")
check(prov["experiment_source_changed_after_confirmatory_opening"] is False,"source unchanged")
check("R36_FAIL" in summary,"summary resolution")
check("Full conjunction in the same seed: **4/12**" in summary,"joint failure explicit")

print(json.dumps({"status":"PASS","checks_total":len(checks),"checks":checks},indent=2))
