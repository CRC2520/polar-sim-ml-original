#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
freeze=json.loads((ROOT/"R36_R1_CONFIRM_FREEZE.json").read_text())
adj=json.loads((ROOT/"R36_R1_CONFIRM_ADJUDICATION.json").read_text())
prov=json.loads((ROOT/"PROVENANCE_R36_R1.json").read_text())
summary=(ROOT/"R36_R1_RESULT_SUMMARY.md").read_text()
checks=[]
def blob(path):
    b=Path(path).read_bytes()
    return hashlib.sha1(("blob "+str(len(b))).encode()+bytes([0])+b).hexdigest()
def check(v,name):
    if not v: raise ValueError(name)
    checks.append(name)
check(freeze["status"]=="FROZEN_BEFORE_CONFIRMATORY_SEEDS","freeze status")
check(blob(ROOT/"r36_r1_metacog.py")==freeze["source_git_blob_sha"],"source frozen identity")
check(blob(ROOT/"adjudicate_r36_r1.py")==freeze["adjudicator_git_blob_sha"],"adjudicator frozen identity")
check(adj["resolution"]=="R36_R1_COMPONENTS_PASS_JOINT_FAIL","frozen resolution")
check(adj["component_pass"] is True,"component pass")
check(adj["median_pass"] is True,"median pass")
check(adj["joint_pass"] is False,"joint fail")
check(adj["joint_pass_count"]==7,"joint 7/12")
check(adj["component_counts"]["metacog_calibration"]==10,"calibration 10/12")
check(adj["component_counts"]["metacog_function"]==12,"gate 12/12")
check(adj["component_counts"]["planning"]==9,"planning 9/12")
check(adj["component_counts"]["planning_invocation"]==12,"invocation 12/12")
check(adj["component_counts"]["D"]==12 and adj["component_counts"]["C"]==11 and adj["component_counts"]["R"]==11,"DCR retained")
check(prov["confirmatory"]["run_id"]==35744364683,"confirmatory provenance")
check(prov["confirmatory"]["artifact_id"]==10702766030,"artifact provenance")
check(prov["thresholds_changed_after_confirmatory_opening"] is False,"thresholds unchanged")
check(prov["experiment_source_changed_after_confirmatory_opening"] is False,"source unchanged")
check(prov["metric_definition_changed_after_confirmatory_opening"] is False,"metric unchanged after opening")
check("Full same-agent conjunction: **7/12**" in summary,"joint failure explicit")
print(json.dumps({"status":"PASS","checks_total":len(checks),"checks":checks},indent=2))
