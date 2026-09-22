#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
freeze = json.loads((ROOT/"R35_CONFIRM_FREEZE.json").read_text())
adj = json.loads((ROOT/"R35_CONFIRM_ADJUDICATION.json").read_text())
prov = json.loads((ROOT/"PROVENANCE_R35.json").read_text())
summary = (ROOT/"R35_RESULT_SUMMARY.md").read_text()

def git_blob(path):
    b = Path(path).read_bytes()
    return hashlib.sha1(("blob "+str(len(b))).encode()+bytes([0])+b).hexdigest()

checks = []
def check(v, name):
    if not v:
        raise ValueError(name)
    checks.append(name)

check(freeze["status"]=="FROZEN_BEFORE_CONFIRMATORY_SEEDS","freeze status")
check(git_blob(ROOT/"r35_access_necessity.py")==freeze["source_git_blob_sha"],"source matches frozen blob")
check(git_blob(ROOT/"adjudicate_r35.py")==freeze["adjudicator_git_blob_sha"],"adjudicator matches frozen blob")
check(adj["source_git_blob_sha"]==freeze["source_git_blob_sha"],"adjudication source identity")
check(adj["confirmatory_seeds"]==freeze["confirmatory_seeds"],"confirmatory seeds preserved")
check(adj["resolution"]=="R35_A_CORE_NECESSITY_WEAKENED","frozen resolution")
check(adj["necessity_supported"] is False,"universal A necessity not supported")
check(adj["core_necessity_weakened"] is True,"A core necessity weakened")
check(adj["current_negative_control_pass"] is True,"CURRENT_MLP null passes")
check(adj["capable_architectures"]==["RNN","GRU","LSTM"],"three capable architectures")
check(adj["necessity_pattern_architectures"]==["RNN"],"only RNN full pattern")
check(adj["heldout_necessity_pattern_architectures"]==[],"no heldout full pattern")
check(adj["seed_guard_counts"]=={"RNN":11,"GRU":8,"WINDOW_MLP":3,"LSTM":8},"seed guards retained")
check(prov["confirmatory"]["run_id"]==35723294150,"confirmatory run provenance")
check(prov["confirmatory"]["artifact_id"]==10691904220,"confirmatory artifact provenance")
check(prov["thresholds_changed_after_confirmatory_opening"] is False,"thresholds unchanged")
check(prov["experiment_source_changed_after_confirmatory_opening"] is False,"source unchanged")
check(prov["frozen_source"]["git_blob_sha"]==freeze["source_git_blob_sha"],"provenance source blob")
check("R35_A_CORE_NECESSITY_WEAKENED" in summary,"summary frozen resolution")
check("must not be silently overwritten" in summary,"v1.0 historical freeze preserved")

print(json.dumps({"status":"PASS","checks_total":len(checks),"checks":checks},indent=2))
