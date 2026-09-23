#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
src=ROOT/"r44_reference.py"
pre=ROOT/"PREREG_R44.md"
res=json.loads((ROOT/"RESULTS_EXTRACT.json").read_text())
prov=json.loads((ROOT/"PROVENANCE_R44.json").read_text())
summary=(ROOT/"R44_RESULT_SUMMARY.md").read_text()

def blob(path):
    b=path.read_bytes()
    return hashlib.sha1(("blob "+str(len(b))).encode()+b"\0"+b).hexdigest()

checks=[]
def check(v,name):
    if not v: raise ValueError(name)
    checks.append(name)

check(blob(src)==res["source_git_blob_sha"],"source identity")
check(blob(pre)==res["protocol_git_blob_sha"],"protocol identity")
check(blob(src)==prov["source_git_blob_sha"],"provenance source identity")
check(blob(pre)==prov["protocol_git_blob_sha"],"provenance protocol identity")
check(res["resolution"]=="R44_PUBLISHED_REFERENCE_COMPATIBILITY_PASS","frozen resolution")
check(res["development"]["resolution"]=="R44_DEVELOPMENT_AUTHORIZE_CONFIRM","development authorized confirmation")
check(res["development"]["summary"]["seed_guard_count"]==8,"development 8/8 guard")
check(res["confirmatory"]["Medium"]["pass"] is True,"Medium pass")
check(res["confirmatory"]["Hard"]["pass"] is True,"Hard pass")
check(res["confirmatory"]["Medium"]["seed_guard_count"]==20,"Medium 20/20 guard")
check(res["confirmatory"]["Hard"]["seed_guard_count"]==20,"Hard 20/20 guard")
check(res["confirmatory"]["Medium"]["median_core_score"]==1.0,"Medium ceiling score")
check(res["confirmatory"]["Hard"]["median_core_score"]==1.0,"Hard ceiling score")
check(res["confirmatory"]["Medium"]["median_iso_gap"]==0.0 and res["confirmatory"]["Hard"]["median_iso_gap"]==0.0,"exact isomorphic equivalence")
check(res["confirmatory"]["Medium"]["median_core_minus_noC"]>0.20,"Medium recurrence effect")
check(res["confirmatory"]["Hard"]["median_core_minus_noC"]>0.20,"Hard recurrence effect")
check(res["confirmatory"]["Medium"]["median_paper_lstm_gap"]==0.0 and res["confirmatory"]["Hard"]["median_paper_lstm_gap"]==0.0,"zero benchmark ceiling gap")
check(prov["source_changed_after_opening"] is False,"source unchanged after opening")
check(prov["thresholds_changed_after_opening"] is False,"thresholds unchanged after opening")
check(prov["confirmatory_seeds_changed_after_opening"] is False,"confirmatory seeds unchanged")
check(prov["independent_replication"] is False,"not independent replication")
check(res["boundaries"]["POLAR_superiority"]=="NOT_ESTABLISHED","no superiority claim")
check(res["boundaries"]["E6b"]=="OPEN","E6b open")
check(res["boundaries"]["D_or_R"]=="NOT_TESTED","D/R not tested")
check(res["boundaries"]["consciousness"]=="NOT_ESTABLISHED","consciousness boundary")
check("benchmark-ceiling compatibility" in summary,"summary comparison boundary")
check("not a direct model-vs-model checkpoint comparison" in summary,"checkpoint boundary")
check("POLAR superiority: NOT ESTABLISHED" in summary,"summary superiority boundary")

print(json.dumps({"status":"PASS","checks_total":len(checks),"checks":checks},indent=2))
