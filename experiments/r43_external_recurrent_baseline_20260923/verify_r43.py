#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
res=json.loads((ROOT/"RESULTS_EXTRACT.json").read_text())
prov=json.loads((ROOT/"PROVENANCE_R43.json").read_text())
summary=(ROOT/"R43_RESULT_SUMMARY.md").read_text()
def blob(path):
    b=path.read_bytes()
    return hashlib.sha1(("blob "+str(len(b))).encode()+b"\0"+b).hexdigest()
checks=[]
def check(v,name):
    if not v: raise ValueError(name)
    checks.append(name)
check(blob(ROOT/"r43_baseline.py")==res["source_git_blob_sha"],"source identity")
check(blob(ROOT/"PREREG_R43.md")==res["protocol_git_blob_sha"],"protocol identity")
check(res["resolution"]=="R43_DEVELOPMENT_FAIL_NO_CONFIRM","frozen adverse resolution")
check(res["eligible_models"]==1,"exactly one eligible model")
check(res["confirmation_opened"] is False,"confirmation unopened")
check(res["reserved_confirmatory_episode_seeds"]==[2147001,2147040],"reserved seeds retained")
check(all(x["history_benefit"]>0 for x in res["development"]),"positive recurrence effect all seeds")
check(res["development"][2]["recurrent_score"]==1.0,"one perfect recurrent seed retained")
check(prov["scientific_changes_after_opening"] is False,"no post-opening scientific changes")
check(prov["independent_replication"] is False,"not independent replication")
check(res["boundaries"]["strong_external_recurrent_baseline"]=="OPEN","strong baseline remains open")
check(res["boundaries"]["POLAR_superiority"]=="NOT_TESTED","no POLAR superiority claim")
check("R43_DEVELOPMENT_FAIL_NO_CONFIRM" in summary,"summary adverse label")
check("2147001–2147040 remain unopened" in summary,"summary confirmation boundary")
print(json.dumps({"status":"PASS","checks_total":len(checks),"checks":checks},indent=2))
