#!/usr/bin/env python3
import hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
res=json.loads((HERE/"RESULTS_EXTRACT.json").read_text())
summary=(HERE/"RESULTS_SUMMARY.md").read_text()
checks=[]

def blob(path):
    b=path.read_bytes()
    return hashlib.sha1(("blob "+str(len(b))).encode()+bytes([0])+b).hexdigest()
def check(v,name):
    if not v: raise ValueError(name)
    checks.append(name)

expected={
"PROTOCOL_R41.md":"3f2014eb608f7a091b025455c4543aeca11f8a7a",
"p0_memory_baseline.py":"8bb586e9c2c43b09eaaa82f67839fc0c355277ce",
"p1_core_transport.py":"d899de22c2ca9e7eb6a2db60f2b1e8284a2d0d90",
"p2_episodic_identity.py":"b12906479a73f4d5b4fb3447b1b0c78ced9a69b1",
}
for name,sha in expected.items():
    check(blob(HERE/name)==sha,f"executed source unchanged {name}")

check(res["run_id"]==35818698334,"workflow run")
check(res["execution_head"]=="59f13bbcb303f141e8f3218cd029c9fecaaca68b","execution head")
check(res["scientific_changes_after_opening"] is False,"no post-opening scientific changes")

p0=res["P0"]
check(p0["resolution"]=="P0_DEV_FAIL_NO_CONFIRM","P0 adverse resolution")
check(p0["confirmation_opened"] is False,"P0 confirmation unopened")
check(all(not x["eligible"] for x in p0["development"]),"P0 0/3 eligible")
check(p0["supervised_instrument"]["pass"] is False,"P0 task instrument adverse retained")
check(p0["POLAR_superiority"]=="NOT_TESTED","no superiority inference")

p1=res["P1"]
check(p1["resolution"]=="P1_SCALE_NORMALIZED_CORE_TRANSPORT_PASS","P1 frozen resolution")
check(p1["standard"]["core_pass"]==30 and p1["stress"]["core_pass"]==31,"P1 joint counts")
check(p1["standard"]["C_count"]==30 and p1["stress"]["C_count"]==31,"P1 C counts")
check(p1["standard"]["medians"]["noC_cost"] < p1["standard"]["medians"]["cost"],"standard NO_C adverse boundary")
check(p1["stress"]["medians"]["noC_cost"] < p1["stress"]["medians"]["cost"],"stress NO_C adverse boundary")
check("no universal closed-loop benefit" in p1["interpretation"],"P1 limited interpretation")

p2=res["P2"]
check(p2["resolution"]=="P2_EPISODIC_OR_IDENTITY_PARTIAL_FAIL","P2 frozen resolution")
check(p2["archive_deletion"]["positive_count"]==11 and p2["archive_deletion"]["pass"] is False,"P2 deletion fail")
check(p2["profile_specificity"]["nonnegative_count"]==16 and p2["profile_specificity"]["pass"] is False,"P2 profile fail")
check(p2["history_fork"]["excess_ge_005_count"]==10 and p2["history_fork"]["pass"] is False,"P2 fork fail")

b=res["retained_boundaries"]
check(b["core"]=="POLAR Core v1.1 D+C+R with conditional A unchanged","Core unchanged")
check(b["global_minimality"]=="OPEN","minimality open")
check(b["E6b"]=="OPEN" and b["E7"]=="OPEN","external validation open")
check(b["consciousness"]=="NOT_ESTABLISHED","consciousness not established")
check("P0_DEV_FAIL_NO_CONFIRM" in summary and "P1_SCALE_NORMALIZED_CORE_TRANSPORT_PASS" in summary and "P2_EPISODIC_OR_IDENTITY_PARTIAL_FAIL" in summary,"summary retains all outcomes")

print(json.dumps({"status":"PASS","checks_total":len(checks),"checks":checks},indent=2))
