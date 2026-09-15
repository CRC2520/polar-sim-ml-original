"""Execution-readiness QA for B1-SC-D2 v1.0. QA optimizer steps are non-scientific."""
from __future__ import annotations
import json,os,tempfile,unittest
from pathlib import Path
import torch
from experiments.b1sc_d2_v1_0 import implementation as d2
from experiments.b1sc_d2_v1_0 import execution as ex
from b1s.execution.core import RoutingActor

class RuleTests(unittest.TestCase):
    def test_branch_parent(self):
        self.assertEqual(ex.EXEC_BRANCH,"research/b1sc-d2-v1.0-execution-readiness-20260915"); self.assertEqual(ex.QUALIFIED_IMPLEMENTATION_COMMIT,"8f6c46e37a6b4cf18b39f21d21b232b2339c7fe1")
    def test_no_request(self): self.assertFalse(ex.START_REQUEST.exists())
    def test_guard_rejects_absence(self):
        old={k:os.environ.get(k) for k in ("GITHUB_REPOSITORY","GITHUB_REF","GITHUB_RUN_ATTEMPT")}
        try:
            os.environ["GITHUB_REPOSITORY"]=d2.REPOSITORY; os.environ["GITHUB_REF"]="refs/heads/"+ex.EXEC_BRANCH; os.environ["GITHUB_RUN_ATTEMPT"]="1"
            with self.assertRaises(RuntimeError): ex.execution_guard()
        finally:
            for k,v in old.items():
                if v is None: os.environ.pop(k,None)
                else: os.environ[k]=v
    def test_registry(self):
        self.assertEqual(len(d2.registry()),16)
        for b in range(8):
            on,off=ex.config_for("S6-ON",b),ex.config_for("S6-OFF-TRAIN",b); self.assertEqual(on["initial_seed"],off["initial_seed"]); self.assertEqual(on["policy_sampling_seed"],off["policy_sampling_seed"])
    def test_profile(self): self.assertEqual((ex.HP["rollout_steps"],ex.HP["minibatch"],ex.HP["epochs"],ex.N_ENVS),(512,128,4,8))
    def test_checkpoints(self): self.assertEqual(d2.CHECKPOINTS,(262144,524288,786432,1048576)); self.assertEqual((d2.DIAGNOSTIC_EPISODES,d2.ENDPOINT_EPISODES,d2.CAUSAL_EPISODES),(32,64,64))
    def test_operators(self):
        self.assertEqual(d2.training_lesion("S6-ON"),()); self.assertEqual(d2.training_lesion("S6-OFF-TRAIN"),(0,3,4)); self.assertEqual(ex.evaluation_lesion("S6-ON","permanent_all_active_lesion",0),(0,3,4)); self.assertEqual(ex.evaluation_lesion("S6-ON","inactive_edge_control",0),(1,)); self.assertEqual(ex.evaluation_lesion("S6-ON","window_9_40_all_active_lesion",8),()); self.assertEqual(ex.evaluation_lesion("S6-ON","window_9_40_all_active_lesion",40),(0,3,4))
    def test_bootstrap(self):
        with self.assertRaises(RuntimeError): ex._bootstrap([1,2,3],"bad")
        self.assertEqual(ex._bootstrap([1,2,3,4,5,6,7,8],"qa")["resamples"],10000)
    def test_gates(self):
        self.assertTrue(d2.training_gate([True]*6+[False]*2,True)["H_TRAIN_D2"]); self.assertTrue(d2.online_gate([True]*6+[False]*2,True,True)["H_ONLINE_D2"]); self.assertEqual(d2.adjudicate(True,False),"TRAINING_SCAFFOLD_SUPPORTED")
    def test_no_final_namespace(self):
        with self.assertRaises(RuntimeError): d2.seed("final",{"x":1})

class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): ex.runtime()
    def test_model(self):
        for c in d2.CONDITIONS:
            cfg=ex.config_for(c,0); torch.manual_seed(cfg["initial_seed"]); a=RoutingActor(d2.S6_MASK); mu,_=a(torch.zeros(2,93),d2.training_lesion(c)); self.assertEqual(tuple(mu.shape),(2,12))
    def test_inactive_exact(self):
        a=RoutingActor(d2.S6_MASK); z=torch.randn(4,93)
        with torch.no_grad(): x,_=a(z); y,_=a(z,d2.inactive_control_lesion())
        self.assertTrue(torch.equal(x,y))
    def test_permanent_operator(self):
        a=RoutingActor(d2.S6_MASK); z=torch.randn(2,93)
        with torch.no_grad(): _,_,x=a(z,trace=True); _,_,y=a(z,d2.permanent_lesion(),trace=True)
        self.assertEqual((x["lambda"]-y["lambda"]).nonzero().flatten().tolist(),list(d2.ACTIVE_EDGES)); self.assertTrue(torch.equal(x["h"],y["h"])); self.assertTrue(torch.equal(x["m"],y["m"]))
    def test_qa_controls(self):
        torch.manual_seed(d2.seed("qa",{"purpose":"actor"})); a=RoutingActor(d2.S6_MASK); x=ex.evaluate_actor(a,"S6-ON",0,"qa",2,"intact"); y=ex.evaluate_actor(a,"S6-ON",0,"qa",2,"sham"); z=ex.evaluate_actor(a,"S6-ON",0,"qa",2,"inactive_edge_control"); self.assertEqual([r["trace_sha256"] for r in x],[r["trace_sha256"] for r in y]); self.assertEqual([r["trace_sha256"] for r in x],[r["trace_sha256"] for r in z])
    def test_qa_microfit(self):
        with tempfile.TemporaryDirectory() as td:
            r=ex.qa_microfit(Path(td)/"microfit"); self.assertEqual(r["status"],"D2_EXECUTION_QA_MICROFIT_PASS_NOT_SCIENTIFIC"); self.assertEqual(r["native_steps_per_condition"],512); self.assertTrue(r["paired_initial_actor"]); self.assertTrue(r["off_message_A_unchanged"]); self.assertFalse(r["scientific_training"]); self.assertFalse(r["scientific_evaluation"])
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/"m.pt"; cfg=ex.config_for("S6-ON",2); torch.manual_seed(cfg["initial_seed"]); a=RoutingActor(d2.S6_MASK); torch.save(dict(actor_state=a.state_dict(),configuration=cfg),p); q=torch.load(p,map_location="cpu"); b=RoutingActor(d2.S6_MASK); b.load_state_dict(q["actor_state"]); self.assertEqual(ex.state_digest(a),ex.state_digest(b))

def run_case(case):
    s=unittest.defaultTestLoader.loadTestsFromTestCase(case); r=unittest.TestResult(); s.run(r); return dict(status="PASS" if r.wasSuccessful() else "FAIL",tests_run=r.testsRun,failures=[str(x[0])+": "+x[1] for x in r.failures],errors=[str(x[0])+": "+x[1] for x in r.errors])
def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=False); rules=run_case(RuleTests); runtime=run_case(RuntimeTests); ok=rules["status"]==runtime["status"]=="PASS"; v=dict(status="D2_EXECUTION_QA_PASS_NOT_AUTHORIZED" if ok else "D2_EXECUTION_QA_FAIL",rule_tests=rules["tests_run"],runtime_tests=runtime["tests_run"],rules=rules,runtime=runtime,start_request_present=ex.START_REQUEST.exists(),qa_microfit_performed=True,qa_microfit_scientific_evidence=False,scientific_training_performed=False,scientific_evaluation_performed=False,**d2.BOUNDARY); ex.write_json(a.out/"RULE_TESTS.json",rules); ex.write_json(a.out/"RUNTIME_TESTS.json",runtime); ex.write_json(a.out/"VALIDATION.json",v); print(json.dumps(v,indent=2,sort_keys=True));
    if not ok: raise SystemExit(1)
if __name__=="__main__": main()
