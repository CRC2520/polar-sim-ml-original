"""Execution-readiness QA for B1-SC v1.0.

All optimizer steps in this suite use only the QA namespace and are explicitly
non-scientific. No START_REQUEST is created.
"""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from experiments.b1sc_v1_0 import implementation as sc
from experiments.b1sc_v1_0 import execution as ex
from b1s.execution.core import RoutingActor, GenericActor

class ExecutionRuleTests(unittest.TestCase):
    def test_new_branch_and_frozen_parent(self):
        self.assertEqual(ex.EXEC_BRANCH, "research/b1sc-v1.0-execution-readiness-20260915")
        self.assertEqual(ex.QUALIFIED_BASE_COMMIT, "5ea44a037bae663ec2885bf8b36d2595c08940c4")

    def test_no_start_request(self):
        self.assertFalse(ex.START_REQUEST.exists())

    def test_execution_guard_rejects_absence(self):
        old = dict()
        for key in ("GITHUB_REPOSITORY","GITHUB_REF","GITHUB_RUN_ATTEMPT"):
            old[key] = __import__("os").environ.get(key)
        try:
            __import__("os").environ["GITHUB_REPOSITORY"] = sc.REPOSITORY
            __import__("os").environ["GITHUB_REF"] = "refs/heads/" + ex.EXEC_BRANCH
            __import__("os").environ["GITHUB_RUN_ATTEMPT"] = "1"
            with self.assertRaises(RuntimeError):
                ex.execution_guard()
        finally:
            for key,value in old.items():
                if value is None:
                    __import__("os").environ.pop(key,None)
                else:
                    __import__("os").environ[key]=value

    def test_registry_exact(self):
        reg = sc.registry()
        self.assertEqual(len(reg),72)
        self.assertEqual({x["role"] for x in reg}, set(sc.ALL_ROLES))
        self.assertTrue(all(x["input_mode"]=="raw" for x in reg))
        self.assertTrue(all(x["native_steps"]==1_048_576 for x in reg))

    def test_scientific_profiles_fixed(self):
        self.assertEqual(ex.GRAPH_HP["rollout_steps"],512)
        self.assertEqual(ex.GRAPH_HP["minibatch"],128)
        self.assertEqual(ex.GRAPH_HP["epochs"],4)
        self.assertEqual(ex.PPO_HP["rollout_steps"],2048)
        self.assertEqual(ex.PPO_HP["minibatch"],64)
        self.assertEqual(ex.PPO_HP["epochs"],10)
        for role in sc.ALL_ROLES:
            cfg=ex.config_for(role,0)
            self.assertEqual(cfg["algorithm"],"PPO")
            self.assertEqual(cfg["native_steps"],1_048_576)

    def test_seed_pairing_graph_roles(self):
        for block in range(sc.BLOCKS):
            seeds={ex.config_for(r,block)["initial_seed"] for r in sc.GRAPH_ROLES}
            self.assertEqual(len(seeds),1)
            self.assertNotEqual(ex.config_for("PPO",block)["initial_seed"],next(iter(seeds)))

    def test_no_final_namespace(self):
        with self.assertRaises(RuntimeError):
            sc.seed("final","x")

    def test_causal_window(self):
        self.assertEqual(sc.CAUSAL_WINDOW,(9,40))
        self.assertFalse(sc.in_causal_window(8))
        self.assertTrue(sc.in_causal_window(9))
        self.assertTrue(sc.in_causal_window(40))
        self.assertFalse(sc.in_causal_window(41))

    def test_primary_secondary_contract(self):
        self.assertEqual(sc.primary_lesion("S6"),(0,3,4))
        self.assertEqual(sc.secondary_lesions("S6"),((0,),(3,),(4,)))
        self.assertEqual(sc.g0_negative_control_lesion(),tuple(range(6)))

    def test_utility_gate_synthetic(self):
        blocks=[True]*6+[False]*2
        self.assertTrue(sc.utility_gate(blocks,True)["utility_gate_pass"])
        self.assertFalse(sc.utility_gate(blocks,False)["utility_gate_pass"])

    def test_bootstrap_requires_eight_blocks(self):
        with self.assertRaises(RuntimeError):
            ex._bootstrap_mean([1,2,3],"bad")
        ok=ex._bootstrap_mean([1,2,3,4,5,6,7,8],"qa-bootstrap")
        self.assertEqual(ok["resamples"],10_000)

class ExecutionRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        ex.runtime()

    def test_model_construction_all_roles(self):
        for role in sc.ALL_ROLES:
            actor=ex.model_for(role)
            z=torch.zeros(2,93)
            with torch.no_grad():
                mu,_=actor(z)
            self.assertEqual(tuple(mu.shape),(2,12))

    def test_g0_negative_control_exact(self):
        actor=RoutingActor("000000")
        z=torch.randn(4,93)
        with torch.no_grad():
            a,_=actor(z)
            b,_=actor(z,sc.g0_negative_control_lesion())
        self.assertTrue(torch.equal(a,b))

    def test_inactive_edge_control_exact(self):
        actor=RoutingActor("000001")
        z=torch.randn(4,93)
        edge=sc.inactive_edge_control("S1")
        with torch.no_grad():
            a,_=actor(z)
            b,_=actor(z,(edge,))
        self.assertTrue(torch.equal(a,b))

    def test_ppo_rejects_edge_lesion(self):
        actor=GenericActor()
        with self.assertRaises(RuntimeError):
            ex.deterministic_action(actor,np.zeros(93,dtype=np.float32),(0,))

    def test_qa_microfit_only(self):
        with tempfile.TemporaryDirectory() as td:
            result=ex.qa_microfit(Path(td)/"microfit")
            self.assertEqual(result["status"],"QA_MICROFIT_PASS_NOT_SCIENTIFIC")
            self.assertEqual(result["native_steps"],512)
            self.assertEqual(result["split"],"qa")
            self.assertTrue(result["actor_updated"])
            self.assertFalse(result["scientific_results"])

    def test_qa_causal_control_and_lesion(self):
        torch.manual_seed(sc.seed("qa","causal-actor",bits=32))
        actor=RoutingActor(sc.GRAPH_MASK["S6"])
        intact=ex._run_causal_condition(actor,"S6",0,0,"intact",(),split="qa")
        sham=ex._run_causal_condition(actor,"S6",0,0,"sham",(),split="qa")
        inactive=sc.inactive_edge_control("S6")
        ctl=ex._run_causal_condition(actor,"S6",0,0,"inactive",(inactive,),split="qa")
        lesion=ex._run_causal_condition(actor,"S6",0,0,"all",sc.primary_lesion("S6"),split="qa")
        self.assertEqual(intact["trace_sha256"],sham["trace_sha256"])
        self.assertEqual(intact["trace_sha256"],ctl["trace_sha256"])
        if intact["decision9_state_sha256"] is not None:
            self.assertEqual(intact["decision9_state_sha256"],lesion["decision9_state_sha256"])

    def test_model_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)
            cfg=ex.config_for("S3",0)
            torch.manual_seed(cfg["initial_seed"])
            actor=ex.model_for("S3")
            critic=ex.Value()
            opt=torch.optim.Adam(list(actor.parameters())+list(critic.parameters()),lr=.0003)
            ex.save_model(p/"MODEL.pt",actor,critic,opt,cfg)
            payload=torch.load(p/"MODEL.pt",map_location="cpu")
            other=ex.model_for("S3")
            other.load_state_dict(payload["actor_state"])
            self.assertEqual(ex.model_state_digest(actor),ex.model_state_digest(other))

def run_case(case):
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(case)
    result=unittest.TestResult()
    suite.run(result)
    return dict(
        status="PASS" if result.wasSuccessful() else "FAIL",
        tests_run=result.testsRun,
        failures=[str(x[0])+": "+x[1] for x in result.failures],
        errors=[str(x[0])+": "+x[1] for x in result.errors],
    )

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rules=run_case(ExecutionRuleTests)
    runtime=run_case(ExecutionRuntimeTests)
    validation=dict(
        status="EXECUTION_QA_PASS_NOT_AUTHORIZED" if rules["status"]==runtime["status"]=="PASS" else "EXECUTION_QA_FAIL",
        rule_tests=rules["tests_run"], runtime_tests=runtime["tests_run"],
        rules=rules, runtime=runtime,
        start_request_present=ex.START_REQUEST.exists(),
        scientific_training_performed=False,
        scientific_evaluation_performed=False,
        qa_microfit_permitted=True,
        qa_microfit_scientific_evidence=False,
        **sc.BOUNDARY,
    )
    ex.write_json(a.out/"RULE_TESTS.json",rules)
    ex.write_json(a.out/"RUNTIME_TESTS.json",runtime)
    ex.write_json(a.out/"VALIDATION.json",validation)
    print(json.dumps(validation,indent=2,sort_keys=True))
    if validation["status"]!="EXECUTION_QA_PASS_NOT_AUTHORIZED":
        raise SystemExit(1)

if __name__=="__main__":
    main()
