"""B1-SC v1.0 QA suite. No scientific training is performed."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import unittest

import numpy as np
import torch

from experiments.b1sc_v1_0 import implementation as sc
from b1s.execution.core import EDGES, GenericActor, RoutingActor
from b1s.execution.instrument import NativeEnv, runtime_lock

def write(path: Path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+"\n",encoding="utf-8")

class RuleTests(unittest.TestCase):
    def test_design_contract(self):
        self.assertEqual(sc.validate_design_contract()["status"],"PASS")
    def test_edge_order_exact(self):
        self.assertEqual(tuple(EDGES), sc.EDGE_ORDER)
    def test_panel_exact_and_72_fits(self):
        self.assertEqual(len(sc.PANEL),9); self.assertEqual(len(sc.registry()),72)
    def test_inverse_cycle(self):
        self.assertEqual(sc.GRAPH_MASK["S6"],"100110")
        self.assertEqual(sc.role_active_pairs("S6"),((0,1),(1,2),(2,0)))
    def test_forward_cycle(self):
        self.assertEqual(sc.role_active_pairs("S5"),((0,2),(1,0),(2,1)))
    def test_chain_and_reciprocity(self):
        self.assertEqual(sc.role_active_pairs("S4"),((0,1),(1,2)))
        self.assertEqual(sc.role_active_pairs("S3"),((1,2),(2,1)))
    def test_primary_and_secondary_lesions(self):
        self.assertEqual(sc.primary_lesion("S3"),(3,5))
        self.assertEqual(sc.secondary_lesions("S3"),((3,),(5,)))
        self.assertEqual(sc.primary_lesion("G0"),())
        self.assertEqual(sc.g0_negative_control_lesion(),tuple(range(6)))
    def test_inactive_edge_control(self):
        self.assertEqual(sc.inactive_edge_control("S1"),0)
        self.assertIsNone(sc.inactive_edge_control("GD"))
    def test_causal_window_exact(self):
        self.assertFalse(sc.in_causal_window(8))
        self.assertTrue(all(sc.in_causal_window(i) for i in range(9,41)))
        self.assertFalse(sc.in_causal_window(41))
    def test_competence_strict_loss_boundary(self):
        a=sc.block_competence(95.0,3.0,95.0+sc.LOSS_MARGIN,2.0)
        self.assertFalse(a["loss_component_pass"])
        b=sc.block_competence(95.0-1e-6,3.0,95.0+sc.LOSS_MARGIN,2.0)
        self.assertTrue(b["pass"])
    def test_competence_displacement_inclusive_boundary(self):
        x=sc.block_competence(0,3,10,2)
        self.assertTrue(x["pass"])
    def test_admission_6_of_8(self):
        six=[{"pass":True}]*6+[{"pass":False}]*2
        five=[{"pass":True}]*5+[{"pass":False}]*3
        self.assertTrue(sc.admission_gate(six,{"pass":True})["admitted"])
        self.assertFalse(sc.admission_gate(five,{"pass":True})["admitted"])
        self.assertFalse(sc.admission_gate([{"pass":True}]*8,{"pass":False})["admitted"])
    def test_practical_harm_boundaries(self):
        self.assertTrue(sc.practical_harm(sc.LOSS_MARGIN+1e-6,-1))
        self.assertFalse(sc.practical_harm(sc.LOSS_MARGIN,-1))
        self.assertTrue(sc.practical_harm(-sc.LOSS_MARGIN,1+1e-6))
        self.assertFalse(sc.practical_harm(-sc.LOSS_MARGIN-1e-6,2))
    def test_causal_gate_6_of_8(self):
        self.assertTrue(sc.causal_gate([True]*6+[False]*2,True)["causal_gate_pass"])
        self.assertFalse(sc.causal_gate([True]*5+[False]*3,True)["causal_gate_pass"])
    def test_practical_advantage_boundaries(self):
        self.assertTrue(sc.practical_advantage(sc.LOSS_MARGIN+1e-6,-1))
        self.assertTrue(sc.practical_advantage(-sc.LOSS_MARGIN,1+1e-6))
        self.assertFalse(sc.practical_advantage(sc.LOSS_MARGIN,-1))
    def test_utility_gate_6_of_8(self):
        self.assertTrue(sc.utility_gate([True]*6+[False]*2,True)["utility_gate_pass"])
        self.assertFalse(sc.utility_gate([True]*8,False)["utility_gate_pass"])
    def test_seed_namespaces(self):
        vals=[sc.seed(s,i) for s in sc.ALLOWED_SPLITS for i in range(64)]
        self.assertEqual(len(vals),len(set(vals)))
        with self.assertRaises(RuntimeError): sc.seed("final",0)
    def test_seed_pairing_across_graph_masks(self):
        r=sc.registry()
        for b in range(sc.BLOCKS):
            graph={x["initial_seed"] for x in r if x["block"]==b and x["mask"] is not None}
            self.assertEqual(len(graph),1)
    def test_no_execution_flags(self):
        d=sc.design()
        self.assertFalse(d["execution_boundary"]["execution_authorized"])
        self.assertFalse(d["execution_boundary"]["training_started"])
        self.assertFalse(d["execution_boundary"]["evaluation_started"])
        self.assertFalse(d["execution_boundary"]["start_request_exists"])
        self.assertFalse((sc.ROOT/"START_REQUEST.json").exists())

class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.manual_seed(sc.seed("qa","torch"))

    def test_runtime_lock(self):
        lock=runtime_lock()
        self.assertIn("versions",lock)
        self.assertEqual(lock["versions"]["torch"],"2.2.2+cpu")

    def test_g0_all_route_negative_control_exact(self):
        z=torch.randn(5,93)
        actor=RoutingActor("000000")
        with torch.no_grad():
            mu,_=actor(z)
            lesion,_=actor(z,sc.g0_negative_control_lesion())
        self.assertTrue(torch.equal(mu,lesion))

    def test_sparse_inactive_edge_exact(self):
        z=torch.randn(5,93)
        actor=RoutingActor("000001")
        edge=sc.inactive_edge_control("S1")
        with torch.no_grad():
            mu,_=actor(z)
            ctl,_=actor(z,(edge,))
        self.assertTrue(torch.equal(mu,ctl))

    def test_all_active_lesion_zeroes_received_messages(self):
        z=torch.randn(3,93)
        for role in ("S1","S3","S4","S5","S6","GD"):
            actor=RoutingActor(sc.GRAPH_MASK[role])
            with torch.no_grad():
                _,_,tr=actor(z,sc.primary_lesion(role),trace=True)
            self.assertEqual(float(tr["r"].abs().max()),0.0)

    def test_individual_lesion_only_changes_target_gate(self):
        z=torch.randn(2,93)
        actor=RoutingActor("100110")
        with torch.no_grad():
            _,_,base=actor(z,trace=True)
            for e in sc.role_active_edges("S6"):
                _,_,les=actor(z,(e,),trace=True)
                self.assertTrue(torch.equal(base["m"],les["m"]))
                self.assertTrue(torch.equal(base["h"],les["h"]))
                delta=(base["lambda"]-les["lambda"]).nonzero().flatten().tolist()
                self.assertEqual(delta,[e])

    def test_ppo_rejects_edge_lesion(self):
        actor=GenericActor(); z=torch.randn(2,93)
        with self.assertRaises(ValueError): actor(z,(0,))

    def test_native_environment_smoke_only(self):
        env=NativeEnv()
        try:
            z,info=env.reset(seed=sc.seed("qa","native"))
            self.assertEqual(z.shape,(93,))
            nxt,reward,term,trunc,audit=env.step(np.zeros(12,dtype=np.float32))
            self.assertEqual(nxt.shape,(93,))
            self.assertTrue(math.isfinite(reward))
            self.assertEqual(audit["requested"],audit["admitted"])
        finally:
            env.close()

def run_suite(case) -> dict:
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(case)
    result=unittest.TestResult()
    suite.run(result)
    return {
        "status":"PASS" if result.wasSuccessful() else "FAIL",
        "tests_run":result.testsRun,
        "failures":[str(x[0]) + ": " + x[1] for x in result.failures],
        "errors":[str(x[0]) + ": " + x[1] for x in result.errors],
        "skipped":[str(x) for x in result.skipped],
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    rules=run_suite(RuleTests); write(a.out/"RULE_TESTS.json",rules)
    runtime=run_suite(RuntimeTests); write(a.out/"RUNTIME_TESTS.json",runtime)
    ok=rules["status"]=="PASS" and runtime["status"]=="PASS"
    validation={
        "status":"QA_PASS_NOT_SCIENTIFIC_EXECUTION" if ok else "QA_FAIL",
        "rule_tests":rules["tests_run"],"runtime_tests":runtime["tests_run"],
        "training_performed":False,"evaluation_performed":False,
        "start_request_present":(sc.ROOT/"START_REQUEST.json").exists(),
        "implementation":sc.implementation_summary(),
        **sc.BOUNDARY,
    }
    write(a.out/"VALIDATION.json",validation)
    print(json.dumps(validation,indent=2,sort_keys=True))
    if not ok: raise SystemExit(1)

if __name__=="__main__":
    main()
