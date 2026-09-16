"""D4 design/implementation QA: anti-leakage, anti-switch and route-gate equivalence."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from experiments.b1sc_d3_v1_0 import implementation as d3
from experiments.b1sc_d4_v1_0 import implementation as d4


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)

    contract = d4.validate_design()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20260916)

    schedule = {
        c: {
            "at_0": d4.training_lambda(c, 0),
            "at_switch_minus_1": d4.training_lambda(c, d4.SWITCH_STEPS - 1),
            "at_switch": d4.training_lambda(c, d4.SWITCH_STEPS),
            "at_terminal": d4.training_lambda(c, d4.NATIVE_STEPS),
        }
        for c in d4.CONDITIONS
    }
    require(schedule["LOCAL-S6-OFF"] == {"at_0":0.0,"at_switch_minus_1":0.0,"at_switch":0.0,"at_terminal":0.0}, "OFF schedule changed")
    require(schedule["LOCAL-S6-ALWAYS"] == {"at_0":1.0,"at_switch_minus_1":1.0,"at_switch":1.0,"at_terminal":1.0}, "ALWAYS schedule changed")
    require(schedule["LOCAL-S6-LATE"] == {"at_0":0.0,"at_switch_minus_1":0.0,"at_switch":1.0,"at_terminal":1.0}, "LATE schedule boundary changed")
    require(d4.diagnostic_lambda("LOCAL-S6-LATE", d4.SWITCH_STEPS) == 0.0, "786432 diagnostic must remain pre-switch")
    require(d4.diagnostic_lambda("LOCAL-S6-LATE", 851_968) == 1.0, "First post-switch diagnostic must use routes on")

    z = torch.randn(7, 93)
    base = d3.D3RoutingActor(d3.S6, d3.LOCAL)
    actor = d4.D4RoutingActor()
    actor.load_state_dict(base.state_dict())
    with torch.no_grad():
        mu_d3_on, _ = base(z)
        mu_d3_off, _ = base(z, lesion=d4.ACTIVE_EDGES)
        mu_d4_on, _, tr_on = actor(z, route_lambda=1.0, trace=True)
        mu_d4_off, _, tr_off = actor(z, route_lambda=0.0, trace=True)
    require(torch.equal(mu_d3_on, mu_d4_on), "D4 lambda=1 is not bitwise D3 LOCAL-S6")
    require(torch.equal(mu_d3_off, mu_d4_off), "D4 lambda=0 is not bitwise all-active D3 lesion")
    require(tuple(float(x) for x in tr_off["edge_lambda"]) == (0.0,)*6, "D4 lambda=0 edge gate changed")
    require(tuple(float(x) for x in tr_on["edge_lambda"]) == (1.0,)*6, "D4 lambda=1 edge gate changed")
    require(tuple(map(tuple, torch.nonzero(tr_on["support"], as_tuple=False).tolist())) == ((0,1),(1,2),(2,0)), "D4 S6 support changed")

    anti_leak = {}
    for node, (lo, hi) in enumerate(d3.LOCAL_SLICES):
        z2 = z.clone()
        mask = torch.ones(93, dtype=torch.bool)
        mask[lo:hi] = False
        z2[:, mask] = torch.randn_like(z2[:, mask]) * 7
        with torch.no_grad():
            _, _, t1 = actor(z, route_lambda=1.0, trace=True)
            _, _, t2 = actor(z2, route_lambda=1.0, trace=True)
        same_h = torch.equal(t1["h"][:, node], t2["h"][:, node])
        require(same_h, f"Remote information leaked into pre-message node {node}")
        anti_leak[str(node)] = same_h

    grad_actor = d4.D4RoutingActor()
    grad_actor.zero_grad(set_to_none=True)
    mu0, _ = grad_actor(z, route_lambda=0.0)
    mu0.square().sum().backward()
    a0 = sum(float(p.grad.abs().sum()) for p in grad_actor.A.parameters() if p.grad is not None)
    require(a0 == 0.0, "Message-network gradient leaked through lambda=0")
    grad_actor.zero_grad(set_to_none=True)
    mu1, _ = grad_actor(z, route_lambda=1.0)
    mu1.square().sum().backward()
    a1 = sum(float(p.grad.abs().sum()) for p in grad_actor.A.parameters() if p.grad is not None)
    require(a1 > 0.0, "Message-network gradient absent at lambda=1")

    require(set(dict(actor.named_parameters())) == set(dict(base.named_parameters())), "D4 trainable parameter names differ from D3")
    for name, p0 in base.named_parameters():
        p1 = dict(actor.named_parameters())[name]
        require(tuple(p0.shape) == tuple(p1.shape) and p0.dtype == p1.dtype, f"D4 trainable shape/dtype mismatch: {name}")

    payload = dict(
        status="D4_TEMPORAL_GATE_QA_PASS_NOT_SCIENCE",
        contract=contract,
        schedule=schedule,
        lambda_on_d3_equivalence="BITWISE_PASS",
        lambda_off_d3_lesion_equivalence="BITWISE_PASS",
        anti_leakage=anti_leak,
        lambda_zero_message_gradient_sum=a0,
        lambda_one_message_gradient_sum=a1,
        snapshot_trainable_compatibility="PASS",
        scientific_training_performed=False,
        scientific_evaluation_performed=False,
        scientific_results_exist=False,
        execution_authorized=False,
        B1E_executed=False,
        final_seeds_generated=False,
    )
    write_json(args.out / "D4_IMPLEMENTATION_QA.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
