"""Source-defined, finite controller families, frozen before development scores.

These are transparent decision rules, not a trained neural architecture. The
competitive adaptation is selection among eight predeclared configurations.
All arms receive the same complete public three-cell observation. A nominated
route is a deterministic feature, never privileged task truth.
"""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction

from b1.contracts.loader import load_contracts

CONFLICT_ID = "B1-CONFLICT-P5-C4-REFILL-001"
# Verify frozen inputs once when the isolated implementation is imported.
# The evaluation gate separately rechecks hashes before each frozen stage.
_CONTRACTS = load_contracts()


class ContractConflict(RuntimeError):
    conflict_id = CONFLICT_ID


class ControllerFailure(RuntimeError):
    pass


@dataclass(frozen=True)
class Limits:
    # Provisional engineering caps, NOT a completed smallest-cap certification.
    memory_scalars: int = 4096
    decision_operations: int = 8192
    planning_horizon: int = 1
    latency_seconds: float | None = None


def get_configurations(comparator=None):
    """Return the entire immutable-before-scores grid; never filter by outcome."""
    grid = [{"config_id": f"cfg{i:02d}",
             "receiver_policy": "domain_aware" if not (i & 1) else "reactive",
             "source_schedule": "on_demand" if not (i & 2) else "proactive",
             "tie_break": "lowest" if not (i & 4) else "highest",
             # The graph itself is part of the same eight-slot search.
             "generic_sources": [(0, 1, 2), (1, 2, 0), (2, 0, 1), (1, 0, 2),
                                 (2, 1, 0), (0, 2, 1), (0, 1, 2), (2, 0, 1)][i]}
            for i in range(8)]
    if comparator == "C4":
        return [{"config_id": "fixed", "receiver_policy": "conventional",
                 "source_schedule": "frozen", "tie_break": "lowest",
                 "generic_sources": (0, 1, 2)}]
    if comparator == "C6":
        return grid  # Exact images of C1 slots; no separate search or fit.
    if comparator is not None and comparator not in {"C0", "C1", "C2", "C3", "C5"}:
        raise ValueError("Unknown comparator")
    return grid


def implementation_choices():
    return {
        "version": "b1-controller-family-v1",
        "selection": "mean primary tuning loss; then useful operations; then config_id",
        "training": "finite configuration selection; no hidden optimization or task fitting",
        "search_slots": {c: len(get_configurations(c)) for c in
                         ("C0", "C1", "C2", "C3", "C4", "C5")},
        "C0": "no nominated input; complete raw records and conventional reconstruction allowed",
        "C1": "same-cell typed route plus complete raw records",
        "C2": "run both fixed cycles separately and average BOTH losses, never pick a cycle",
        "C3": "three generic typed edges; eight frozen slots include identity connectivity",
        "C4": {"P5": {"status": "blocked", "conflict_id": CONFLICT_ID},
               "P6": "fixed last-live-feedback selector with report fallback",
               "P7": "fixed version-aware maintenance state machine, lowest-ID ties"},
        "C5": "six typed slots and exactly twice C3 memory/decision allowance; descriptive",
        "C6": "exact rational conjugacy; same configuration, no independent optimization",
        "resource_unit": "instrumented scalar-copy/read, record-inspection, route-slot and decision primitive; not CPU instructions",
        "resource_status": "provisional caps; smallest common cap and latency certification pending",
        "limits": Limits().__dict__,
        "configs": get_configurations(),
    }


def scalar_count(value):
    if isinstance(value, dict):
        return sum(1 + scalar_count(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(scalar_count(v) for v in value)
    return 1


class Controller:
    def __init__(self, pilot, comparator, config_id="cfg00", cycle=1,
                 lesion=False, limits=None):
        if pilot not in {"P5", "P6", "P7"}:
            raise ValueError("Unknown pilot")
        if comparator == "C6":
            raise ValueError("Use C6Controller for the full conjugated realization")
        if comparator == "C4" and pilot == "P5":
            raise ContractConflict(CONFLICT_ID + ": frozen JSON and protocol prescribe different actions")
        if cycle not in (1, 2):
            raise ValueError("C2 requires cycle 1 or 2, both retained by evaluation")
        if lesion and comparator != "C1":
            raise ValueError("Acute route lesion is defined on a frozen C1 realization")
        configs = get_configurations(comparator)
        if comparator == "C4":
            config_id = "fixed"
        matching = [c for c in configs if c["config_id"] == config_id]
        if not matching:
            raise ValueError("Configuration is outside the frozen search space")
        contracts = _CONTRACTS
        self.pilot, self.comparator = pilot, comparator
        self.config = deepcopy(matching[0])
        self.constants = contracts.pilot(pilot)["task_contract"]["constants"]
        self.clock = contracts.pilots["shared_contract"]["clock"]
        self.cycle, self.lesion = cycle, lesion
        base = limits or Limits()
        if base.planning_horizon != 1:
            raise ValueError("B0 permits exactly one future epoch of planning")
        if base.memory_scalars <= 0 or base.decision_operations <= 0:
            raise ValueError("Resource budgets must be positive")
        self.limits = Limits(base.memory_scalars * (2 if comparator == "C5" else 1),
                             base.decision_operations * (2 if comparator == "C5" else 1),
                             1, base.latency_seconds)
        # Explicit paired parameters and memory are part of the C6 transform.
        self.parameters = {"activation_gain": {"A": Fraction(1), "B": Fraction(1)},
                           "configuration": deepcopy(self.config)}
        self.memory = {"last_actions": [], "raw_history": []}
        self.last_decision_report = {}
        self._operations = 0

    def _tick(self, n=1):
        self._operations += n
        if self._operations > self.limits.decision_operations:
            raise ControllerFailure("decision_budget_exhausted")

    def _route(self, obs):
        """Feature calculation always runs, including C0 and the acute sham."""
        self._tick()
        cid, epoch = obs["cell_id"], obs["epoch"]
        if self.pilot == "P5":
            pending = obs.get("pending_deliveries", [])
            records = pending.items() if isinstance(pending, dict) else enumerate(pending)
            forecast = 0
            due = epoch + 1
            for key, record in records:
                self._tick()
                if isinstance(record, dict):
                    scheduled = record.get("due_epoch", record.get("epoch", key))
                    units = record.get("units", record.get("dose", record.get("amount", 0)))
                else:
                    scheduled, units = key, record
                if int(scheduled) == due:
                    forecast += units
            return {"source_cell_id": cid, "scheduled_delivery_epoch": due,
                    "delivered_unit_forecast": forecast}
        if self.pilot == "P6":
            arrived = list(obs.get("arriving_reports", []))
            eligible = []
            for r in arrived:
                self._tick()
                if (r.get("transfer_eligible") and r.get("report_value") == 1
                        and r.get("candidate_id") in obs["routine_catalogue"]):
                    eligible.append(r)
            eligible.sort(key=lambda r: (r["candidate_id"], r.get("assay_epoch", 0)))
            self._tick(len(eligible) * max(1, len(eligible).bit_length()))
            r = eligible[0] if eligible else {}
            return {"source_cell_id": cid, "candidate_id": r.get("candidate_id"),
                    "report_value": r.get("report_value", 0),
                    "transfer_eligible": r.get("transfer_eligible", False),
                    "report_epoch": r.get("report_epoch", epoch)}
        repaired = [r for r in obs.get("completion_events", [])
                    if r.get("operation", r.get("type")) in {"repair", "drift_repair"}]
        self._tick(len(obs.get("completion_events", [])))
        r = repaired[-1] if repaired else {}
        return {"source_cell_id": cid, "instance_id": r.get("instance_id"),
                "target_version": r.get("target_version"),
                "post_repair_drift": r.get("post_repair_drift", r.get("drift")),
                "completion_epoch": r.get("completion_epoch", epoch)}

    def _select_routine(self, obs, route, conventional):
        feedback = obs.get("feedback", [])
        self._tick(len(feedback) + 1)
        if feedback:
            ordered = sorted(feedback, key=lambda r: r["job_id"])
            catalogue = obs["routine_catalogue"]
            identified = []
            for f in ordered:
                self._tick(2)
                if f["routine_id"] not in catalogue or f["correct"] not in (True, False, 0, 1):
                    raise ControllerFailure("invalid_live_feedback")
                other = next(q for q in catalogue if q != f["routine_id"])
                identified.append(f["routine_id"] if f["correct"] else other)
            if len(set(identified)) != 1:
                raise ControllerFailure("conflicting_deterministic_feedback")
            return identified[0]
        # Conventional fallback: newest eligible positive raw report, all arms
        # retain the same source records even if a nominated slot is shuffled.
        if (not conventional and route and route["source_cell_id"] == obs["cell_id"]
                and route.get("transfer_eligible") and route.get("report_value") == 1
                and route.get("candidate_id") in obs["routine_catalogue"]):
            self._tick()
            self._route_consumed.append({"cell_id": obs["cell_id"], "use": "routine_selection"})
            return route["candidate_id"]
        records = list(obs.get("reports", []))
        self._tick(len(records))
        positives = [r for r in records if r.get("transfer_eligible") and
                     r.get("report_value") == 1 and
                     r.get("candidate_id") in obs["routine_catalogue"] and
                     r.get("source_cell_id", obs["cell_id"]) == obs["cell_id"]]
        if positives:
            positives.sort(key=lambda r: (-r.get("report_epoch", 0), r["candidate_id"],
                                          r.get("assay_epoch", 0)))
            self._tick(len(positives) * max(1, len(positives).bit_length()))
            return positives[0]["candidate_id"]
        return obs["current_routine"]

    def _decide(self, obs, route):
        self._tick(4)  # Read public state, context, configuration and decision dispatch.
        conventional = self.comparator == "C4"
        proactive = self.config["source_schedule"] == "proactive"
        reverse = self.config["tie_break"] == "highest"
        domain = conventional or self.config["receiver_policy"] == "domain_aware"
        if self.pilot == "P5":
            d_a = self.constants["production_capacity_per_epoch"]
            d_b = self.constants["replenishment_capacity_per_epoch"]
            stock = obs["reserve"]
            service = min(d_a, stock, obs["demand"])
            # Refill recipes in tunable arms are implementation choices, never
            # labeled the frozen C4 while its contradictory definitions persist.
            target = d_a * (2 if proactive else 1)
            refill = min(d_b, max(0, target - (stock - service)))
            if proactive:
                refill = d_b
            if obs["epoch"] + 1 >= obs["horizon"]:
                refill = 0
            forecast = route.get("delivered_unit_forecast", 0) if route else 0
            if domain and route and route["source_cell_id"] == obs["cell_id"]:
                # Planning writes an explicit next-epoch request, not an action
                # based on unavailable current effects or a longer horizon.
                self._next_plan = min(d_a, stock - service + forecast + refill)
                self._route_consumed.append({"cell_id": obs["cell_id"], "use": "next_epoch_plan",
                                             "planned_request": self._next_plan})
            self._tick(12)
            return {"A": Fraction(service, d_a) if domain else Fraction(1),
                    "B": Fraction(refill, d_b)}
        if self.pilot == "P6":
            routine = self._select_routine(obs, route if domain else None, conventional)
            # The conventional policy does not add assays in ordinary feedback
            # episodes; no-feedback startup retains current q0 exactly.
            assay = int(proactive or (not conventional and not obs.get("feedback")))
            catalogue = sorted(obs["routine_catalogue"], reverse=reverse)
            candidate = next((q for q in catalogue if q != routine), routine)
            self._tick(4)
            return {"A": Fraction(assay), "B": Fraction(1),
                    "candidate_id": candidate, "routine_id": routine,
                    "selector_source": "controller_policy"}
        required = obs["required_version"]
        approved = obs["approved_versions"]
        mode = obs["context"]["mode"]
        repairs, deploys = [], []
        for instance in obs["instances"]:
            self._tick(7)
            iid = instance.get("instance_id", instance["id"])
            current = instance.get("current_version", instance["target_version"])
            drift, locked = instance["drift"], instance["locked"]
            ready = drift == 0
            if (domain and route and route.get("source_cell_id") == obs["cell_id"]
                    and route.get("instance_id") == iid
                    and route.get("target_version") == current
                    and route.get("completion_epoch", obs["epoch"]) <= obs["epoch"]):
                # Typed completion evidence is usable but never overrides the
                # equally available current physical state (a new drift may
                # already have occurred). Raw reconstruction can be identical.
                ready = route.get("post_repair_drift") == 0 and drift == 0
                self._tick(3)
                self._route_consumed.append({"cell_id": obs["cell_id"], "use": "deployment_readiness",
                                             "instance_id": iid})
            if locked:
                continue
            if current == required:
                if drift:
                    repairs.append(iid)
            elif required in approved:
                if mode == "replacement" or ready:
                    deploys.append(iid)
                else:
                    repairs.append(iid)
        repairs.sort(reverse=reverse)
        deploys.sort(reverse=reverse)
        repair = repairs[0] if repairs else None
        deploy = deploys[0] if deploys else None
        if not conventional and not domain and repair is not None:
            # A genuine sequential coordination alternative in the finite grid.
            deploy = None
        # On-demand and proactive share the legal state machine; proactive never
        # repairs clean objects or changes already-correct versions to add cost.
        self._tick(5)
        action = {"A": Fraction(int(repair is not None)),
                  "B": Fraction(int(deploy is not None)), "target_version": required}
        if repair is not None:
            action["repair_target"] = repair
        if deploy is not None:
            action["deploy_target"] = deploy
        return action

    def act(self, observations):
        if len(observations) != self.clock["replicated_cells"]:
            raise ControllerFailure("expected_complete_three_cell_observation")
        if sorted(o["cell_id"] for o in observations) != [0, 1, 2]:
            raise ControllerFailure("cell_identity_mismatch")
        if len({o["epoch"] for o in observations}) != 1:
            raise ControllerFailure("unsynchronized_bundle")
        self._operations = 0
        self._route_consumed = []
        raw = deepcopy(sorted(observations, key=lambda o: o["cell_id"]))
        self._tick(scalar_count(raw))
        messages = [self._route(o) for o in raw]
        n = len(raw)
        if self.comparator == "C2":
            sources = [(recipient - self.cycle) % n for recipient in range(n)]
        elif self.comparator in {"C3", "C5"}:
            sources = list(self.config["generic_sources"])
        else:
            sources = list(range(n))
        slots = [{"source": source, "recipient": recipient,
                  "payload": deepcopy(messages[source]), "slot": recipient}
                 for recipient, source in enumerate(sources)]
        if self.comparator == "C5":
            slots += [{**deepcopy(s), "slot": s["slot"] + n} for s in slots]
        self._tick(sum(scalar_count(s["payload"]) for s in slots))
        actions = []
        for recipient, obs in enumerate(raw):
            self._tick()
            route = messages[sources[recipient]]
            if self.comparator == "C0" or self.lesion:
                # Raw reconstruction receives precisely the same unmasked
                # original source evidence, and is charged separately.
                route = self._route(obs) if self.config["receiver_policy"] == "domain_aware" else None
            action = self._decide(obs, route)
            action["A"] *= self.parameters["activation_gain"]["A"]
            action["B"] *= self.parameters["activation_gain"]["B"]
            self._tick(2)
            actions.append(action)
        retention = 2 if self.comparator == "C5" else 1
        history = (self.memory["raw_history"] + [raw])[-retention:]
        memory = {"last_actions": deepcopy(actions), "raw_history": history}
        count = scalar_count(memory) + scalar_count(self.parameters)
        if count > self.limits.memory_scalars:
            raise ControllerFailure("memory_budget_exhausted")
        self._tick(scalar_count(actions))
        self.memory = memory
        self.last_decision_report = {
            "comparator": self.comparator, "pilot": self.pilot,
            "config_id": self.config["config_id"], "cycle": self.cycle,
            "acute_lesion": self.lesion, "parameters_frozen": True,
            "raw_information_preserved": True, "route_slots": slots,
            "route_use_masked": self.lesion, "useful_operations": self._operations,
            "route_consumed": deepcopy(self._route_consumed),
            "nominated_route_bypassed": self.lesion or self.comparator == "C0",
            "operation_unit": implementation_choices()["resource_unit"],
            "memory_scalars": count, "memory_cap": self.limits.memory_scalars,
            "decision_cap": self.limits.decision_operations,
            "planning_horizon": self.limits.planning_horizon,
            "certified_smallest_common_cap": False,
            "joint_manipulation": False,
        }
        return actions
