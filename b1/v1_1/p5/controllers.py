"""Canonical P5 inventory control under the prospective v1.1 disposition.

No graph, memory history or dummy calculation is added to match actual usage.
The same complete observation and maximum resources remain available.
"""
from copy import deepcopy
from fractions import Fraction

from b1.controllers.core import ControllerFailure, Limits, scalar_count


DEFAULT_LIMITS = Limits(memory_scalars=16384, decision_operations=65536,
                        planning_horizon=1, latency_seconds=None)


class P5ConventionalController:
    def __init__(self, *, limits=None, full_refill_witness=False):
        self.pilot = "P5"
        self.comparator = "C4_FULL_REFILL_CEILING_WITNESS" if full_refill_witness else "C4"
        self.full_refill_witness = bool(full_refill_witness)
        self.limits = limits or DEFAULT_LIMITS
        if self.limits.planning_horizon != 1:
            raise ValueError("Only the frozen one-future-epoch horizon is permitted")
        self.parameters = {"production_capacity": 2, "replenishment_capacity": 2,
                           "policy": self.comparator, "terminal_refill": self.full_refill_witness}
        self.config = {"config_id": "fixed-full-witness" if full_refill_witness else "fixed-minimal-v1.1"}
        self.memory = {"last_actions": []}
        self.last_decision_report = {}
        self._operations = 0

    def _tick(self, count):
        self._operations += count
        if self._operations > self.limits.decision_operations:
            raise ControllerFailure("decision_budget_exhausted")

    def act(self, observations):
        if len(observations) != 3 or sorted(o["cell_id"] for o in observations) != [0, 1, 2]:
            raise ControllerFailure("expected_complete_three_cell_observation")
        if len({o["epoch"] for o in observations}) != 1:
            raise ControllerFailure("unsynchronized_bundle")
        self._operations = 0
        # Identity/epoch reads and comparisons used to validate three cells.
        self._tick(12)
        actions = []
        for obs in sorted(observations, key=lambda o: o["cell_id"]):
            if self.full_refill_witness:
                action = {"A": Fraction(1), "B": Fraction(1)}
                self._tick(2)
            else:
                service = min(2, obs["reserve"], obs["demand"])
                refill = min(2, max(0, 2 - (obs["reserve"] - service)))
                if obs["epoch"] + 1 >= obs["horizon"]:
                    refill = 0
                action = {"A": Fraction(service, 2), "B": Fraction(refill, 2)}
                # Public reads, min/max, subtraction, branch, rational outputs.
                self._tick(17)
            actions.append(action)
        self.memory = {"last_actions": deepcopy(actions)}
        self._tick(scalar_count(actions))
        count = scalar_count(self.memory) + scalar_count(self.parameters)
        if count > self.limits.memory_scalars:
            raise ControllerFailure("memory_budget_exhausted")
        self.last_decision_report = {
            "pilot": "P5", "comparator": self.comparator,
            "config_id": self.config["config_id"], "cycle": 1,
            "acute_lesion": False, "parameters_frozen": True,
            "raw_information_preserved": True, "route_slots": [],
            "route_consumed": [], "nominated_route_bypassed": True,
            "useful_operations": self._operations,
            "operation_unit": "logical read/comparison/arithmetic/output/copy proxy; not CPU instructions",
            "operation_count_scope": "canonical policy and validated input; expanded accounting is a separate observational profiler",
            "memory_scalars": count, "memory_cap": self.limits.memory_scalars,
            "decision_cap": self.limits.decision_operations,
            "planning_horizon": 1, "resource_cap_binding": False,
            "joint_manipulation": False,
        }
        return actions
