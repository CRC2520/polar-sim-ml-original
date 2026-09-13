"""Exact block-diagonal coordinate conjugacy, including state and dynamics.

On every numeric A/B coordinate pair T(A,B)=(A-B,A+B). Environmental
quantities that are not a channel pair (stock, versions, object identity,
timestamps, permissions) transform by the identity map. Every nested tree is
traversed; it is not a change of action labels alone. Fractions represent the
exact binary value of input floats and all additions/subtractions/division by2
are rational. Tolerance is zero. Runtime overhead is reported separately.
"""
from copy import deepcopy
from fractions import Fraction
import math
from numbers import Real

from .core import Controller, Limits


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise TypeError("A coordinate must be a finite numeric scalar")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite coordinate")
    return Fraction(value)


def encode_pair(a, b):
    a, b = rational(a), rational(b)
    return {"d": a - b, "q": a + b}


def decode_pair(pair):
    if set(pair) != {"d", "q"}:
        raise ValueError("Expected d and q exactly")
    d, q = rational(pair["d"]), rational(pair["q"])
    return (q + d) / 2, (q - d) / 2


def feasible_pair(pair, capacities, *, admitted=False):
    """Exact transformed continuous domain AND original admitted actuator grid."""
    a, b = decode_pair(pair)
    d, q = rational(pair["d"]), rational(pair["q"])
    if not (abs(d) <= q <= 2 - abs(d)) or not (0 <= a <= 1 and 0 <= b <= 1):
        return False
    if admitted:
        return all((v * cap).denominator == 1 for v, cap in zip((a, b), capacities))
    return True


def _numeric(value):
    return isinstance(value, (int, float, Fraction)) and not isinstance(value, bool)


def encode_tree(value):
    """Tagged bijection; preserve tuple/list distinctions and all nonpair fields."""
    if isinstance(value, dict):
        if "A" in value and "B" in value and _numeric(value["A"]) and _numeric(value["B"]):
            return {"$kind": "paired_mapping", "coordinates": encode_pair(value["A"], value["B"]),
                    "rest": [(deepcopy(k), encode_tree(v)) for k, v in value.items() if k not in ("A", "B")]}
        if (isinstance(value.get("A"), dict) and isinstance(value.get("B"), dict)
                and all(k in value["A"] and k in value["B"] for k in
                        ("requested_activation", "admitted_activation"))):
            transformed = {}
            for field in ("requested_activation", "admitted_activation"):
                if _numeric(value["A"][field]) and _numeric(value["B"][field]):
                    transformed[field] = encode_pair(value["A"][field], value["B"][field])
            return {"$kind": "actuator_records", "coordinates": transformed,
                    "A_identity": encode_tree({k: v for k, v in value["A"].items() if k not in transformed}),
                    "B_identity": encode_tree({k: v for k, v in value["B"].items() if k not in transformed}),
                    "rest": encode_tree({k: v for k, v in value.items() if k not in ("A", "B")})}
        return {"$kind": "mapping", "items": [(deepcopy(k), encode_tree(v)) for k, v in value.items()]}
    if isinstance(value, list):
        return {"$kind": "list", "items": [encode_tree(v) for v in value]}
    if isinstance(value, tuple):
        return {"$kind": "tuple", "items": [encode_tree(v) for v in value]}
    if isinstance(value, Limits):
        return {"$kind": "limits", "items": encode_tree(value.__dict__)}
    if value is None or isinstance(value, (str, bool, int, float, Fraction)):
        return {"$kind": "identity", "value": deepcopy(value)}
    raise TypeError(f"Unsupported state type in exact recoding: {type(value).__name__}")


def decode_tree(value):
    kind = value["$kind"]
    if kind == "paired_mapping":
        a, b = decode_pair(value["coordinates"])
        return {"A": a, "B": b, **{k: decode_tree(v) for k, v in value["rest"]}}
    if kind == "mapping":
        return {k: decode_tree(v) for k, v in value["items"]}
    if kind == "actuator_records":
        a, b = decode_tree(value["A_identity"]), decode_tree(value["B_identity"])
        for field, pair in value["coordinates"].items():
            a[field], b[field] = decode_pair(pair)
        return {"A": a, "B": b, **decode_tree(value["rest"])}
    if kind in {"list", "tuple"}:
        items = [decode_tree(v) for v in value["items"]]
        return tuple(items) if kind == "tuple" else items
    if kind == "limits":
        return Limits(**decode_tree(value["items"]))
    if kind == "identity":
        return deepcopy(value["value"])
    raise ValueError("Unknown coordinate-tree tag")


def coordinate_pairs(value):
    if isinstance(value, dict):
        count = int(value.get("$kind") == "paired_mapping")
        if value.get("$kind") == "actuator_records":
            count += len(value["coordinates"])
        return count + sum(coordinate_pairs(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(coordinate_pairs(v) for v in value)
    return 0


class C6Controller:
    """Store ONLY transformed realization state between decisions.

    The transition is literally T_state F(T_state^-1(state), T_obs^-1(obs)),
    followed by T_action. Decoding is the conjugated arithmetic implementation,
    not an independently selected model. No live untransformed C1 object is
    retained in this wrapper. Snapshot includes parameters, routes, raw-history
    memory, constraints and all decision state, including the last report.
    """
    tolerance_absolute = Fraction(0)
    tolerance_relative = Fraction(0)

    def __init__(self, original):
        if not isinstance(original, Controller) or original.comparator != "C1":
            raise TypeError("C6 must be the image of a specified C1 realization")
        self.state = encode_tree(deepcopy(original.__dict__))
        self.last_decision_report = {}

    def act_encoded(self, encoded_observations):
        original = Controller.__new__(Controller)
        original.__dict__ = decode_tree(self.state)
        try:
            actions = original.act(decode_tree(encoded_observations))
        finally:
            # A failed decision is still a state transition. Preserve its exact
            # image before propagating the original exception to the runner.
            self.state = encode_tree(original.__dict__)
        encoded_actions = encode_tree(actions)
        self.last_decision_report = {
            **deepcopy(original.last_decision_report), "comparator": "C6",
            "original_comparator": "C1", "tolerance_absolute": "0",
            "tolerance_relative": "0", "numeric_format": "exact rational",
            "transformed_state_pairs": coordinate_pairs(self.state),
            "transformed_action_pairs": coordinate_pairs(encoded_actions),
            "conjugated_components": ["state", "observation", "memory", "constraints",
                                      "parameters", "decoder", "actions", "transition"],
            "coordinate_overhead": "recursive encode/decode plus exact rational arithmetic; not charged as useful original-policy operations and not a speed comparison",
        }
        return encoded_actions

    def act(self, observations):
        """Public physical adapter; internal input/state/actions remain recoded."""
        return decode_tree(self.act_encoded(encode_tree(observations)))


class ConjugatedTask:
    """Exact environment transition image, with transformed complete task state.

    This adapter is used for algebraic instrument tests. Evaluator-only hidden
    schedules remain in private transformed state and are never added to the
    public observation. Construction performs no random draw.
    """
    def __init__(self, task):
        self._task_type = type(task)
        self._state = encode_tree(deepcopy(task.__dict__))

    def _restore(self):
        task = self._task_type.__new__(self._task_type)
        task.__dict__ = decode_tree(self._state)
        return task

    def observe_encoded(self):
        task = self._restore()
        observation = task.observe()
        self._state = encode_tree(task.__dict__)
        return encode_tree(observation)

    def step_encoded(self, action):
        task = self._restore()
        result = task.step(decode_tree(action))
        self._state = encode_tree(task.__dict__)
        return encode_tree(result)

    def observe(self):
        return decode_tree(self.observe_encoded())

    def step(self, action):
        return decode_tree(self.step_encoded(encode_tree(action)))
