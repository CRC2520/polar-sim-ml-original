"""Operational task channels; labels are not validated psychological constructs."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Polarity:
    name: str
    positive: str
    negative: str
    positive_operation: str
    negative_operation: str
    cooperation: str
    conflict: str


POLARITIES = (
    Polarity("Poder vs Vulnerabilidad", "Poder", "Vulnerabilidad",
             "Allocate effort to an independently executed task", "Allocate effort to requesting external assistance",
             "Execute one task while requesting help on another", "Both compete for the same communication/time budget"),
    Polarity("Placer vs Dolor", "Placer", "Dolor",
             "Allocate effort to acquiring a supplied reward", "Allocate effort to detecting a supplied loss signal",
             "Reward acquisition and loss monitoring can run concurrently", "Monitoring and acquisition consume a shared budget"),
    Polarity("Integración vs Fragmentación", "Integración", "Fragmentación",
             "Allocate effort to shared task aggregation", "Allocate effort to independent task decomposition",
             "Decompose subtasks then aggregate their outputs", "Aggregation and decomposition consume shared computation"),
    Polarity("Control vs Rendición", "Control", "Rendición",
             "Allocate effort to direct actuation", "Allocate effort to delegated actuation",
             "Direct one channel and delegate another", "A single actuator cannot accept contradictory commands"),
    Polarity("Deseo vs Límite", "Deseo", "Límite",
             "Allocate effort to requested throughput", "Allocate effort to maintaining a supplied reserve",
             "Throughput and reserves meet different task demands", "Both draw on a finite supplied resource"),
    Polarity("Libertad vs Orden", "Libertad", "Orden",
             "Allocate effort to exploring supplied alternatives", "Allocate effort to executing a supplied routine",
             "Explore some channels while executing routines in others", "Exploration and routine execution share time"),
    Polarity("Preservación vs Transformación", "Preservación", "Transformación",
             "Allocate effort to maintaining an existing task configuration", "Allocate effort to changing a task configuration",
             "Maintain one subsystem while changing another", "A configuration cannot simultaneously retain and change one value"),
    Polarity("Reconocimiento vs Autenticidad", "Reconocimiento", "Autenticidad",
             "Allocate effort to meeting an externally supplied target", "Allocate effort to retaining a predeclared local target",
             "External and local targets can agree", "They conflict only where the supplied targets or resources conflict"),
)


def to_signed_intensity(q):
    """Bijective coordinates for two nonnegative channels, not net activation alone."""
    q = np.asarray(q, dtype=float)
    if q.shape[-1] != 2 or not np.isfinite(q).all() or np.any((q < 0) | (q > 1)):
        raise ValueError("q must have final dimension 2 and finite values in [0, 1]")
    return q[..., 0] - q[..., 1], q.sum(axis=-1)


def from_signed_intensity(signed, intensity):
    signed, intensity = np.broadcast_arrays(np.asarray(signed, float), np.asarray(intensity, float))
    q = np.stack(((intensity + signed) / 2, (intensity - signed) / 2), axis=-1)
    if not np.isfinite(q).all() or np.any((q < -1e-12) | (q > 1 + 1e-12)):
        raise ValueError("infeasible signed/intensity coordinates")
    return np.clip(q, 0, 1)


def swap_poles(array, types=None):
    result = np.asarray(array).copy()
    if result.ndim < 2 or result.shape[-1] != 2:
        raise ValueError("expected [..., type, pole] array")
    if types is None:
        return result[..., ::-1].copy()
    result[..., types, :] = result[..., types, ::-1]
    return result
