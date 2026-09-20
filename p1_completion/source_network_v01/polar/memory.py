"""Auditable cue-addressed internal memory, independent of repeated input."""
import numpy as np
from .polarities import swap_poles


class EpisodicMemory:
    def __init__(self, shape, learning_rate=0.5, retention=0.995):
        if not 0 < learning_rate <= 1 or not 0 <= retention <= 1:
            raise ValueError("invalid memory learning rate or retention")
        self.shape, self.learning_rate, self.retention = tuple(shape), learning_rate, retention
        self.records, self.events = {}, []
        self.clock = 0

    def tick(self):
        self.clock += 1

    def _array(self, value, name):
        result = np.asarray(value, dtype=float)
        if result.shape != self.shape or not np.isfinite(result).all():
            raise ValueError(f"{name} must be finite with shape {self.shape}")
        return result

    def learn(self, cue, target, mask=None):
        if cue is None:
            return {"operation": "no_cue", "written_channels": 0}
        target = self._array(target, "target")
        if np.any((target < 0) | (target > 1)):
            raise ValueError("memory targets must lie in [0,1]")
        mask = np.ones(self.shape, bool) if mask is None else np.asarray(mask, bool)
        if mask.shape != self.shape:
            raise ValueError("memory mask has wrong shape")
        cue = str(cue)
        if cue not in self.records:
            self.records[cue] = {"value": np.zeros(self.shape), "counts": np.zeros(self.shape, int),
                                 "updated": np.full(self.shape, self.clock, int)}
        record = self.records[cue]
        first = mask & (record["counts"] == 0)
        repeated = mask & ~first
        record["value"][first] = target[first]
        record["value"][repeated] += self.learning_rate * (target[repeated] - record["value"][repeated])
        record["counts"][mask] += 1
        record["updated"][mask] = self.clock
        event = {"operation": "learn", "cue": cue, "step": self.clock, "written_channels": int(mask.sum())}
        self.events.append(event)
        return event.copy()

    def recall(self, cue):
        record = self.records.get(str(cue)) if cue is not None else None
        if record is None:
            return np.zeros(self.shape), np.zeros(self.shape), {"cue": cue, "found": False, "counts": np.zeros(self.shape, int).tolist()}
        counts = record["counts"]
        confidence = counts / (counts + 1) * self.retention ** (self.clock - record["updated"])
        return record["value"].copy(), confidence, {"cue": str(cue), "found": True, "counts": counts.tolist(),
            "age": (self.clock - record["updated"]).tolist()}

    def erase(self, cue=None):
        if cue is None:
            self.records.clear()
        else:
            self.records.pop(str(cue), None)
        self.events.append({"operation": "erase", "cue": cue, "step": self.clock})

    def edit(self, cue, target, mask=None):
        target = self._array(target, "edited target")
        if np.any((target < 0) | (target > 1)):
            raise ValueError("edited target outside [0,1]")
        mask = np.ones(self.shape, bool) if mask is None else np.asarray(mask, bool)
        if mask.shape != self.shape:
            raise ValueError("memory edit mask has wrong shape")
        if str(cue) not in self.records:
            self.learn(cue, target, mask)
        else:
            rec = self.records[str(cue)]
            rec["value"][mask] = target[mask]
            rec["counts"][mask] = np.maximum(rec["counts"][mask], 1)
            rec["updated"][mask] = self.clock
        self.events.append({"operation": "edit", "cue": str(cue), "step": self.clock, "channels": int(mask.sum())})

    def shuffle(self, seed=0):
        """Permute cue associations without changing record values or memory capacity."""
        keys = sorted(self.records)
        values = [self.records[key] for key in keys]
        order = np.random.default_rng(seed).permutation(len(keys))
        self.records = {key: values[i] for key, i in zip(keys, order)}
        self.events.append({"operation": "shuffle", "seed": int(seed), "step": self.clock, "order": order.tolist()})

    def reverse_convention(self, types=None):
        for record in self.records.values():
            for key in record:
                record[key] = swap_poles(record[key], types)
        self.events.append({"operation": "reverse_convention", "step": self.clock})

    def snapshot(self):
        return {"clock": self.clock, "learning_rate": self.learning_rate, "retention": self.retention,
                "records": {key: {field: value.tolist() for field, value in record.items()}
                            for key, record in self.records.items()}}
