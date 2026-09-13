"""Load canonical contract bytes without rewriting or resolving contradictions.

Hashes establish provenance, not the scientific consistency of the sources.
Affected task implementations must still stop on a documented contradiction.
"""
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import yaml


B0_FREEZE_SHA256 = "7d9abe376401ca53c8e8f9f519b75c6bb227dfc49749059bb72eb24050978cc1"
FROZEN_DIRECTORY = Path(__file__).with_name("frozen_b0")
REQUIRED_FILES = (
    "PILOT_CONTRACTS_v1.json", "CONSTRUCT_DICTIONARY_v1.yaml",
    "B1_PROTOCOL_DRAFT_v1.md", "B1_DECISION_RULES_v1.yaml",
    "PD_OPERATIONAL_FOUNDATIONS_v3.0.md",
)


class ContractError(ValueError):
    """A canonical input is missing, altered, or structurally inconsistent."""


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate mapping key: {key!r}")
        result[key] = value
    return result


class _UniqueSafeLoader(yaml.SafeLoader):
    pass


def _yaml_mapping(loader, node, deep=False):
    return _unique_pairs((loader.construct_object(key, deep=deep),
                          loader.construct_object(value, deep=deep))
                         for key, value in node.value)


_UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                                 _yaml_mapping)


def strict_json(text):
    def reject_constant(value):
        raise ContractError(f"Nonfinite JSON constant: {value}")
    return json.loads(text, object_pairs_hook=_unique_pairs,
                      parse_constant=reject_constant)


def strict_yaml(text):
    return yaml.load(text, Loader=_UniqueSafeLoader)


@dataclass(frozen=True)
class FrozenContracts:
    """Properties return copies so callers cannot mutate another caller's rules."""
    _freeze: dict
    _pilots: dict
    _rules: dict
    _dictionary: dict
    _texts: dict
    _hashes: dict

    @property
    def freeze(self):
        return deepcopy(self._freeze)

    @property
    def pilots(self):
        return deepcopy(self._pilots)

    @property
    def rules(self):
        return deepcopy(self._rules)

    @property
    def dictionary(self):
        return deepcopy(self._dictionary)

    @property
    def hashes(self):
        return dict(self._hashes)

    @property
    def source_commit(self):
        return self._freeze["source_commit"]

    def text(self, filename):
        return self._texts[filename]

    def pilot(self, pilot_id):
        for pilot in self._pilots["pilots"]:
            if pilot["id"] == pilot_id:
                return deepcopy(pilot)
        raise ContractError(f"Unknown pilot: {pilot_id}")

    def field(self, pilot_id, identifier):
        for field in self.pilot(pilot_id)["fields"]:
            if field["number"] == identifier or field["name"] == identifier:
                return deepcopy(field["value"])
        raise ContractError(f"Unknown field {pilot_id}.{identifier}")


def load_contracts(directory=None):
    directory = Path(directory) if directory is not None else FROZEN_DIRECTORY
    try:
        raw_freeze = (directory / "B0_FREEZE.json").read_bytes()
        if sha256(raw_freeze).hexdigest() != B0_FREEZE_SHA256:
            raise ContractError("B0_FREEZE.json does not match the verified publication")
        freeze = strict_json(raw_freeze.decode("utf-8"))
        if (freeze["status"] != "B0_COMPLETE" or
                freeze["ready_for_b1_design"] is not True or
                freeze["ready_for_b1_confirmatory_run"] is not False):
            raise ContractError("Unexpected B0 handoff status")
        texts, hashes = {}, {"B0_FREEZE.json": B0_FREEZE_SHA256}
        for filename in REQUIRED_FILES:
            raw = (directory / filename).read_bytes()
            digest = sha256(raw).hexdigest()
            if digest != freeze["artifact_sha256"]["b0/" + filename]:
                raise ContractError(f"Frozen contract hash mismatch: {filename}")
            texts[filename], hashes[filename] = raw.decode("utf-8"), digest
        pilots = strict_json(texts["PILOT_CONTRACTS_v1.json"])
        rules = strict_yaml(texts["B1_DECISION_RULES_v1.yaml"])
        dictionary = strict_yaml(texts["CONSTRUCT_DICTIONARY_v1.yaml"])
        if not (pilots["source_commit"] == rules["source_commit"] == freeze["source_commit"]):
            raise ContractError("Canonical source commits differ")
        if [p["id"] for p in pilots["pilots"]] != rules["scope"]["pilots"]:
            raise ContractError("Canonical pilot registries differ")
        return FrozenContracts(freeze, pilots, rules, dictionary, texts, hashes)
    except (OSError, KeyError, UnicodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise ContractError(f"Cannot load canonical contracts: {exc}") from exc
