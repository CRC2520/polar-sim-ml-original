"""Implement only the frozen B1-D namespace; no final entropy/seed API exists.

Calibration access requires a content-verified tuning freeze. Merely selecting a
role or bundle index cannot open it. This module does not open calibration on
import, initialization, policy export, or freeze verification.
"""
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re

from b1.contracts import load_contracts
from b1.contracts.loader import strict_json


class SeedPolicyError(ValueError):
    pass


_VERIFIED = object()


@dataclass(frozen=True)
class _TuningFreeze:
    manifest_sha256: str
    source_commit: str
    manifest_path: str
    _verification: object


def verify_tuning_freeze(manifest_path, *, contracts=None):
    """Verify closure evidence without deriving or inspecting any holdout seed.

    Manifest fields: status=TUNING_CLOSED; source_commit (canonical B0 source);
    tuning_bundle_indices=0..31; configuration_frozen=true;
    calibration_comparator_identities_masked=true; calibration_consumed=false;
    artifact_sha256 mapping relative paths to frozen code/config/tuning ledger.
    The mapping must cover at least one source .py, one config .json and one
    tuning ledger .json, named by implementation_files/configuration_file/
    tuning_ledger_file. All paths resolve within the manifest's parent tree.
    """
    contracts = contracts or load_contracts()
    manifest_path = Path(manifest_path)
    try:
        raw = manifest_path.read_bytes()
        manifest = strict_json(raw.decode("utf-8"))
        tuning_range = contracts.rules["calibration"]["development_bundle_indices"]["tuning"]
        if (manifest.get("status") != "TUNING_CLOSED" or
                manifest.get("source_commit") != contracts.source_commit or
                manifest.get("tuning_bundle_indices") != list(range(tuning_range[0], tuning_range[1] + 1)) or
                manifest.get("configuration_frozen") is not True or
                manifest.get("calibration_comparator_identities_masked") is not True or
                manifest.get("calibration_consumed") is not False):
            raise SeedPolicyError("Tuning closure or unopened blinded calibration is not established")
        hashes = manifest.get("artifact_sha256", {})
        implementation = manifest.get("implementation_files", [])
        config, ledger = manifest.get("configuration_file"), manifest.get("tuning_ledger_file")
        required = implementation + [config, ledger]
        if not implementation or not all(isinstance(x, str) and x in hashes for x in required):
            raise SeedPolicyError("Frozen implementation, configuration and tuning ledger are required")
        if not all(x.endswith(".py") for x in implementation) or not config.endswith(".json") or not ledger.endswith(".json"):
            raise SeedPolicyError("Invalid freeze artifact types")
        root = manifest_path.parent.resolve()
        for name, expected in hashes.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or Path(name).is_absolute() or not re.fullmatch(r"[0-9a-f]{64}", expected):
                raise SeedPolicyError("Invalid artifact path or checksum")
            if sha256(path.read_bytes()).hexdigest() != expected:
                raise SeedPolicyError(f"Tuning artifact changed: {name}")
        ledger_data = strict_json((root / ledger).read_text())
        if ledger_data.get("bundle_indices") != manifest["tuning_bundle_indices"] or ledger_data.get("development_only") is not True:
            raise SeedPolicyError("Tuning ledger does not demonstrate the declared development block")
        return _TuningFreeze(sha256(raw).hexdigest(), contracts.source_commit,
                            str(manifest_path.resolve()), _VERIFIED)
    except (OSError, KeyError, TypeError, UnicodeError) as exc:
        raise SeedPolicyError(f"Cannot verify tuning freeze: {exc}") from exc


class DevelopmentSeeds:
    """Deterministic counter streams with a complete per-use derivation ledger.

    `derive(pilot_id, role, bundle_index, substream_id)` accepts only development
    roles. Substream IDs should explicitly encode cell/episode/epoch/event.
    `open_calibration(token)` does not itself derive a calibration seed.
    Historical final seeds supplied by the caller are collision exclusions; a
    collision blocks instead of changing the canonical B1-D derivation rule.
    """
    def __init__(self, source_commit=None, usage_log=None, *, contracts=None, historical_seeds=()):
        self.contracts = contracts or load_contracts()
        self.rules = self.contracts.rules
        self.source_commit = source_commit or self.contracts.source_commit
        if self.source_commit != self.contracts.source_commit:
            raise SeedPolicyError("B1-D source_commit must be the canonical B0 source")
        calibration = self.rules["calibration"]
        self.namespace = calibration["development_seed_namespace"]
        self.ranges = calibration["development_bundle_indices"]
        self.pilots = tuple(self.rules["scope"]["pilots"])
        # Roles are explicitly enumerated in the authoritative prose protocol.
        protocol = self.contracts.text("B1_PROTOCOL_DRAFT_v1.md")
        match = re.search(r"Roles are `([^`]+)`, `([^`]+)`, `([^`]+)`, and `([^`]+)`", protocol)
        if match is None:
            raise SeedPolicyError("Canonical development role list cannot be located")
        self.roles = tuple(match.groups())
        self._calibration_token = None
        self.usage_log = usage_log if usage_log is not None else []
        self.historical_seeds = frozenset(historical_seeds)
        if any(type(seed) is not int or not 0 <= seed < 2**64 for seed in self.historical_seeds):
            raise SeedPolicyError("Historical seed exclusions must be unsigned 64-bit integers")
        self._derived_inputs = {}

    def open_calibration(self, token):
        if (not isinstance(token, _TuningFreeze) or token._verification is not _VERIFIED or
                token.source_commit != self.source_commit):
            raise SeedPolicyError("Calibration requires a verified tuning-freeze token")
        current = verify_tuning_freeze(token.manifest_path, contracts=self.contracts)
        if current.manifest_sha256 != token.manifest_sha256:
            raise SeedPolicyError("The tuning freeze changed after token verification")
        self._calibration_token = token

    def authorize(self, pilot_id, role, bundle_index, substream_id):
        if pilot_id not in self.pilots or role not in self.roles:
            raise SeedPolicyError("Unknown pilot or non-development role")
        if type(bundle_index) is not int:
            raise SeedPolicyError("Bundle index must be an integer")
        tuning = self.ranges["tuning"]
        calibration = self.ranges["blinded_calibration"]
        if role == "calibration":
            if not calibration[0] <= bundle_index <= calibration[1] or self._calibration_token is None:
                raise SeedPolicyError("Blinded calibration is unopened or index is outside its block")
        elif not tuning[0] <= bundle_index <= tuning[1]:
            raise SeedPolicyError("Training/tuning/development evaluation requires the tuning block")
        if not isinstance(substream_id, str) or not substream_id or "|" in substream_id or any(ord(c) < 32 for c in substream_id):
            raise SeedPolicyError("Substream must be a nonempty, unambiguous printable identifier")

    def derive(self, pilot_id, role, bundle_index, substream_id):
        self.authorize(pilot_id, role, bundle_index, substream_id)
        fields = [self.namespace, self.source_commit, pilot_id, role, str(bundle_index), substream_id]
        text = "|".join(fields)
        digest = sha256(text.encode("utf-8")).digest()
        seed = int.from_bytes(digest[:8], "big", signed=False)
        if seed in self.historical_seeds:
            raise SeedPolicyError("Development seed collides with a historical final; no replacement was generated")
        if seed in self._derived_inputs and self._derived_inputs[seed] != text:
            raise SeedPolicyError("Distinct development streams collide; no replacement was generated")
        self._derived_inputs[seed] = text
        self.usage_log.append({"use_index": len(self.usage_log), "namespace": self.namespace,
            "source_commit": self.source_commit, "pilot_id": pilot_id, "role": role,
            "bundle_index": bundle_index, "substream_id": substream_id,
            "derivation_sha256": digest.hex(), "seed": seed, "development_only": True,
            "confirmatory": False, "reusable_as_final": False,
            "tuning_freeze_sha256": self._calibration_token.manifest_sha256
                 if role == "calibration" else None})
        return seed

    def policy(self):
        return {"namespace": self.namespace, "source_commit": self.source_commit,
            "roles": list(self.roles), "bundle_indices": self.ranges,
            "hash_fields": self.rules["calibration"]["development_seed_hash_fields"],
            "seed_function": self.rules["calibration"]["seed_function"],
            "calibration_open": self._calibration_token is not None,
            "usage_count": len(self.usage_log), "final_seeds_generated": False,
            "final_seed_generation_supported": False, "B1E_execution_supported": False}
