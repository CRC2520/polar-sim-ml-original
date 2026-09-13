"""Read-only completeness and content verification for the P5 v1.1 archive."""
from collections import Counter, defaultdict
from hashlib import sha256
import gzip
import json
from pathlib import Path

from b1.evaluation.runner import canonical_bytes
from b1.v1_1.p5.benchmark import ARMS, FLAGS


def verify(directory):
    root = Path(directory)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    expected_files = set(manifest["artifact_sha256"]) | {"MANIFEST.json"}
    actual_files = {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}
    if expected_files != actual_files:
        raise ValueError("P5 archive file set mismatch")
    for path, expected in manifest["artifact_sha256"].items():
        if sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError("P5 archive hash mismatch: " + path)
    result = json.loads((root / "RESULTS.json").read_text())
    rows = [json.loads(line) for line in (root / "episodes.jsonl").read_text().splitlines()]
    expected_episodes = {(i, c, cycle) for i in range(32) for c, cycle in ARMS}
    actual_episodes = Counter((r["bundle_index"], r["comparator"], r["cycle"]) for r in rows)
    if set(actual_episodes) != expected_episodes or any(n != 1 for n in actual_episodes.values()):
        raise ValueError("P5 episode coverage missing or duplicated")
    index = json.loads((root / "traces/INDEX.json").read_text())
    traces, identities = defaultdict(list), set()
    for shard in index["shards"]:
        data = (root / "traces" / shard["path"]).read_bytes()
        if sha256(data).hexdigest() != shard["sha256"] or len(data) != shard["bytes"]:
            raise ValueError("P5 shard mismatch")
        records = [json.loads(line) for line in gzip.decompress(data).splitlines()]
        if len(records) != shard["records"]:
            raise ValueError("P5 shard count mismatch")
        for record in records:
            ep = (record["bundle_index"], record["comparator"], record["cycle"])
            identity = ep + (record["epoch"], record["cell_id"])
            if identity in identities:
                raise ValueError("P5 duplicated event identity")
            if any(record.get(k) != v for k, v in FLAGS.items()):
                raise ValueError("P5 result scope mismatch")
            if (record["namespace"] != "PD-B1-D-v1.1" or record["source_commit"] != manifest["source_commit"]
                    or record["context"] != "ordinary_utility" or record["route_status"] != "intact"
                    or record["action"] != record["event"]["proposal"]):
                raise ValueError("P5 event provenance/schema mismatch")
            identities.add(identity)
            traces[ep].append(record["event"])
    if len(identities) != 32 * len(ARMS) * 96 or len(identities) != index["records"]:
        raise ValueError("P5 trace coverage incomplete")
    for row in rows:
        ep = (row["bundle_index"], row["comparator"], row["cycle"])
        events = traces[ep]
        if {(e["epoch"], e["cell_id"]) for e in events} != {(t, c) for t in range(32) for c in range(3)}:
            raise ValueError("P5 per-episode coverage incomplete")
        digest = sha256(b"".join(canonical_bytes(e) for e in events)).hexdigest()
        if digest != row["event_trace_sha256"]:
            raise ValueError("P5 event replay identity mismatch")
        if sum(e["missed"] for e in events) != row["missed_jobs"]:
            raise ValueError("P5 scalar/event outcome mismatch")
        if row["source_commit"] != manifest["source_commit"]:
            raise ValueError("P5 episode source mismatch")
    if result["episode_count"] != len(rows) or result["event_count"] != len(identities):
        raise ValueError("P5 aggregate count mismatch")
    return {"status": "PASS", "episode_count": len(rows), "event_count": len(identities),
            "source_commit": manifest["source_commit"], "hashes_verified": len(manifest["artifact_sha256"]), **FLAGS}


if __name__ == "__main__":
    import sys
    print(json.dumps(verify(sys.argv[1]), sort_keys=True))
