"""Create or verify the R10 scientific freeze manifest."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
FILES=(
    "r10_completion/PREREG_R10.md",
    "r10_completion/config.py",
    "r10_completion/core.py",
    "r10_completion/exp1_causal_state.py",
    "r10_completion/exp2_isomorphic.py",
    "r10_completion/exp3_gate.py",
    "r10_completion/exp4_transfer.py",
    "r10_completion/exp5_integrated.py",
    "r10_completion/runner.py",
    "r10_completion/test_r10.py",
)

def sha(path):
    return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()

def build():
    record={
        "schema":"POLAR-R10-FREEZE-v1",
        "base_r9_commit":"a0696148e568c94365a26311953430b082a3725b",
        "scientific_files":{p:sha(p) for p in FILES},
        "pilot_seeds":[970101,970102,970103,970104],
        "final_seeds":list(range(972001,972017)),
        "criteria_locked":True,
        "note":"Local Git-backed freeze before confirmatory R10 execution; not external preregistration."
    }
    canonical=json.dumps(record,sort_keys=True,separators=(",",":")).encode()
    record["freeze_sha256"]=hashlib.sha256(canonical).hexdigest()
    return record

def verify(record):
    current=build()
    if record != current:
        raise ValueError("R10 scientific source differs from frozen manifest")
    return True

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--output",type=Path)
    ap.add_argument("--verify",type=Path)
    args=ap.parse_args()
    if args.verify:
        verify(json.loads(args.verify.read_text()))
        print("R10 FREEZE VERIFY PASS")
        return
    rec=build()
    text=json.dumps(rec,indent=2,sort_keys=True)+"\n"
    if args.output:
        args.output.write_text(text)
    print(text,end="")
if __name__=="__main__": main()
