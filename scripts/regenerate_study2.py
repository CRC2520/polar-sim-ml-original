#!/usr/bin/env python3
"""Rebuild Study 2 metrics, figures and decisions from complete saved traces."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from study2.evaluation import regenerate

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = regenerate(args.manifest, args.out or args.manifest.parent/"generated")
    print(f"Verified and regenerated {result['trace_count']} trials: {result['decision']}")
