#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smio_clrp.experiments.registry import update_best_registry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--best-dir", default="results/best")
    parser.add_argument("--replace-ties", action="store_true")
    args = parser.parse_args()
    accepted = update_best_registry(args.instance_dir, args.run_dir, args.best_dir, args.replace_ties)
    print(f"accepted: {len(accepted)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
