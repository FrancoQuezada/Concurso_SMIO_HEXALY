#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smio_clrp.experiments.submission import build_submission_bundle


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--solution-dir", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    output = build_submission_bundle(args.instance_dir, args.solution_dir, args.output)
    print(f"bundle: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
