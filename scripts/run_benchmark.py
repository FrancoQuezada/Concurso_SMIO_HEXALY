#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smio_clrp.experiments.benchmark import run_benchmark_from_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    rows = run_benchmark_from_config(args.config)
    print(f"runs: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
