#!/usr/bin/env python3
"""Scan all .sol files matching a glob and report the lowest-cost feasible one.

Usage:
    python scripts/find_best.py data/official/clrp-small-04.txt "results/competition/small04_*.sol"
"""
from __future__ import annotations

import argparse
import sys
from glob import glob
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("instance_path")
parser.add_argument("glob_pattern")
args = parser.parse_args()

instance = read_instance(args.instance_path)
best_path = None
best_cost = float("inf")
checked = 0
for path in sorted(glob(args.glob_pattern)):
    checked += 1
    try:
        solution = read_solution(path)
        if not validate_solution(instance, solution).is_feasible:
            continue
        cost = objective_cost(instance, solution)
    except Exception:
        continue
    if cost < best_cost:
        best_cost = cost
        best_path = path

print(f"checked {checked} files")
if best_path is None:
    print("no feasible solution found")
else:
    print(f"BEST: {best_cost:.2f}  ({best_path})")
