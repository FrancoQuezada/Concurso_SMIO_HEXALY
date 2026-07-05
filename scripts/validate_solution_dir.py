#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-dir", required=True)
    parser.add_argument("--solution-dir", required=True)
    args = parser.parse_args()
    instances = {read_instance(path).name: read_instance(path) for path in Path(args.instance_dir).glob("*.txt")}
    checked = 0
    failed = 0
    for solution_path in sorted(Path(args.solution_dir).rglob("*.sol")):
        instance_name = re.sub(r"_seed\d+$", "", solution_path.stem)
        instance = instances.get(instance_name)
        if instance is None:
            print(f"missing instance: {solution_path}")
            failed += 1
            continue
        result = validate_solution(instance, read_solution(solution_path))
        checked += 1
        if not result.is_feasible:
            print(f"infeasible: {solution_path}: {'; '.join(result.errors)}")
            failed += 1
    print(f"checked: {checked}")
    print(f"failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
