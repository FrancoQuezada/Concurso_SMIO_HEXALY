#!/usr/bin/env python3
"""Quick interactive test of Fix-and-Optimize on a given instance.

By default builds a fast greedy initial solution, then runs F&O (heuristic
and/or mip backend) on top of it and reports before/after cost, feasibility,
and timing. Pass --initial-solution to start from an existing .sol file
instead (e.g. a teammate's ALNS output) -- this is the real competition
workflow: ALNS produces a first-solution, F&O refines it into the final one.

Usage:
    python scripts/try_fixopt.py data/samples/fixopt_demo.txt
    python scripts/try_fixopt.py data/samples/fixopt_demo.txt --backend mip
    python scripts/try_fixopt.py data/samples/fixopt_demo.txt --iterations 100 --time-limit 60
    python scripts/try_fixopt.py data/samples/fixopt_demo.txt --initial-solution path/to/alns_result.sol
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smio_clrp.algorithms.base import SolverConfig
from smio_clrp.algorithms.constructive.greedy import GreedyNearestDepotSolver
from smio_clrp.algorithms.fixopt.fixopt_solver import FixOptimizeSolver
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_path")
    parser.add_argument(
        "--initial-solution",
        default=None,
        help="path to an existing .sol file to start from (e.g. a teammate's ALNS result) "
        "instead of building a fresh greedy solution",
    )
    parser.add_argument("--backend", choices=["heuristic", "mip", "both"], default="both")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--time-limit", type=float, default=30.0, help="seconds per backend")
    parser.add_argument("--mip-time-limit", type=float, default=3.0, help="seconds per MIP subproblem")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="results/try_fixopt")
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    print(f"instance: {instance.name} ({len(instance.customers)} customers, {len(instance.depots)} depots)")

    if args.initial_solution:
        t0 = time.perf_counter()
        initial_solution = read_solution(args.initial_solution)
        initial_validation = validate_solution(instance, initial_solution)
        if not initial_validation.is_feasible:
            print(f"ERROR: {args.initial_solution} is not feasible for this instance:")
            for error in initial_validation.errors:
                print(f"  - {error}")
            return 1
        initial_cost = objective_cost(instance, initial_solution)
        print(f"initial ({args.initial_solution}): cost={initial_cost:.1f} routes={len(initial_solution.routes)} "
              f"time={time.perf_counter() - t0:.2f}s feasible=True")
    else:
        t0 = time.perf_counter()
        greedy_result = GreedyNearestDepotSolver(SolverConfig(seed=args.seed)).solve(instance)
        initial_solution = greedy_result.solution
        initial_cost = objective_cost(instance, initial_solution)
        print(f"initial (greedy): cost={initial_cost:.1f} routes={len(initial_solution.routes)} "
              f"time={time.perf_counter() - t0:.2f}s feasible={validate_solution(instance, initial_solution).is_feasible}")
    print()

    backends = ["heuristic", "mip"] if args.backend == "both" else [args.backend]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for backend in backends:
        solver = FixOptimizeSolver(
            initial_solution,
            SolverConfig(
                seed=args.seed,
                time_limit_seconds=args.time_limit,
                metadata={
                    "fixopt_iterations": args.iterations,
                    "fixopt_backend": backend,
                    "mip_time_limit": args.mip_time_limit,
                },
            ),
        )
        t0 = time.perf_counter()
        result = solver.solve(instance)
        elapsed = time.perf_counter() - t0

        print(f"--- backend={backend} ---")
        if result.solution is None:
            print(f"  FAILED: {result.metadata.get('error')}")
            continue
        feasible = validate_solution(instance, result.solution).is_feasible
        delta = result.cost - initial_cost
        pct = 100 * delta / initial_cost
        print(f"  cost={result.cost:.1f} (delta={delta:+.1f}, {pct:+.1f}%) feasible={feasible} time={elapsed:.2f}s")
        print(f"  iterations={result.metadata.get('iterations')} "
              f"improved={result.metadata.get('improved_subproblems')} "
              f"failed={result.metadata.get('failed_subproblems')} "
              f"skipped={result.metadata.get('skipped_subproblems')}")
        sol_path = output_dir / f"{instance.name}_{backend}.sol"
        write_solution(result.solution, sol_path, instance=instance)
        print(f"  solution written: {sol_path}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
