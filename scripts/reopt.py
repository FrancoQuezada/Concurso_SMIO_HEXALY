#!/usr/bin/env python3
"""Reoptimize an existing .sol file with fix-and-optimize using larger subproblems
than the default 12-customer window, to try to escape whatever local optimum the
seed-sweep pipeline converged to.

Usage:
    python scripts/reopt.py data/official/clrp-small-04.txt results/competition/best.sol \
        --output results/competition/best_reopt.sol --backend mip \
        --max-customers-per-subproblem 40 --max-routes-per-subproblem 10
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smio_clrp.algorithms.base import SolverConfig
from smio_clrp.algorithms.fixopt.fixopt_solver import FixOptimizeSolver
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("instance_path")
parser.add_argument("initial_solution")
parser.add_argument("--output", required=True)
parser.add_argument("--backend", choices=["heuristic", "mip"], default="mip")
parser.add_argument("--max-customers-per-subproblem", type=int, default=25)
parser.add_argument("--max-routes-per-subproblem", type=int, default=6)
parser.add_argument(
    "--neighborhood-types",
    default=None,
    help="Comma-separated neighborhood types to force (e.g. 'route' with high "
    "--max-routes/customers-per-subproblem re-optimizes the WHOLE instance as one set-"
    "partitioning MIP each iteration, instead of rotating through small local windows).",
)
parser.add_argument(
    "--max-candidate-routes",
    type=int,
    default=2000,
    help="Cap on generated multi-customer candidate routes per MIP subproblem (on top of "
    "always-included singletons). The 2000 default was sized for the 12-25 customer "
    "default window; a small, fully-released whole-instance subproblem (few dozen "
    "customers, few depots) can afford a much larger budget for a genuinely exhaustive "
    "re-optimization instead of a thin, easily-infeasible candidate set.",
)
parser.add_argument(
    "--max-subset-size-slack",
    type=int,
    default=2,
    help="Slack added on top of capacity/avg-demand 'typical route size' when capping "
    "candidate subset sizes. Too small a value can make the model unable to represent the "
    "incumbent's own longer routes as a candidate, showing up as spurious infeasibilities.",
)
parser.add_argument("--iterations", type=int, default=2000)
parser.add_argument("--time-limit", type=float, default=180.0)
parser.add_argument("--mip-time-limit", type=float, default=15.0)
parser.add_argument("--seed", type=int, default=1)
args = parser.parse_args()

instance = read_instance(args.instance_path)
initial_solution = read_solution(args.initial_solution)
initial_validation = validate_solution(instance, initial_solution)
if not initial_validation.is_feasible:
    print(f"ERROR: initial solution infeasible: {initial_validation.errors}")
    raise SystemExit(1)
initial_cost = objective_cost(instance, initial_solution)
print(f"initial cost: {initial_cost:.2f}")

solver = FixOptimizeSolver(
    initial_solution,
    SolverConfig(
        seed=args.seed,
        time_limit_seconds=args.time_limit,
        metadata={
            "fixopt_iterations": args.iterations,
            "fixopt_backend": args.backend,
            "mip_time_limit": args.mip_time_limit,
            "max_customers_per_subproblem": args.max_customers_per_subproblem,
            "max_routes_per_subproblem": args.max_routes_per_subproblem,
            "neighborhood_types": args.neighborhood_types,
            "max_extra_candidate_routes": args.max_candidate_routes,
            "max_subset_size_slack": args.max_subset_size_slack,
        },
    ),
)
t0 = time.perf_counter()
result = solver.solve(instance)
elapsed = time.perf_counter() - t0

if result.solution is None:
    print(f"FAILED: {result.metadata.get('error')}")
    raise SystemExit(1)

feasible = validate_solution(instance, result.solution).is_feasible
delta = result.cost - initial_cost
pct = 100 * delta / initial_cost
print(f"final cost: {result.cost:.2f} (delta={delta:+.2f}, {pct:+.2f}%) feasible={feasible} time={elapsed:.1f}s")
print(f"iterations={result.metadata.get('iterations')} improved={result.metadata.get('improved_subproblems')} "
      f"failed={result.metadata.get('failed_subproblems')} skipped={result.metadata.get('skipped_subproblems')}")

if feasible and result.cost < initial_cost:
    write_solution(result.solution, args.output, instance=instance)
    print(f"written: {args.output}")
else:
    print("no improvement -- not writing output")
