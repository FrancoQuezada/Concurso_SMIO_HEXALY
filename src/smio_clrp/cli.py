from __future__ import annotations

import argparse
import sys
from pathlib import Path

from smio_clrp.algorithms.base import SolverConfig
from smio_clrp.algorithms.alns import ALNSSolver
from smio_clrp.algorithms.constructive.greedy import GreedyNearestDepotSolver
from smio_clrp.algorithms.constructive.multistart import MultiStartConstructiveSolver
from smio_clrp.algorithms.constructive.regret import RegretInsertionSolver
from smio_clrp.algorithms.constructive.savings import SavingsConstructiveSolver
from smio_clrp.algorithms.clustering import ClusteredConstructiveSolver, ClusteredHybridSolver
from smio_clrp.algorithms.fixopt import FixOptimizeSolver, HybridALNSFixOptSolver
from smio_clrp.algorithms.local_search.solver import ConstructiveLocalSearchSolver
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.experiments.benchmark import run_benchmark_from_config
from smio_clrp.experiments.compare import compare_algorithms
from smio_clrp.experiments.registry import list_best, update_best_registry
from smio_clrp.experiments.run_batch import BatchConfig, run_batch
from smio_clrp.experiments.submission import build_submission_bundle
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clrp", description="SMIO-Hexaly CLRP toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Read an instance and print statistics")
    parse_parser.add_argument("instance_path")
    parse_parser.set_defaults(func=_cmd_parse)

    solve_parser = subparsers.add_parser("solve", help="Solve an instance")
    solve_parser.add_argument("instance_path")
    solve_parser.add_argument("--algorithm", default="greedy_nearest_depot")
    solve_parser.add_argument("--output", required=True)
    solve_parser.add_argument("--seed", type=int, default=1)
    solve_parser.add_argument("--num-starts", type=int, default=10)
    solve_parser.add_argument("--time-limit", type=float, default=None)
    solve_parser.add_argument("--regret-k", type=int, choices=[2, 3], default=2)
    solve_parser.add_argument("--local-search", action="store_true")
    solve_parser.add_argument("--max-iterations", type=int, default=50)
    solve_parser.add_argument("--destroy-fraction-min", type=float, default=0.15)
    solve_parser.add_argument("--destroy-fraction-max", type=float, default=0.35)
    solve_parser.add_argument("--initial-temperature", type=float, default=None, help="Absolute SA start temperature. Default: auto-scale to 1%% of the initial solution cost.")
    solve_parser.add_argument("--cooling-rate", type=float, default=0.995)
    solve_parser.add_argument("--local-search-frequency", default="best")
    solve_parser.add_argument("--verbose", action="store_true")
    solve_parser.add_argument("--fixopt-iterations", type=int, default=50)
    solve_parser.add_argument("--fixopt-time-limit", type=float, default=None)
    solve_parser.add_argument("--fixopt-backend", choices=["auto", "heuristic", "mip"], default="auto")
    solve_parser.add_argument("--max-customers-per-subproblem", type=int, default=12)
    solve_parser.add_argument("--max-routes-per-subproblem", type=int, default=3)
    solve_parser.add_argument("--mip-time-limit", type=float, default=5.0)
    solve_parser.add_argument("--neighborhood-types", default=None)
    solve_parser.add_argument("--cluster-method", choices=["auto", "kmeans", "kmedoids"], default="auto")
    solve_parser.add_argument("--cluster-count", type=int, default=None)
    solve_parser.add_argument("--cluster-iterations", type=int, default=20)
    solve_parser.set_defaults(func=_cmd_solve)

    validate_parser = subparsers.add_parser("validate", help="Validate a solution")
    validate_parser.add_argument("instance_path")
    validate_parser.add_argument("solution_path")
    validate_parser.set_defaults(func=_cmd_validate)

    cost_parser = subparsers.add_parser("cost", help="Print recomputed objective cost")
    cost_parser.add_argument("instance_path")
    cost_parser.add_argument("solution_path")
    cost_parser.set_defaults(func=_cmd_cost)

    batch_parser = subparsers.add_parser("batch-solve", help="Solve all .txt instances in a directory")
    batch_parser.add_argument("instance_dir")
    batch_parser.add_argument("--output-dir", required=True)
    batch_parser.add_argument("--algorithm", default="greedy_nearest_depot")
    batch_parser.add_argument("--seed", type=int, default=1)
    batch_parser.add_argument("--num-starts", type=int, default=10)
    batch_parser.add_argument("--time-limit", type=float, default=None)
    batch_parser.add_argument("--regret-k", type=int, choices=[2, 3], default=2)
    batch_parser.add_argument("--local-search", action="store_true")
    batch_parser.add_argument("--max-iterations", type=int, default=50)
    batch_parser.add_argument("--destroy-fraction-min", type=float, default=0.15)
    batch_parser.add_argument("--destroy-fraction-max", type=float, default=0.35)
    batch_parser.add_argument("--initial-temperature", type=float, default=None, help="Absolute SA start temperature. Default: auto-scale to 1%% of the initial solution cost.")
    batch_parser.add_argument("--cooling-rate", type=float, default=0.995)
    batch_parser.add_argument("--local-search-frequency", default="best")
    batch_parser.add_argument("--verbose", action="store_true")
    batch_parser.add_argument("--fixopt-iterations", type=int, default=50)
    batch_parser.add_argument("--fixopt-time-limit", type=float, default=None)
    batch_parser.add_argument("--fixopt-backend", choices=["auto", "heuristic", "mip"], default="auto")
    batch_parser.add_argument("--max-customers-per-subproblem", type=int, default=12)
    batch_parser.add_argument("--max-routes-per-subproblem", type=int, default=3)
    batch_parser.add_argument("--mip-time-limit", type=float, default=5.0)
    batch_parser.add_argument("--neighborhood-types", default=None)
    batch_parser.add_argument("--cluster-method", choices=["auto", "kmeans", "kmedoids"], default="auto")
    batch_parser.add_argument("--cluster-count", type=int, default=None)
    batch_parser.add_argument("--cluster-iterations", type=int, default=20)
    batch_parser.set_defaults(func=_cmd_batch_solve)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run a JSON-configured benchmark")
    benchmark_parser.add_argument("--config", required=True)
    benchmark_parser.set_defaults(func=_cmd_benchmark)

    update_best_parser = subparsers.add_parser("update-best", help="Update local best-solution registry")
    update_best_parser.add_argument("--instance-dir", required=True)
    update_best_parser.add_argument("--run-dir", required=True)
    update_best_parser.add_argument("--best-dir", default="results/best")
    update_best_parser.add_argument("--replace-ties", action="store_true")
    update_best_parser.set_defaults(func=_cmd_update_best)

    list_best_parser = subparsers.add_parser("list-best", help="List local best solutions")
    list_best_parser.add_argument("--best-dir", default="results/best")
    list_best_parser.set_defaults(func=_cmd_list_best)

    compare_parser = subparsers.add_parser("compare", help="Compare algorithms from a summary CSV")
    compare_parser.add_argument("--summary-csv", required=True)
    compare_parser.add_argument("--output-csv", default=None)
    compare_parser.set_defaults(func=_cmd_compare)

    bundle_parser = subparsers.add_parser("build-submission", help="Build a validated solution bundle")
    bundle_parser.add_argument("--instance-dir", required=True)
    bundle_parser.add_argument("--solution-dir", required=True)
    bundle_parser.add_argument("--output", default=None)
    bundle_parser.set_defaults(func=_cmd_build_submission)
    return parser


def _cmd_parse(args: argparse.Namespace) -> int:
    instance = read_instance(args.instance_path)
    total_demand = sum(customer.demand for customer in instance.customers)
    print(f"name: {instance.name}")
    print(f"depots: {len(instance.depots)}")
    print(f"customers: {len(instance.customers)}")
    print(f"vehicle_capacity: {instance.vehicle_capacity:g}")
    print(f"route_fixed_cost: {instance.route_fixed_cost:g}")
    print(f"distance_format: {instance.distance_format}")
    print(f"total_demand: {total_demand:g}")
    return 0


def _cmd_solve(args: argparse.Namespace) -> int:
    instance = read_instance(args.instance_path)
    solver = _make_solver(args)
    result = solver.solve(instance)
    if result.solution is None:
        print(f"feasible: false")
        print(f"error: {result.metadata.get('error', 'solver failed')}")
        return 1
    write_solution(result.solution, args.output, instance=instance)
    written = read_solution(args.output)
    validation = validate_solution(instance, written)
    print(f"solution: {args.output}")
    print(f"feasible: {str(validation.is_feasible).lower()}")
    print(f"cost: {validation.cost:.10f}")
    if validation.errors:
        print("errors:")
        for error in validation.errors:
            print(f"- {error}")
        return 1
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    instance = read_instance(args.instance_path)
    solution = read_solution(args.solution_path)
    validation = validate_solution(instance, solution)
    print(f"feasible: {str(validation.is_feasible).lower()}")
    print(f"cost: {validation.cost:.10f}")
    if validation.warnings:
        print("warnings:")
        for warning in validation.warnings:
            print(f"- {warning}")
    if validation.errors:
        print("errors:")
        for error in validation.errors:
            print(f"- {error}")
        return 1
    return 0


def _cmd_cost(args: argparse.Namespace) -> int:
    instance = read_instance(args.instance_path)
    solution = read_solution(args.solution_path)
    print(f"{objective_cost(instance, solution):.10f}")
    return 0


def _cmd_batch_solve(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir)
    parameters = _solver_parameters_from_args(args)
    rows = run_batch(
        BatchConfig(
            instance_dir=args.instance_dir,
            output_dir=output_dir.parent,
            run_id=output_dir.name,
            algorithms=[args.algorithm],
            seeds=[args.seed],
            time_limits={args.algorithm: args.time_limit},
            algorithm_parameters={args.algorithm: parameters},
            overwrite=True,
        )
    )
    print(f"summary: {output_dir / 'summary.csv'}")
    return 0 if all(row.feasible for row in rows) else 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    rows = run_benchmark_from_config(args.config)
    print(f"runs: {len(rows)}")
    return 0


def _cmd_update_best(args: argparse.Namespace) -> int:
    accepted = update_best_registry(args.instance_dir, args.run_dir, args.best_dir, args.replace_ties)
    print(f"accepted: {len(accepted)}")
    return 0


def _cmd_list_best(args: argparse.Namespace) -> int:
    for row in list_best(args.best_dir):
        print(f"{row['instance']},{row['cost']},{row['solution_path']}")
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    rows = compare_algorithms(args.summary_csv, args.output_csv)
    print(f"rows: {len(rows)}")
    return 0


def _cmd_build_submission(args: argparse.Namespace) -> int:
    output = build_submission_bundle(args.instance_dir, args.solution_dir, args.output)
    print(f"bundle: {output}")
    return 0


def _make_solver(args: argparse.Namespace):
    parameters = _solver_parameters_from_args(args)
    config = SolverConfig(
        seed=args.seed,
        time_limit_seconds=args.time_limit,
        metadata=parameters,
    )
    _base_constructive_algorithms = {"greedy_nearest_depot", "savings", "regret", "multistart"}
    if args.local_search and args.algorithm in _base_constructive_algorithms:
        args.algorithm = "constructive_ls"
    if args.algorithm == "greedy_nearest_depot":
        return GreedyNearestDepotSolver(config)
    if args.algorithm == "savings":
        return SavingsConstructiveSolver(config)
    if args.algorithm == "regret":
        return RegretInsertionSolver(config)
    if args.algorithm == "multistart":
        return MultiStartConstructiveSolver(config)
    if args.algorithm == "constructive_ls":
        return ConstructiveLocalSearchSolver(config)
    if args.algorithm == "alns":
        return ALNSSolver(config)
    if args.algorithm == "fixopt":
        return FixOptimizeSolver(config=config)
    if args.algorithm == "hybrid":
        return HybridALNSFixOptSolver(config)
    if args.algorithm == "clustered":
        return ClusteredConstructiveSolver(config)
    if args.algorithm == "clustered_hybrid":
        return ClusteredHybridSolver(config)
    available = "greedy_nearest_depot, savings, regret, multistart, constructive_ls, alns, fixopt, hybrid, clustered, clustered_hybrid"
    raise ValueError(f"Unsupported algorithm '{args.algorithm}'. Available: {available}")


def _solver_parameters_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "num_starts": args.num_starts,
        "regret_k": args.regret_k,
        "max_iterations": args.max_iterations,
        "destroy_fraction_min": args.destroy_fraction_min,
        "destroy_fraction_max": args.destroy_fraction_max,
        "initial_temperature": args.initial_temperature,
        "cooling_rate": args.cooling_rate,
        "local_search_frequency": args.local_search_frequency,
        "verbose": args.verbose,
        "fixopt_iterations": args.fixopt_iterations,
        "fixopt_time_limit": args.fixopt_time_limit,
        "fixopt_backend": args.fixopt_backend,
        "max_customers_per_subproblem": args.max_customers_per_subproblem,
        "max_routes_per_subproblem": args.max_routes_per_subproblem,
        "mip_time_limit": args.mip_time_limit,
        "neighborhood_types": args.neighborhood_types,
        "cluster_method": args.cluster_method,
        "cluster_count": args.cluster_count,
        "cluster_iterations": args.cluster_iterations,
    }


if __name__ == "__main__":
    raise SystemExit(main())
