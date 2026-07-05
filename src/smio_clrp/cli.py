from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from smio_clrp.algorithms.base import SolverConfig
from smio_clrp.algorithms.constructive.greedy import GreedyNearestDepotSolver
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
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

    solve_parser = subparsers.add_parser("solve", help="Solve an instance with a baseline algorithm")
    solve_parser.add_argument("instance_path")
    solve_parser.add_argument("--algorithm", default="greedy_nearest_depot")
    solve_parser.add_argument("--output", required=True)
    solve_parser.add_argument("--seed", type=int, default=1)
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
    batch_parser.set_defaults(func=_cmd_batch_solve)
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
    _ensure_supported_algorithm(args.algorithm)
    instance = read_instance(args.instance_path)
    solver = GreedyNearestDepotSolver(SolverConfig(seed=args.seed))
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
    _ensure_supported_algorithm(args.algorithm)
    instance_dir = Path(args.instance_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.csv"
    rows: list[dict[str, object]] = []
    for instance_path in sorted(instance_dir.glob("*.txt")):
        instance = read_instance(instance_path)
        solver = GreedyNearestDepotSolver(SolverConfig(seed=args.seed))
        result = solver.solve(instance)
        output_path = output_dir / f"{instance_path.stem}.sol"
        error = ""
        if result.solution is not None:
            write_solution(result.solution, output_path, instance=instance)
        else:
            error = str(result.metadata.get("error", "solver failed"))
        rows.append(
            {
                "instance": instance.name,
                "cost": "" if result.cost is None else f"{result.cost:.10f}",
                "feasible": result.feasible,
                "runtime": f"{result.runtime_seconds:.6f}",
                "algorithm": result.algorithm_name,
                "seed": args.seed,
                "error": error,
            }
        )
    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["instance", "cost", "feasible", "runtime", "algorithm", "seed", "error"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: {summary_path}")
    return 0 if all(row["feasible"] for row in rows) else 1


def _ensure_supported_algorithm(algorithm: str) -> None:
    if algorithm != "greedy_nearest_depot":
        raise ValueError(f"Unsupported algorithm '{algorithm}'. Available: greedy_nearest_depot")


if __name__ == "__main__":
    raise SystemExit(main())
