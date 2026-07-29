from __future__ import annotations

import argparse
import time
from pathlib import Path

from smio_clrp.algorithms.base import SolverConfig
from smio_clrp.algorithms.fixopt.fixopt_solver import FixOptimizeSolver
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


def main() -> int:
    parser = argparse.ArgumentParser(description='Configurable large-neighborhood FixOpt runner')
    parser.add_argument('instance')
    parser.add_argument('seed_solution')
    parser.add_argument('--output', required=True)
    parser.add_argument('--backend', choices=['heuristic', 'mip'], default='heuristic')
    parser.add_argument('--iterations', type=int, default=100)
    parser.add_argument('--time-limit', type=float, default=180.0)
    parser.add_argument('--mip-time-limit', type=float, default=5.0)
    parser.add_argument('--max-customers', type=int, default=24)
    parser.add_argument('--max-routes', type=int, default=4)
    parser.add_argument(
        '--neighborhoods',
        default='boundary,route_pair,depot,expensive,route',
    )
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    instance = read_instance(args.instance)
    initial = read_solution(args.seed_solution)
    initial_validation = validate_solution(instance, initial)
    if not initial_validation.is_feasible:
        raise ValueError('Seed solution is infeasible: ' + '; '.join(initial_validation.errors))

    config = SolverConfig(
        seed=args.seed,
        time_limit_seconds=args.time_limit,
        metadata={
            'fixopt_iterations': args.iterations,
            'fixopt_backend': args.backend,
            'fixopt_time_limit': args.time_limit,
            'mip_time_limit': args.mip_time_limit,
            'max_customers_per_subproblem': args.max_customers,
            'max_routes_per_subproblem': args.max_routes,
            'neighborhood_types': args.neighborhoods,
        },
    )
    initial_cost = objective_cost(instance, initial)
    started = time.perf_counter()
    result = FixOptimizeSolver(initial, config).solve(instance)
    elapsed = time.perf_counter() - started
    if result.solution is None:
        raise RuntimeError(str(result.metadata.get('error', 'FixOpt failed')))
    validation = validate_solution(instance, result.solution)
    if not validation.is_feasible:
        raise RuntimeError('FixOpt output is infeasible: ' + '; '.join(validation.errors))

    write_solution(result.solution, Path(args.output), instance=instance)
    iterations = result.metadata.get('iterations')
    improved_subproblems = result.metadata.get('improved_subproblems')
    failed_subproblems = result.metadata.get('failed_subproblems')
    skipped_subproblems = result.metadata.get('skipped_subproblems')
    print(f'solution: {args.output}')
    print('feasible: true')
    print(f'initial_cost: {initial_cost:.10f}')
    print(f'cost: {validation.cost:.10f}')
    print(f'improvement: {initial_cost - validation.cost:.10f}')
    print(f'iterations: {iterations}')
    print(f'improved_subproblems: {improved_subproblems}')
    print(f'failed_subproblems: {failed_subproblems}')
    print(f'skipped_subproblems: {skipped_subproblems}')
    print(f'runtime_seconds: {elapsed:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
