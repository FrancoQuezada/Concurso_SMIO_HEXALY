from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvrp

from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


DISTANCE_SCALE = 10


@dataclass
class DepotTask:
    depot_id: int
    customer_ids: list[int]
    demands: list[int]
    matrix: np.ndarray
    seed_routes: list[list[int]]
    vehicle_capacity: int
    route_fixed_cost: int
    time_limit: float
    seed: int


def _local_matrix(instance, depot_id: int, customer_ids: list[int]) -> np.ndarray:
    indices = [instance.node_index[('depot', depot_id)]] + [
        instance.node_index[('customer', customer_id)] for customer_id in customer_ids
    ]
    if instance.distance_matrix is not None:
        return np.rint(
            instance.distance_matrix[np.ix_(indices, indices)] * DISTANCE_SCALE
        ).astype(np.int64)
    nodes = [instance.depots_by_id[depot_id]] + [
        instance.customers_by_id[customer_id] for customer_id in customer_ids
    ]
    x = np.asarray([float(node.x) for node in nodes])
    y = np.asarray([float(node.y) for node in nodes])
    return np.rint(
        np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) * DISTANCE_SCALE
    ).astype(np.int64)


def _solve_depot(task: DepotTask) -> tuple[int, list[list[int]], float, int]:
    model = pyvrp.Model()
    locations = [model.add_depot(0.0, 0.0, name=f'd{task.depot_id}')]
    for customer_id, demand in zip(task.customer_ids, task.demands):
        locations.append(
            model.add_client(0.0, 0.0, delivery=demand, name=f'c{customer_id}')
        )
    model.add_vehicle_type(
        num_available=len(task.seed_routes),
        capacity=task.vehicle_capacity,
        start_depot=locations[0],
        end_depot=locations[0],
        fixed_cost=task.route_fixed_cost,
        name=f'depot_{task.depot_id}',
    )
    for from_index, origin in enumerate(locations):
        for to_index, destination in enumerate(locations):
            if from_index != to_index:
                model.add_edge(
                    origin,
                    destination,
                    distance=int(task.matrix[from_index, to_index]),
                )

    data = model.data()
    local_index = {
        customer_id: index + 1 for index, customer_id in enumerate(task.customer_ids)
    }
    initial = pyvrp.Solution(
        data,
        [
            pyvrp.Route(data, [local_index[customer_id] for customer_id in route], 0)
            for route in task.seed_routes
        ],
    )
    result = model.solve(
        pyvrp.stop.MaxRuntime(max(1.0, task.time_limit)),
        seed=task.seed,
        display=False,
        initial_solution=initial,
    )
    if not result.is_feasible():
        return task.depot_id, task.seed_routes, float('inf'), 0
    routes = [
        [task.customer_ids[index - 1] for index in route.visits()]
        for route in result.best.routes()
    ]
    return task.depot_id, routes, float(result.cost()), int(result.num_iterations)


def main() -> int:
    parser = argparse.ArgumentParser(description='Parallel per-depot PyVRP intensification')
    parser.add_argument('instance')
    parser.add_argument('seed_solution')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds-per-depot', type=float, default=20.0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    instance = read_instance(args.instance)
    seed_solution = read_solution(args.seed_solution)
    validation = validate_solution(instance, seed_solution)
    if not validation.is_feasible:
        raise ValueError('Seed solution is infeasible: ' + '; '.join(validation.errors))

    routes_by_depot: dict[int, list[list[int]]] = {}
    for route in seed_solution.routes:
        routes_by_depot.setdefault(route.depot_id, []).append(list(route.customer_ids))

    tasks: list[DepotTask] = []
    for depot_id, seed_routes in sorted(routes_by_depot.items()):
        customer_ids = [customer_id for route in seed_routes for customer_id in route]
        tasks.append(
            DepotTask(
                depot_id=depot_id,
                customer_ids=customer_ids,
                demands=[
                    int(round(instance.customers_by_id[customer_id].demand))
                    for customer_id in customer_ids
                ],
                matrix=_local_matrix(instance, depot_id, customer_ids),
                seed_routes=seed_routes,
                vehicle_capacity=int(round(instance.vehicle_capacity)),
                route_fixed_cost=int(round(instance.route_fixed_cost * DISTANCE_SCALE)),
                time_limit=args.seconds_per_depot,
                seed=args.seed + depot_id,
            )
        )

    solved: dict[int, list[list[int]]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_solve_depot, task): task.depot_id for task in tasks}
        for future in as_completed(futures):
            depot_id, routes, pyvrp_cost, iterations = future.result()
            solved[depot_id] = routes
            print(
                f'depot={depot_id} routes={len(routes)} '
                f'pyvrp_cost={pyvrp_cost:.0f} iterations={iterations}',
                flush=True,
            )

    solution = Solution(
        instance.name,
        [
            Route(depot_id, customer_ids)
            for depot_id in sorted(solved)
            for customer_ids in solved[depot_id]
        ],
    )
    final_validation = validate_solution(instance, solution)
    if not final_validation.is_feasible:
        raise RuntimeError('PyVRP intensification is infeasible: ' + '; '.join(final_validation.errors))
    write_solution(solution, Path(args.output), instance=instance)
    initial_cost = objective_cost(instance, seed_solution)
    final_cost = objective_cost(instance, solution)
    print(f'solution: {args.output}')
    print('feasible: true')
    print(f'initial_cost: {initial_cost:.10f}')
    print(f'cost: {final_cost:.10f}')
    print(f'improvement: {initial_cost - final_cost:.10f}')
    print(f'routes: {len(solution.routes)}')
    print(f'depots_opened: {len(solution.opened_depot_ids)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
