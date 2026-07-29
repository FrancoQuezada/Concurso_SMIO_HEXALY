from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

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
    time_limit: int


def _local_distance_matrix(instance, depot_id: int, customer_ids: list[int]) -> np.ndarray:
    """Return integer arc costs matching the repository distance convention."""
    global_indices = [instance.node_index[("depot", depot_id)]] + [
        instance.node_index[("customer", customer_id)] for customer_id in customer_ids
    ]
    if instance.distance_matrix is not None:
        distances = instance.distance_matrix[np.ix_(global_indices, global_indices)]
        return np.rint(distances * DISTANCE_SCALE).astype(np.int64)

    nodes = [instance.depots_by_id[depot_id]] + [
        instance.customers_by_id[customer_id] for customer_id in customer_ids
    ]
    x = np.asarray([float(node.x) for node in nodes])
    y = np.asarray([float(node.y) for node in nodes])
    distances = np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :])
    return np.rint(distances * DISTANCE_SCALE).astype(np.int64)


def _solve_depot(task: DepotTask) -> tuple[int, list[list[int]], int]:
    vehicles = len(task.seed_routes)
    manager = pywrapcp.RoutingIndexManager(len(task.customer_ids) + 1, vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def transit(from_index: int, to_index: int) -> int:
        return int(task.matrix[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    transit_index = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    for vehicle in range(vehicles):
        routing.SetFixedCostOfVehicle(task.route_fixed_cost, vehicle)

    local_demands = [0, *task.demands]

    def demand(index: int) -> int:
        return local_demands[manager.IndexToNode(index)]

    demand_index = routing.RegisterUnaryTransitCallback(demand)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [task.vehicle_capacity] * vehicles,
        True,
        "Capacity",
    )

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    parameters.time_limit.seconds = task.time_limit

    id_to_local = {customer_id: index + 1 for index, customer_id in enumerate(task.customer_ids)}
    seed_local = [[id_to_local[customer_id] for customer_id in route] for route in task.seed_routes]
    seed_assignment = routing.ReadAssignmentFromRoutes(seed_local, True)
    assignment = None
    if seed_assignment is not None:
        assignment = routing.SolveFromAssignmentWithParameters(seed_assignment, parameters)
    if assignment is None:
        assignment = routing.SolveWithParameters(parameters)
    if assignment is None:
        return task.depot_id, task.seed_routes, -1

    routes: list[list[int]] = []
    objective = assignment.ObjectiveValue()
    for vehicle in range(vehicles):
        index = routing.Start(vehicle)
        route: list[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node:
                route.append(task.customer_ids[node - 1])
            index = assignment.Value(routing.NextVar(index))
        if route:
            routes.append(route)
    return task.depot_id, routes, objective


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel per-depot CLRP intensification")
    parser.add_argument("instance")
    parser.add_argument("seed_solution")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds-per-depot", type=int, default=30)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    instance = read_instance(args.instance)
    seed = read_solution(args.seed_solution)
    validation = validate_solution(instance, seed)
    if not validation.is_feasible:
        raise ValueError("Seed solution is infeasible: " + "; ".join(validation.errors))

    routes_by_depot: dict[int, list[list[int]]] = {}
    for route in seed.routes:
        routes_by_depot.setdefault(route.depot_id, []).append(list(route.customer_ids))

    tasks: list[DepotTask] = []
    for depot_id, seed_routes in sorted(routes_by_depot.items()):
        customer_ids = [customer_id for route in seed_routes for customer_id in route]
        local_matrix = _local_distance_matrix(instance, depot_id, customer_ids)
        tasks.append(
            DepotTask(
                depot_id=depot_id,
                customer_ids=customer_ids,
                demands=[int(round(instance.customers_by_id[c].demand)) for c in customer_ids],
                matrix=local_matrix,
                seed_routes=seed_routes,
                vehicle_capacity=int(round(instance.vehicle_capacity)),
                route_fixed_cost=int(round(instance.route_fixed_cost * DISTANCE_SCALE)),
                time_limit=max(1, args.seconds_per_depot),
            )
        )

    solved: dict[int, list[list[int]]] = {}
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(_solve_depot, task): task.depot_id for task in tasks}
        for future in as_completed(futures):
            depot_id, routes, depot_objective = future.result()
            solved[depot_id] = routes
            print(f"depot={depot_id} routes={len(routes)} routing_objective={depot_objective}", flush=True)

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
        raise RuntimeError("Intensified solution is infeasible: " + "; ".join(final_validation.errors))
    write_solution(solution, Path(args.output), instance=instance)
    print(f"solution: {args.output}")
    print("feasible: true")
    print(f"cost: {objective_cost(instance, solution):.10f}")
    print(f"routes: {len(solution.routes)}")
    print(f"depots_opened: {len(solution.opened_depot_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
