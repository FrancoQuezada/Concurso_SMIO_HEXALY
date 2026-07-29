from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from smio_clrp.core.distance import distance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


DISTANCE_SCALE = 10


def _global_matrix(instance) -> np.ndarray:
    if instance.distance_matrix is not None:
        return np.rint(instance.distance_matrix * DISTANCE_SCALE).astype(np.int64)
    nodes = [*instance.depots, *instance.customers]
    x = np.asarray([float(node.x) for node in nodes])
    y = np.asarray([float(node.y) for node in nodes])
    return np.rint(
        np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) * DISTANCE_SCALE
    ).astype(np.int64)


def _neighbor_pairs(instance, depot_ids: set[int], neighbors: int) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for depot_id in sorted(depot_ids):
        nearby = sorted(
            (other for other in depot_ids if other != depot_id),
            key=lambda other: (
                distance(instance, ('depot', depot_id), ('depot', other)),
                other,
            ),
        )
        for other in nearby[: max(1, neighbors)]:
            pairs.add(tuple(sorted((depot_id, other))))
    return sorted(
        pairs,
        key=lambda pair: (
            distance(instance, ('depot', pair[0]), ('depot', pair[1])),
            pair,
        ),
    )


def _optimize_pair(
    instance,
    solution: Solution,
    first_depot: int,
    second_depot: int,
    global_matrix: np.ndarray,
    time_limit: int,
) -> Solution:
    depot_ids = [first_depot, second_depot]
    pair_routes = [route for route in solution.routes if route.depot_id in depot_ids]
    untouched = [
        Route(route.depot_id, list(route.customer_ids))
        for route in solution.routes
        if route.depot_id not in depot_ids
    ]
    customers = [customer_id for route in pair_routes for customer_id in route.customer_ids]
    route_counts = {
        depot_id: sum(route.depot_id == depot_id for route in pair_routes)
        for depot_id in depot_ids
    }
    local_customer_index = {
        customer_id: index + len(depot_ids) for index, customer_id in enumerate(customers)
    }
    global_indices = [
        instance.node_index[('depot', depot_id)] for depot_id in depot_ids
    ] + [
        instance.node_index[('customer', customer_id)] for customer_id in customers
    ]
    matrix = global_matrix[np.ix_(global_indices, global_indices)]

    starts: list[int] = []
    vehicle_depot_ids: list[int] = []
    depot_vehicles: dict[int, list[int]] = {depot_id: [] for depot_id in depot_ids}
    for local_depot, depot_id in enumerate(depot_ids):
        depot = instance.depots_by_id[depot_id]
        vehicle_count = min(depot.vehicle_limit, route_counts[depot_id] + 1)
        for _ in range(vehicle_count):
            vehicle = len(starts)
            starts.append(local_depot)
            vehicle_depot_ids.append(depot_id)
            depot_vehicles[depot_id].append(vehicle)

    manager = pywrapcp.RoutingIndexManager(
        len(depot_ids) + len(customers),
        len(starts),
        starts,
        starts,
    )
    routing = pywrapcp.RoutingModel(manager)

    def transit(from_index: int, to_index: int) -> int:
        return int(matrix[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    transit_index = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    fixed_cost = int(round(instance.route_fixed_cost * DISTANCE_SCALE))
    for vehicle in range(len(starts)):
        routing.SetFixedCostOfVehicle(fixed_cost, vehicle)

    demands = [0] * len(depot_ids) + [
        int(round(instance.customers_by_id[customer_id].demand)) for customer_id in customers
    ]

    def demand(index: int) -> int:
        return demands[manager.IndexToNode(index)]

    demand_index = routing.RegisterUnaryTransitCallback(demand)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [int(round(instance.vehicle_capacity))] * len(starts),
        True,
        'Capacity',
    )
    dimension = routing.GetDimensionOrDie('Capacity')
    solver = routing.solver()
    active = [routing.ActiveVehicleVar(vehicle) for vehicle in range(len(starts))]
    solver.Add(solver.Sum(active) <= len(pair_routes))
    for depot_id in depot_ids:
        vehicles = depot_vehicles[depot_id]
        solver.Add(
            solver.Sum([dimension.CumulVar(routing.End(vehicle)) for vehicle in vehicles])
            <= int(round(instance.depots_by_id[depot_id].capacity))
        )
        solver.Add(solver.Sum([active[vehicle] for vehicle in vehicles]) >= 1)

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    )
    parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    parameters.time_limit.seconds = max(1, time_limit)

    vehicle_routes: list[list[int]] = [[] for _ in starts]
    next_vehicle = {depot_id: 0 for depot_id in depot_ids}
    for route in pair_routes:
        vehicle = depot_vehicles[route.depot_id][next_vehicle[route.depot_id]]
        vehicle_routes[vehicle] = [
            local_customer_index[customer_id] for customer_id in route.customer_ids
        ]
        next_vehicle[route.depot_id] += 1
    seed_assignment = routing.ReadAssignmentFromRoutes(vehicle_routes, True)
    if seed_assignment is None:
        return solution
    assignment = routing.SolveFromAssignmentWithParameters(seed_assignment, parameters)
    if assignment is None:
        return solution

    optimized_routes: list[Route] = []
    depot_count = len(depot_ids)
    for vehicle, depot_id in enumerate(vehicle_depot_ids):
        index = routing.Start(vehicle)
        customer_ids: list[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node >= depot_count:
                customer_ids.append(customers[node - depot_count])
            index = assignment.Value(routing.NextVar(index))
        if customer_ids:
            optimized_routes.append(Route(depot_id, customer_ids))

    candidate = Solution(solution.instance_name, untouched + optimized_routes)
    validation = validate_solution(instance, candidate)
    return candidate if validation.is_feasible else solution


def main() -> int:
    parser = argparse.ArgumentParser(description='Joint OR-Tools intensification of depot pairs')
    parser.add_argument('instance')
    parser.add_argument('seed_solution')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds-per-pair', type=int, default=5)
    parser.add_argument('--neighbors-per-depot', type=int, default=2)
    parser.add_argument('--passes', type=int, default=1)
    args = parser.parse_args()

    instance = read_instance(args.instance)
    seed = read_solution(args.seed_solution)
    validation = validate_solution(instance, seed)
    if not validation.is_feasible:
        raise ValueError('Seed solution is infeasible: ' + '; '.join(validation.errors))

    current = Solution(
        seed.instance_name,
        [Route(route.depot_id, list(route.customer_ids)) for route in seed.routes],
    )
    current_cost = objective_cost(instance, current)
    initial_cost = current_cost
    matrix = _global_matrix(instance)
    pairs = _neighbor_pairs(instance, current.opened_depot_ids, args.neighbors_per_depot)
    accepted = 0
    started = time.perf_counter()
    for pass_index in range(max(1, args.passes)):
        improved_pass = False
        for first_depot, second_depot in pairs:
            candidate = _optimize_pair(
                instance,
                current,
                first_depot,
                second_depot,
                matrix,
                args.seconds_per_pair,
            )
            candidate_cost = objective_cost(instance, candidate)
            if candidate_cost + 1e-9 < current_cost:
                delta = candidate_cost - current_cost
                current = candidate
                current_cost = candidate_cost
                accepted += 1
                improved_pass = True
                print(
                    f'pass={pass_index + 1} pair={first_depot},{second_depot} '
                    f'delta={delta:.2f} cost={current_cost:.2f}',
                    flush=True,
                )
        if not improved_pass:
            break

    final_validation = validate_solution(instance, current)
    if not final_validation.is_feasible:
        raise RuntimeError('Pair intensification produced an infeasible solution')
    write_solution(current, Path(args.output), instance=instance)
    print(f'solution: {args.output}')
    print('feasible: true')
    print(f'initial_cost: {initial_cost:.10f}')
    print(f'cost: {final_validation.cost:.10f}')
    print(f'improvement: {initial_cost - final_validation.cost:.10f}')
    print(f'pairs: {len(pairs)}')
    print(f'accepted_pairs: {accepted}')
    print(f'runtime_seconds: {time.perf_counter() - started:.3f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
