from __future__ import annotations

import argparse
import time
from collections import Counter
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


def _parse_depots(raw: str | None, available: set[int]) -> set[int]:
    if raw is None:
        return set(available)
    selected = {int(item.strip()) for item in raw.split(',') if item.strip()}
    unknown = selected - available
    if unknown:
        raise ValueError(f'Unknown depot IDs: {sorted(unknown)}')
    if not selected:
        raise ValueError('At least one depot must be allowed')
    return selected


def _scaled_distance_matrix(instance) -> np.ndarray:
    if instance.distance_matrix is not None:
        return np.rint(instance.distance_matrix * DISTANCE_SCALE).astype(np.int64)

    nodes = [*instance.depots, *instance.customers]
    x = np.asarray([float(node.x) for node in nodes])
    y = np.asarray([float(node.y) for node in nodes])
    return np.rint(
        np.hypot(x[:, None] - x[None, :], y[:, None] - y[None, :]) * DISTANCE_SCALE
    ).astype(np.int64)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve a large CLRP instance with OR-Tools")
    parser.add_argument("instance")
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-limit", type=int, default=600)
    parser.add_argument("--seed-solution")
    parser.add_argument("--log-search", action="store_true")
    parser.add_argument('--depots', help='Comma-separated allowed depot IDs')
    parser.add_argument('--require-all-depots', action='store_true')
    parser.add_argument('--max-routes', type=int)
    parser.add_argument(
        '--metaheuristic',
        choices=['gls', 'tabu', 'generic_tabu', 'simulated_annealing'],
        default='gls',
    )
    parser.add_argument('--gls-lambda', type=float, default=0.1)
    parser.add_argument('--multi-armed-bandit', action='store_true')
    parser.add_argument(
        '--vehicle-slack-per-depot',
        type=int,
        help='With a seed, configure at most its route count plus this many vehicles per depot',
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    instance = read_instance(args.instance)
    if instance.distance_format not in {'COORDS', 'FULL_MATRIX'}:
        raise ValueError("This competition solver requires a FULL_MATRIX instance")
    if any(abs(customer.demand - round(customer.demand)) > 1e-9 for customer in instance.customers):
        raise ValueError("OR-Tools capacity dimensions require integral customer demands")
    if abs(instance.vehicle_capacity - round(instance.vehicle_capacity)) > 1e-9:
        raise ValueError("OR-Tools capacity dimensions require integral vehicle capacity")

    if args.max_routes is not None and args.max_routes < 1:
        raise ValueError('--max-routes must be positive')
    if args.vehicle_slack_per_depot is not None and args.vehicle_slack_per_depot < 0:
        raise ValueError('--vehicle-slack-per-depot cannot be negative')
    if args.gls_lambda <= 0:
        raise ValueError('--gls-lambda must be positive')

    allowed_depot_ids = _parse_depots(args.depots, set(instance.depots_by_id))
    seed = None
    seed_route_counts: Counter[int] = Counter()
    if args.seed_solution:
        seed = read_solution(args.seed_solution)
        seed_validation = validate_solution(instance, seed)
        if not seed_validation.is_feasible:
            raise ValueError('Seed solution is infeasible: ' + '; '.join(seed_validation.errors))
        disallowed = seed.opened_depot_ids - allowed_depot_ids
        if disallowed:
            raise ValueError(f'Seed uses disallowed depots: {sorted(disallowed)}')
        seed_route_counts.update(route.depot_id for route in seed.routes)
    elif args.vehicle_slack_per_depot is not None:
        raise ValueError('--vehicle-slack-per-depot requires --seed-solution')

    starts: list[int] = []
    vehicle_depot_ids: list[int] = []
    depot_vehicle_indices: dict[int, list[int]] = {depot.id: [] for depot in instance.depots}
    disabled_vehicle_indices: list[int] = []
    for depot_index, depot in enumerate(instance.depots):
        is_allowed = depot.id in allowed_depot_ids
        vehicle_count = depot.vehicle_limit if is_allowed else 1
        if is_allowed and args.vehicle_slack_per_depot is not None:
            vehicle_count = min(
                vehicle_count,
                max(1, seed_route_counts[depot.id] + args.vehicle_slack_per_depot),
            )
        if vehicle_count < seed_route_counts[depot.id]:
            raise ValueError(f'Configured fleet is smaller than the seed at depot {depot.id}')
        for _ in range(vehicle_count):
            vehicle = len(starts)
            starts.append(depot_index)
            vehicle_depot_ids.append(depot.id)
            depot_vehicle_indices[depot.id].append(vehicle)
            if not is_allowed:
                disabled_vehicle_indices.append(vehicle)

    manager = pywrapcp.RoutingIndexManager(
        len(instance.depots) + len(instance.customers),
        len(starts),
        starts,
        starts,
    )
    routing = pywrapcp.RoutingModel(manager)
    matrix = _scaled_distance_matrix(instance)

    def transit(from_index: int, to_index: int) -> int:
        return int(matrix[manager.IndexToNode(from_index), manager.IndexToNode(to_index)])

    transit_index = routing.RegisterTransitCallback(transit)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_index)
    for vehicle, depot_id in enumerate(vehicle_depot_ids):
        depot = instance.depots_by_id[depot_id]
        # If every allowed depot is required, opening cost is constant and must not
        # distort route moves. Otherwise use a small amortized opening-cost proxy.
        amortized_opening = (
            0.0 if args.require_all_depots else depot.opening_cost / max(1, depot.vehicle_limit)
        )
        fixed_cost = (instance.route_fixed_cost + amortized_opening) * DISTANCE_SCALE
        routing.SetFixedCostOfVehicle(int(round(fixed_cost)), vehicle)

    depot_count = len(instance.depots)
    demands = [0] * depot_count + [int(round(customer.demand)) for customer in instance.customers]

    def demand(index: int) -> int:
        return demands[manager.IndexToNode(index)]

    demand_index = routing.RegisterUnaryTransitCallback(demand)
    routing.AddDimensionWithVehicleCapacity(
        demand_index,
        0,
        [int(round(instance.vehicle_capacity))] * len(starts),
        True,
        "Capacity",
    )
    capacity_dimension = routing.GetDimensionOrDie("Capacity")
    solver = routing.solver()
    active_vehicles = [routing.ActiveVehicleVar(vehicle) for vehicle in range(len(starts))]
    for vehicle in disabled_vehicle_indices:
        solver.Add(active_vehicles[vehicle] == 0)
    if args.max_routes is not None:
        solver.Add(solver.Sum(active_vehicles) <= args.max_routes)
    for depot in instance.depots:
        if depot.id not in allowed_depot_ids:
            continue
        end_loads = [capacity_dimension.CumulVar(routing.End(v)) for v in depot_vehicle_indices[depot.id]]
        solver.Add(solver.Sum(end_loads) <= int(round(depot.capacity)))
        if args.require_all_depots:
            depot_active = [routing.ActiveVehicleVar(v) for v in depot_vehicle_indices[depot.id]]
            solver.Add(solver.Sum(depot_active) >= 1)

    parameters = pywrapcp.DefaultRoutingSearchParameters()
    parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PARALLEL_CHEAPEST_INSERTION
    metaheuristics = {
        'gls': routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH,
        'tabu': routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH,
        'generic_tabu': routing_enums_pb2.LocalSearchMetaheuristic.GENERIC_TABU_SEARCH,
        'simulated_annealing': routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING,
    }
    parameters.local_search_metaheuristic = metaheuristics[args.metaheuristic]
    parameters.guided_local_search_lambda_coefficient = args.gls_lambda
    parameters.use_multi_armed_bandit_concatenate_operators = args.multi_armed_bandit
    parameters.time_limit.seconds = max(1, args.time_limit)
    parameters.log_search = args.log_search

    started = time.perf_counter()
    assignment = None
    if args.seed_solution:
        seed = read_solution(args.seed_solution)
        seed_validation = validate_solution(instance, seed)
        if not seed_validation.is_feasible:
            raise ValueError("Seed solution is infeasible: " + "; ".join(seed_validation.errors))
        vehicle_routes: list[list[int]] = [[] for _ in starts]
        next_vehicle = {depot.id: 0 for depot in instance.depots}
        for route in seed.routes:
            slot = next_vehicle[route.depot_id]
            vehicles = depot_vehicle_indices[route.depot_id]
            if slot >= len(vehicles):
                raise ValueError(f"Seed uses too many vehicles at depot {route.depot_id}")
            vehicle_routes[vehicles[slot]] = [instance.node_index[("customer", cid)] for cid in route.customer_ids]
            next_vehicle[route.depot_id] += 1
        seed_assignment = routing.ReadAssignmentFromRoutes(vehicle_routes, True)
        if seed_assignment is not None:
            assignment = routing.SolveFromAssignmentWithParameters(seed_assignment, parameters)
    if assignment is None:
        assignment = routing.SolveWithParameters(parameters)
    if assignment is None:
        raise RuntimeError("OR-Tools did not find a feasible solution")

    routes: list[Route] = []
    for vehicle, depot_id in enumerate(vehicle_depot_ids):
        index = routing.Start(vehicle)
        customer_ids: list[int] = []
        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            if node >= depot_count:
                customer_ids.append(instance.customers[node - depot_count].id)
            index = assignment.Value(routing.NextVar(index))
        if customer_ids:
            routes.append(Route(depot_id, customer_ids))

    solution = Solution(instance.name, routes)
    validation = validate_solution(instance, solution)
    if not validation.is_feasible:
        raise RuntimeError("Extracted solution is infeasible: " + "; ".join(validation.errors))
    write_solution(solution, Path(args.output), instance=instance)
    print(f"solution: {args.output}")
    print("feasible: true")
    print(f"cost: {objective_cost(instance, solution):.10f}")
    print(f"routes: {len(routes)}")
    print(f"depots_opened: {len(solution.opened_depot_ids)}")
    print(f"runtime_seconds: {time.perf_counter() - started:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
