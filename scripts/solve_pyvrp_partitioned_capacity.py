from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict
from pathlib import Path

import pyvrp
from pyvrp.search import (
    Exchange10,
    Exchange11,
    Exchange20,
    Exchange21,
    Exchange22,
    Exchange30,
    Exchange31,
    Exchange32,
    Exchange33,
    PerturbationParams,
    RelocateWithDepot,
    SwapRoutes,
    SwapStar,
    SwapTails,
)
from pyvrp.solve import IteratedLocalSearchParams, NeighbourhoodParams, SolveParams

from smio_clrp.algorithms.common import route_load
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


DISTANCE_SCALE = 10


def _partition_capacities(
    loads: list[int], total: int, vehicle: int, strategy: str, rng: random.Random
) -> list[int]:
    capacities = list(loads)
    slack = min(total, len(loads) * vehicle) - sum(loads)
    if slack < 0:
        raise ValueError('Route loads exceed the depot capacity partition')
    if strategy in {'fill', 'random-fill'}:
        order = sorted(range(len(loads)), key=lambda index: (-loads[index], index))
        if strategy == 'random-fill':
            rng.shuffle(order)
        for index in order:
            added = min(slack, vehicle - capacities[index])
            capacities[index] += added
            slack -= added
    elif strategy == 'balanced':
        while slack:
            index = min(
                (idx for idx in range(len(capacities)) if capacities[idx] < vehicle),
                key=lambda idx: (capacities[idx], idx),
            )
            capacities[index] += 1
            slack -= 1
    elif strategy == 'random-spread':
        while slack:
            eligible = [idx for idx, capacity in enumerate(capacities) if capacity < vehicle]
            if not eligible:
                break
            capacities[rng.choice(eligible)] += 1
            slack -= 1
    else:
        # Spread headroom proportionally, one unit at a time. This preserves
        # flexibility in every route while favouring routes with more room.
        while slack:
            eligible = [idx for idx, capacity in enumerate(capacities) if capacity < vehicle]
            if not eligible:
                break
            for index in eligible:
                if slack == 0:
                    break
                capacities[index] += 1
                slack -= 1
    return capacities


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Global PyVRP search with depot capacity partitioned among vehicles'
    )
    parser.add_argument('instance')
    parser.add_argument('initial_solution')
    parser.add_argument('--output', required=True)
    parser.add_argument('--time-limit', type=float, default=300.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--strategy',
        choices=['spread', 'balanced', 'fill', 'random-fill', 'random-spread'],
        default='spread',
    )
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--restart-iterations', type=int, default=150000)
    parser.add_argument('--history-length', type=int, default=300)
    parser.add_argument('--num-neighbours', type=int, default=50)
    parser.add_argument('--min-perturbations', type=int, default=1)
    parser.add_argument('--max-perturbations', type=int, default=25)
    parser.add_argument('--extended-operators', action='store_true')
    args = parser.parse_args()

    instance = read_instance(args.instance)
    seed = read_solution(args.initial_solution)
    validation = validate_solution(instance, seed)
    if not validation.is_feasible:
        raise ValueError('Initial solution is infeasible: ' + '; '.join(validation.errors))

    model = pyvrp.Model()
    locations = []
    for depot in instance.depots:
        locations.append(model.add_depot(float(depot.x), float(depot.y), name=f'd{depot.id}'))
    for customer in instance.customers:
        locations.append(
            model.add_client(
                float(customer.x),
                float(customer.y),
                delivery=int(round(customer.demand)),
                name=f'c{customer.id}',
            )
        )

    depot_index_by_id = {depot.id: index for index, depot in enumerate(instance.depots)}
    indexed_routes: dict[int, list[tuple[int, Route]]] = defaultdict(list)
    for route_index, route in enumerate(seed.routes):
        indexed_routes[route.depot_id].append((route_index, route))

    vehicle_type_by_route: dict[int, int] = {}
    capacity_summary: dict[int, list[int]] = {}
    for depot_id, depot_routes in sorted(indexed_routes.items()):
        loads = [int(round(route_load(instance, route))) for _, route in depot_routes]
        capacities = _partition_capacities(
            loads,
            int(round(instance.depots_by_id[depot_id].capacity)),
            int(round(instance.vehicle_capacity)),
            args.strategy,
            random.Random(args.seed * 1009 + depot_id),
        )
        capacity_summary[depot_id] = capacities
        for (route_index, _), capacity in zip(depot_routes, capacities):
            vehicle_type_by_route[route_index] = len(vehicle_type_by_route)
            model.add_vehicle_type(
                num_available=1,
                capacity=capacity,
                start_depot=locations[depot_index_by_id[depot_id]],
                end_depot=locations[depot_index_by_id[depot_id]],
                fixed_cost=int(round(instance.route_fixed_cost * DISTANCE_SCALE)),
                name=f'depot_{depot_id}_route_{route_index}',
            )

    coordinates = [
        (float(depot.x), float(depot.y)) for depot in instance.depots
    ] + [
        (float(customer.x), float(customer.y)) for customer in instance.customers
    ]
    matrix = instance.distance_matrix
    for from_index, origin in enumerate(locations):
        for to_index, destination in enumerate(locations):
            if from_index == to_index:
                continue
            if matrix is not None:
                arc_cost = int(round(float(matrix[from_index, to_index]) * DISTANCE_SCALE))
            else:
                x1, y1 = coordinates[from_index]
                x2, y2 = coordinates[to_index]
                arc_cost = int(round(math.hypot(x1 - x2, y1 - y2) * DISTANCE_SCALE))
            model.add_edge(origin, destination, distance=arc_cost)

    data = model.data()
    initial = pyvrp.Solution(
        data,
        [
            pyvrp.Route(
                data,
                [instance.node_index[('customer', customer_id)] for customer_id in route.customer_ids],
                vehicle_type_by_route[route_index],
            )
            for route_index, route in enumerate(seed.routes)
        ],
    )
    node_ops = None
    route_ops = None
    if args.extended_operators:
        node_ops = [
            Exchange10,
            Exchange20,
            Exchange30,
            Exchange11,
            Exchange21,
            Exchange31,
            Exchange22,
            Exchange32,
            Exchange33,
            SwapTails,
            RelocateWithDepot,
        ]
        route_ops = [SwapRoutes, SwapStar]
    solve_kwargs = {}
    if node_ops is not None:
        solve_kwargs['node_ops'] = node_ops
        solve_kwargs['route_ops'] = route_ops
    result = model.solve(
        pyvrp.stop.MaxRuntime(max(1.0, args.time_limit)),
        seed=args.seed,
        display=not args.quiet,
        initial_solution=initial,
        params=SolveParams(
            ils=IteratedLocalSearchParams(
                num_iters_no_improvement=max(1, args.restart_iterations),
                history_length=max(1, args.history_length),
                exhaustive_on_best=True,
            ),
            neighbourhood=NeighbourhoodParams(
                num_neighbours=max(1, args.num_neighbours),
            ),
            perturbation=PerturbationParams(
                min_perturbations=max(0, args.min_perturbations),
                max_perturbations=max(args.min_perturbations, args.max_perturbations),
            ),
            **solve_kwargs,
        ),
    )
    if not result.is_feasible():
        raise RuntimeError('PyVRP did not retain a partition-feasible solution')

    depot_count = len(instance.depots)
    routes = [
        Route(
            instance.depots[route.start_depot()].id,
            [instance.customers[index - depot_count].id for index in route.visits()],
        )
        for route in result.best.routes()
    ]
    solution = Solution(instance.name, routes)
    final_validation = validate_solution(instance, solution)
    if not final_validation.is_feasible:
        raise RuntimeError('Partitioned PyVRP output is infeasible: ' + '; '.join(final_validation.errors))
    write_solution(solution, Path(args.output), instance=instance)
    initial_cost = objective_cost(instance, seed)
    final_cost = objective_cost(instance, solution)
    print(f'solution: {args.output}')
    print('feasible: true')
    print(f'initial_cost: {initial_cost:.10f}')
    print(f'cost: {final_cost:.10f}')
    print(f'improvement: {initial_cost - final_cost:.10f}')
    print(f'routes: {len(solution.routes)}')
    print(f'depots_opened: {len(solution.opened_depot_ids)}')
    print(f'iterations: {result.num_iterations}')
    print(f'runtime_seconds: {result.runtime:.3f}')
    for depot_id, capacities in sorted(capacity_summary.items()):
        print(f'partition_{depot_id}: {','.join(map(str, capacities))}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
