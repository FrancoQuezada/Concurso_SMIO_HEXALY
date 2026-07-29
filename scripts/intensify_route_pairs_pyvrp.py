from __future__ import annotations

import argparse
import math
from pathlib import Path

import pyvrp

from smio_clrp.algorithms.common import route_load
from smio_clrp.core.distance import distance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


DISTANCE_SCALE = 10


def _centroid(instance, route: Route) -> tuple[float, float]:
    customers = [instance.customers_by_id[customer_id] for customer_id in route.customer_ids]
    return (
        sum(float(customer.x) for customer in customers) / len(customers),
        sum(float(customer.y) for customer in customers) / len(customers),
    )


def _solve_pair(
    instance,
    first: Route,
    second: Route,
    capacities: tuple[int, int],
    seconds: float,
    seed: int,
    use_initial: bool,
):
    depot_ids = [first.depot_id, second.depot_id]
    customer_ids = list(dict.fromkeys([*first.customer_ids, *second.customer_ids]))
    model = pyvrp.Model()
    locations = [
        model.add_depot(
            float(instance.depots_by_id[depot_id].x),
            float(instance.depots_by_id[depot_id].y),
            name=f'd{depot_id}',
        )
        for depot_id in depot_ids
    ]
    for customer_id in customer_ids:
        customer = instance.customers_by_id[customer_id]
        locations.append(
            model.add_client(
                float(customer.x),
                float(customer.y),
                delivery=int(round(customer.demand)),
                name=f'c{customer_id}',
            )
        )
    for vehicle_type, (depot_index, capacity) in enumerate(zip(range(2), capacities)):
        model.add_vehicle_type(
            num_available=1,
            capacity=capacity,
            start_depot=locations[depot_index],
            end_depot=locations[depot_index],
            fixed_cost=int(round(instance.route_fixed_cost * DISTANCE_SCALE)),
            name=f'v{vehicle_type}',
        )
    keys = [('depot', depot_id) for depot_id in depot_ids] + [
        ('customer', customer_id) for customer_id in customer_ids
    ]
    for from_index, origin in enumerate(locations):
        for to_index, destination in enumerate(locations):
            if from_index != to_index:
                model.add_edge(
                    origin,
                    destination,
                    distance=int(round(distance(instance, keys[from_index], keys[to_index]) * DISTANCE_SCALE)),
                )
    data = model.data()
    local_index = {customer_id: index + 2 for index, customer_id in enumerate(customer_ids)}
    initial = pyvrp.Solution(
        data,
        [
            pyvrp.Route(data, [local_index[item] for item in first.customer_ids], 0),
            pyvrp.Route(data, [local_index[item] for item in second.customer_ids], 1),
        ],
    )
    result = model.solve(
        pyvrp.stop.MaxRuntime(max(0.01, seconds)),
        seed=seed,
        display=False,
        collect_stats=False,
        initial_solution=initial if use_initial else None,
    )
    if not result.is_feasible():
        return None
    routes = [
        Route(
            depot_ids[route.start_depot()],
            [customer_ids[index - 2] for index in route.visits()],
        )
        for route in result.best.routes()
    ]
    old_cost = (
        2 * instance.route_fixed_cost
        + route_distance(instance, first)
        + route_distance(instance, second)
    )
    new_cost = len(routes) * instance.route_fixed_cost + sum(
        route_distance(instance, route) for route in routes
    )
    return routes, old_cost - new_cost


def main() -> int:
    parser = argparse.ArgumentParser(description='PyVRP intensification of nearby route pairs')
    parser.add_argument('instance')
    parser.add_argument('seed_solution')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds-per-pair', type=float, default=0.05)
    parser.add_argument('--neighbors-per-route', type=int, default=8)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--pool-output')
    parser.add_argument('--diverse-pool', action='store_true')
    parser.add_argument('--max-pair-loss', type=float, default=0.0)
    args = parser.parse_args()

    instance = read_instance(args.instance)
    current = read_solution(args.seed_solution)
    validation = validate_solution(instance, current)
    if not validation.is_feasible:
        raise ValueError('Seed is infeasible: ' + '; '.join(validation.errors))
    initial_cost = objective_cost(instance, current)
    evaluated = 0
    accepted = 0
    pool_routes: dict[tuple[int, tuple[int, ...]], Route] = {}

    for pass_index in range(max(1, args.passes)):
        routes = current.routes
        centroids = [_centroid(instance, route) for route in routes]
        pairs: set[tuple[int, int]] = set()
        for first_index, first in enumerate(routes):
            candidates = sorted(
                (
                    math.hypot(
                        centroids[first_index][0] - centroids[second_index][0],
                        centroids[first_index][1] - centroids[second_index][1],
                    ),
                    second_index,
                )
                for second_index, second in enumerate(routes)
                if second_index != first_index and second.depot_id != first.depot_id
            )
            for _, second_index in candidates[: max(1, args.neighbors_per_route)]:
                pairs.add(tuple(sorted((first_index, second_index))))

        depot_loads: dict[int, float] = {}
        for route in routes:
            depot_loads[route.depot_id] = depot_loads.get(route.depot_id, 0.0) + route_load(instance, route)
        best_solution = current
        best_cost = objective_cost(instance, current)
        for pair_number, (first_index, second_index) in enumerate(sorted(pairs)):
            first = routes[first_index]
            second = routes[second_index]
            capacities = (
                int(round(min(
                    instance.vehicle_capacity,
                    route_load(instance, first)
                    + instance.depots_by_id[first.depot_id].capacity
                    - depot_loads[first.depot_id],
                ))),
                int(round(min(
                    instance.vehicle_capacity,
                    route_load(instance, second)
                    + instance.depots_by_id[second.depot_id].capacity
                    - depot_loads[second.depot_id],
                ))),
            )
            result = _solve_pair(
                instance,
                first,
                second,
                capacities,
                args.seconds_per_pair,
                args.seed + pass_index * 100000 + pair_number,
                not args.diverse_pool,
            )
            evaluated += 1
            if result is None or result[1] < -max(0.0, args.max_pair_loss) - 1e-9:
                continue
            replacement, _ = result
            candidate_routes = [
                route for index, route in enumerate(routes)
                if index not in {first_index, second_index}
            ] + replacement
            candidate = Solution(instance.name, candidate_routes)
            candidate_validation = validate_solution(instance, candidate)
            if candidate_validation.is_feasible:
                for route in replacement:
                    key = min(
                        (route.depot_id, tuple(route.customer_ids)),
                        (route.depot_id, tuple(reversed(route.customer_ids))),
                    )
                    pool_routes.setdefault(key, route)
            if candidate_validation.is_feasible and candidate_validation.cost < best_cost - 1e-9:
                best_solution = candidate
                best_cost = candidate_validation.cost
        if best_cost >= objective_cost(instance, current) - 1e-9:
            break
        current = best_solution
        accepted += 1
        print(f'pass={pass_index + 1} cost={best_cost:.10f}', flush=True)

    final_validation = validate_solution(instance, current)
    if not final_validation.is_feasible:
        raise RuntimeError('Route-pair output is infeasible: ' + '; '.join(final_validation.errors))
    write_solution(current, Path(args.output), instance=instance)
    if args.pool_output:
        write_solution(
            Solution(instance.name, list(pool_routes.values())),
            Path(args.pool_output),
            instance=instance,
        )
    print(f'solution: {args.output}')
    print('feasible: true')
    print(f'initial_cost: {initial_cost:.10f}')
    print(f'cost: {final_validation.cost:.10f}')
    print(f'improvement: {initial_cost - final_validation.cost:.10f}')
    print(f'evaluated_pairs: {evaluated}')
    print(f'accepted_passes: {accepted}')
    print(f'routes: {len(current.routes)}')
    print(f'pool_routes: {len(pool_routes)}')
    if args.pool_output:
        print(f'pool_output: {args.pool_output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
