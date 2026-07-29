from __future__ import annotations

import argparse
import itertools
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


SCALE = 10


def _centroid(instance, route: Route) -> tuple[float, float]:
    nodes = [instance.customers_by_id[item] for item in route.customer_ids]
    return (
        sum(float(node.x) for node in nodes) / len(nodes),
        sum(float(node.y) for node in nodes) / len(nodes),
    )


def _solve(instance, routes: list[Route], capacities: list[int], seconds: float, seed: int):
    customer_ids = list(dict.fromkeys(item for route in routes for item in route.customer_ids))
    model = pyvrp.Model()
    locations = [
        model.add_depot(
            float(instance.depots_by_id[route.depot_id].x),
            float(instance.depots_by_id[route.depot_id].y),
            name=f'd{route.depot_id}',
        )
        for route in routes
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
    for index, capacity in enumerate(capacities):
        model.add_vehicle_type(
            num_available=1,
            capacity=capacity,
            start_depot=locations[index],
            end_depot=locations[index],
            fixed_cost=int(round(instance.route_fixed_cost * SCALE)),
            name=f'v{index}',
        )
    keys = [('depot', route.depot_id) for route in routes] + [
        ('customer', item) for item in customer_ids
    ]
    for from_index, origin in enumerate(locations):
        for to_index, destination in enumerate(locations):
            if from_index != to_index:
                model.add_edge(
                    origin,
                    destination,
                    distance=int(round(distance(instance, keys[from_index], keys[to_index]) * SCALE)),
                )
    data = model.data()
    offset = len(routes)
    local = {customer_id: index + offset for index, customer_id in enumerate(customer_ids)}
    initial = pyvrp.Solution(
        data,
        [
            pyvrp.Route(data, [local[item] for item in route.customer_ids], index)
            for index, route in enumerate(routes)
        ],
    )
    result = model.solve(
        pyvrp.stop.MaxRuntime(max(0.01, seconds)),
        seed=seed,
        display=False,
        collect_stats=False,
        initial_solution=initial,
    )
    if not result.is_feasible():
        return None
    depot_ids = [route.depot_id for route in routes]
    replacement = [
        Route(
            depot_ids[route.start_depot()],
            [customer_ids[index - offset] for index in route.visits()],
        )
        for route in result.best.routes()
    ]
    return replacement


def main() -> int:
    parser = argparse.ArgumentParser(description='PyVRP intensification of route triples')
    parser.add_argument('instance')
    parser.add_argument('seed_solution')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds-per-triple', type=float, default=0.1)
    parser.add_argument('--neighbors-per-route', type=int, default=6)
    parser.add_argument('--passes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    instance = read_instance(args.instance)
    current = read_solution(args.seed_solution)
    validation = validate_solution(instance, current)
    if not validation.is_feasible:
        raise ValueError('Seed is infeasible: ' + '; '.join(validation.errors))
    initial_cost = objective_cost(instance, current)
    evaluated = 0
    accepted = 0
    for pass_index in range(max(1, args.passes)):
        routes = current.routes
        centroids = [_centroid(instance, route) for route in routes]
        triples: set[tuple[int, int, int]] = set()
        for anchor, route in enumerate(routes):
            nearby = [
                index
                for _, index in sorted(
                    (
                        math.hypot(
                            centroids[anchor][0] - centroids[index][0],
                            centroids[anchor][1] - centroids[index][1],
                        ),
                        index,
                    )
                    for index, other in enumerate(routes)
                    if index != anchor and other.depot_id != route.depot_id
                )[: max(2, args.neighbors_per_route)]
            ]
            for second, third in itertools.combinations(nearby, 2):
                depot_ids = {route.depot_id, routes[second].depot_id, routes[third].depot_id}
                if len(depot_ids) == 3:
                    triples.add(tuple(sorted((anchor, second, third))))

        depot_loads: dict[int, float] = {}
        for route in routes:
            depot_loads[route.depot_id] = depot_loads.get(route.depot_id, 0.0) + route_load(instance, route)
        base_cost = objective_cost(instance, current)
        best = current
        best_cost = base_cost
        for number, indices in enumerate(sorted(triples)):
            selected = [routes[index] for index in indices]
            capacities = [
                int(round(min(
                    instance.vehicle_capacity,
                    route_load(instance, route)
                    + instance.depots_by_id[route.depot_id].capacity
                    - depot_loads[route.depot_id],
                )))
                for route in selected
            ]
            replacement = _solve(
                instance,
                selected,
                capacities,
                args.seconds_per_triple,
                args.seed + pass_index * 100000 + number,
            )
            evaluated += 1
            if replacement is None:
                continue
            candidate = Solution(
                instance.name,
                [route for index, route in enumerate(routes) if index not in indices] + replacement,
            )
            candidate_validation = validate_solution(instance, candidate)
            if candidate_validation.is_feasible and candidate_validation.cost < best_cost - 1e-9:
                best = candidate
                best_cost = candidate_validation.cost
        if best_cost >= base_cost - 1e-9:
            break
        current = best
        accepted += 1
        print(f'pass={pass_index + 1} cost={best_cost:.10f}', flush=True)

    final_validation = validate_solution(instance, current)
    if not final_validation.is_feasible:
        raise RuntimeError('Triple output is infeasible: ' + '; '.join(final_validation.errors))
    write_solution(current, Path(args.output), instance=instance)
    print(f'solution: {args.output}')
    print('feasible: true')
    print(f'initial_cost: {initial_cost:.10f}')
    print(f'cost: {final_validation.cost:.10f}')
    print(f'improvement: {initial_cost - final_validation.cost:.10f}')
    print(f'evaluated_triples: {evaluated}')
    print(f'accepted_passes: {accepted}')
    print(f'routes: {len(current.routes)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
