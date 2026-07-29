from __future__ import annotations

import argparse
import math
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
    RelocateWithDepot,
    SwapRoutes,
    SwapStar,
    SwapTails,
)
from pyvrp.solve import SolveParams

from smio_clrp.algorithms.common import route_load
from smio_clrp.core.distance import distance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


SCALE = 10


def _capacities(loads: list[int], total: int, vehicle_capacity: int) -> list[int]:
    result = list(loads)
    slack = min(total, len(loads) * vehicle_capacity) - sum(loads)
    while slack > 0:
        changed = False
        for index in range(len(result)):
            if result[index] < vehicle_capacity:
                result[index] += 1
                slack -= 1
                changed = True
                if slack == 0:
                    break
        if not changed:
            break
    return result


def _solve_pair(instance, routes: list[Route], seconds: float, seed: int) -> list[Route] | None:
    depot_ids = sorted({route.depot_id for route in routes})
    customer_ids = list(dict.fromkeys(item for route in routes for item in route.customer_ids))
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
    routes_by_depot: dict[int, list[tuple[int, Route]]] = defaultdict(list)
    for index, route in enumerate(routes):
        routes_by_depot[route.depot_id].append((index, route))
    type_by_route: dict[int, int] = {}
    for depot_index, depot_id in enumerate(depot_ids):
        selected = routes_by_depot[depot_id]
        caps = _capacities(
            [int(round(route_load(instance, route))) for _, route in selected],
            int(round(instance.depots_by_id[depot_id].capacity)),
            int(round(instance.vehicle_capacity)),
        )
        for (route_index, _), capacity in zip(selected, caps):
            type_by_route[route_index] = len(type_by_route)
            model.add_vehicle_type(
                num_available=1,
                capacity=capacity,
                start_depot=locations[depot_index],
                end_depot=locations[depot_index],
                fixed_cost=int(round(instance.route_fixed_cost * SCALE)),
                name=f'v{route_index}',
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
                    distance=int(round(distance(instance, keys[from_index], keys[to_index]) * SCALE)),
                )
    data = model.data()
    offset = len(depot_ids)
    local = {customer_id: index + offset for index, customer_id in enumerate(customer_ids)}
    initial = pyvrp.Solution(
        data,
        [
            pyvrp.Route(
                data,
                [local[item] for item in route.customer_ids],
                type_by_route[index],
            )
            for index, route in enumerate(routes)
        ],
    )
    result = model.solve(
        pyvrp.stop.MaxRuntime(max(0.1, seconds)),
        seed=seed,
        display=False,
        collect_stats=False,
        initial_solution=initial,
        params=SolveParams(
            node_ops=[
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
            ],
            route_ops=[SwapRoutes, SwapStar],
        ),
    )
    if not result.is_feasible():
        return None
    return [
        Route(
            depot_ids[route.start_depot()],
            [customer_ids[index - offset] for index in route.visits()],
        )
        for route in result.best.routes()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description='Extended PyVRP intensification of depot pairs')
    parser.add_argument('instance')
    parser.add_argument('seed_solution')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seconds-per-pair', type=float, default=5.0)
    parser.add_argument('--neighbors-per-depot', type=int, default=3)
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
        depot_ids = sorted(current.opened_depot_ids)
        pairs: set[tuple[int, int]] = set()
        for depot_id in depot_ids:
            nearest = sorted(
                (
                    distance(instance, ('depot', depot_id), ('depot', other)),
                    other,
                )
                for other in depot_ids
                if other != depot_id
            )[: max(1, args.neighbors_per_depot)]
            pairs.update(tuple(sorted((depot_id, other))) for _, other in nearest)
        base_cost = objective_cost(instance, current)
        best = current
        best_cost = base_cost
        for number, pair in enumerate(sorted(pairs)):
            selected = [route for route in current.routes if route.depot_id in pair]
            replacement = _solve_pair(
                instance,
                selected,
                args.seconds_per_pair,
                args.seed + pass_index * 10000 + number,
            )
            evaluated += 1
            if replacement is None:
                continue
            candidate = Solution(
                instance.name,
                [route for route in current.routes if route.depot_id not in pair] + replacement,
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
        raise RuntimeError('Depot-pair output is infeasible: ' + '; '.join(final_validation.errors))
    write_solution(current, Path(args.output), instance=instance)
    print(f'solution: {args.output}')
    print('feasible: true')
    print(f'initial_cost: {initial_cost:.10f}')
    print(f'cost: {final_validation.cost:.10f}')
    print(f'improvement: {initial_cost - final_validation.cost:.10f}')
    print(f'evaluated_pairs: {evaluated}')
    print(f'accepted_passes: {accepted}')
    print(f'routes: {len(current.routes)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
