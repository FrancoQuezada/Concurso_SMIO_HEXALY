from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from smio_clrp.algorithms.common import route_load
from smio_clrp.core.distance import distance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


EPS = 1e-9


def _remove_delta(instance, route: Route, position: int) -> float:
    customer_id = route.customer_ids[position]
    depot = ('depot', route.depot_id)
    customer = ('customer', customer_id)
    previous = depot if position == 0 else ('customer', route.customer_ids[position - 1])
    following = depot if position + 1 == len(route.customer_ids) else (
        'customer', route.customer_ids[position + 1]
    )
    delta = (
        distance(instance, previous, following)
        - distance(instance, previous, customer)
        - distance(instance, customer, following)
    )
    if len(route.customer_ids) == 1:
        delta -= instance.route_fixed_cost
    return delta


def _best_insertion(instance, route: Route, customer_id: int) -> tuple[float, int]:
    depot = ('depot', route.depot_id)
    customer = ('customer', customer_id)
    best = (float('inf'), 0)
    for position in range(len(route.customer_ids) + 1):
        previous = depot if position == 0 else ('customer', route.customer_ids[position - 1])
        following = depot if position == len(route.customer_ids) else (
            'customer', route.customer_ids[position]
        )
        delta = (
            distance(instance, previous, customer)
            + distance(instance, customer, following)
            - distance(instance, previous, following)
        )
        best = min(best, (delta, position))
    return best


def _nearest_neighbor(instance, depot_id: int, customer_ids: list[int]) -> list[int]:
    remaining = set(customer_ids)
    result: list[int] = []
    current = ('depot', depot_id)
    while remaining:
        selected = min(
            remaining,
            key=lambda customer_id: (
                distance(instance, current, ('customer', customer_id)),
                customer_id,
            ),
        )
        result.append(selected)
        remaining.remove(selected)
        current = ('customer', selected)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Repair aggregate depot capacities with customer-level transfers'
    )
    parser.add_argument('instance')
    parser.add_argument('seed_solution')
    parser.add_argument('--output', required=True)
    parser.add_argument('--time-limit', type=float, default=120.0)
    parser.add_argument('--routes-per-depot', type=int, default=4)
    args = parser.parse_args()

    instance = read_instance(args.instance)
    seed = read_solution(args.seed_solution)
    validation = validate_solution(instance, seed)
    only_depot_capacity = bool(validation.errors) and all(
        error.startswith('Depot ') and ' assigned demand ' in error and ' exceeds capacity ' in error
        for error in validation.errors
    )
    if validation.is_feasible:
        write_solution(seed, Path(args.output), instance=instance)
        print(f'solution: {args.output}')
        print('feasible: true')
        print(f'cost: {objective_cost(instance, seed):.10f}')
        print('customers_moved: 0')
        return 0
    if not only_depot_capacity:
        raise ValueError('Seed has non-depot-capacity errors: ' + '; '.join(validation.errors))

    routes = seed.routes
    route_loads = [route_load(instance, route) for route in routes]
    depot_loads: dict[int, float] = defaultdict(float)
    routes_by_depot: dict[int, list[int]] = defaultdict(list)
    for route_index, route in enumerate(routes):
        depot_loads[route.depot_id] += route_loads[route_index]
        routes_by_depot[route.depot_id].append(route_index)

    overloaded = {
        depot_id: load - instance.depots_by_id[depot_id].capacity
        for depot_id, load in depot_loads.items()
        if load > instance.depots_by_id[depot_id].capacity + EPS
    }
    target_depots = {
        depot_id: instance.depots_by_id[depot_id].capacity - depot_loads.get(depot_id, 0.0)
        for depot_id in seed.opened_depot_ids
        if depot_loads.get(depot_id, 0.0) < instance.depots_by_id[depot_id].capacity - EPS
    }
    movable: list[tuple[int, int, int, float]] = []
    for route_index, route in enumerate(routes):
        if route.depot_id not in overloaded:
            continue
        for position, customer_id in enumerate(route.customer_ids):
            demand = instance.customers_by_id[customer_id].demand
            movable.append((customer_id, route_index, position, demand))

    # A candidate is (customer, target route). Keeping only the cheapest few
    # routes per target depot makes the MIP compact while retaining geography.
    insertion: dict[tuple[int, int], tuple[float, int]] = {}
    source_by_customer: dict[int, tuple[int, int, float]] = {}
    for customer_id, source_route, source_position, demand in movable:
        source_by_customer[customer_id] = (source_route, source_position, demand)
        for depot_id in target_depots:
            options: list[tuple[float, int, int]] = []
            for target_route in routes_by_depot[depot_id]:
                if route_loads[target_route] + demand > instance.vehicle_capacity + EPS:
                    continue
                delta, position = _best_insertion(instance, routes[target_route], customer_id)
                options.append((delta, target_route, position))
            for delta, target_route, position in sorted(options)[: max(1, args.routes_per_depot)]:
                insertion[customer_id, target_route] = (delta, position)

    model = gp.Model('customer_depot_capacity_repair')
    model.Params.TimeLimit = max(1.0, args.time_limit)
    model.Params.MIPGap = 0.0
    x = {
        key: model.addVar(vtype=GRB.BINARY, name=f'x_{key[0]}_{key[1]}')
        for key in insertion
    }
    for customer_id in source_by_customer:
        variables = [variable for (candidate, _), variable in x.items() if candidate == customer_id]
        if variables:
            model.addConstr(gp.quicksum(variables) <= 1, name=f'move_once_{customer_id}')
    for target_route in range(len(routes)):
        variables = [variable for (_, candidate), variable in x.items() if candidate == target_route]
        if variables:
            # Prevent interacting insertion deltas; subsequent repair rounds can
            # place additional customers in the same route if needed.
            model.addConstr(gp.quicksum(variables) <= 1, name=f'one_insert_{target_route}')
            model.addConstr(
                gp.quicksum(
                    source_by_customer[customer_id][2] * variable
                    for (customer_id, candidate), variable in x.items()
                    if candidate == target_route
                )
                <= instance.vehicle_capacity - route_loads[target_route],
                name=f'route_capacity_{target_route}',
            )
    for source_depot, overload in overloaded.items():
        model.addConstr(
            gp.quicksum(
                source_by_customer[customer_id][2] * variable
                for (customer_id, _), variable in x.items()
                if routes[source_by_customer[customer_id][0]].depot_id == source_depot
            )
            >= overload,
            name=f'relieve_{source_depot}',
        )
    for target_depot, slack in target_depots.items():
        model.addConstr(
            gp.quicksum(
                source_by_customer[customer_id][2] * variable
                for (customer_id, target_route), variable in x.items()
                if routes[target_route].depot_id == target_depot
            )
            <= slack,
            name=f'depot_capacity_{target_depot}',
        )

    model.setObjective(
        gp.quicksum(
            (
                _remove_delta(
                    instance,
                    routes[source_by_customer[customer_id][0]],
                    source_by_customer[customer_id][1],
                )
                + insertion[customer_id, target_route][0]
                + 1e-4 * source_by_customer[customer_id][2]
            )
            * variable
            for (customer_id, target_route), variable in x.items()
        ),
        GRB.MINIMIZE,
    )
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError('No repair exists with the generated insertion candidates')

    selected = {
        customer_id: target_route
        for (customer_id, target_route), variable in x.items()
        if variable.X > 0.5
    }
    removed = set(selected)
    repaired_routes: list[Route] = []
    for route_index, route in enumerate(routes):
        customer_ids = [customer_id for customer_id in route.customer_ids if customer_id not in removed]
        arriving = [customer_id for customer_id, target in selected.items() if target == route_index]
        if arriving:
            customer_id = arriving[0]
            position = insertion[customer_id, route_index][1]
            customer_ids.insert(position, customer_id)
        if customer_ids:
            repaired_routes.append(Route(route.depot_id, customer_ids))

    solution = Solution(instance.name, repaired_routes)
    final_validation = validate_solution(instance, solution)
    if not final_validation.is_feasible:
        raise RuntimeError('Customer repair is infeasible: ' + '; '.join(final_validation.errors))
    write_solution(solution, Path(args.output), instance=instance)
    print(f'solution: {args.output}')
    print('feasible: true')
    print(f'initial_cost: {objective_cost(instance, seed):.10f}')
    print(f'cost: {objective_cost(instance, solution):.10f}')
    print(f'customers_moved: {len(selected)}')
    print(f'routes: {len(solution.routes)}')
    print(f'depots_opened: {len(solution.opened_depot_ids)}')
    print(f'mip_gap: {model.MIPGap:.8f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
