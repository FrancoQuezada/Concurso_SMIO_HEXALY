from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from smio_clrp.algorithms.common import route_load
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


def main() -> int:
    parser = argparse.ArgumentParser(description='Set-partitioning MIP over a pool of CLRP routes')
    parser.add_argument('instance')
    parser.add_argument('incumbent')
    parser.add_argument('--solutions-dir', default='solutions')
    parser.add_argument('--pattern', default='clrp-medium-01*.sol')
    parser.add_argument('--output', required=True)
    parser.add_argument('--time-limit', type=float, default=180.0)
    args = parser.parse_args()

    instance = read_instance(args.instance)
    incumbent = read_solution(args.incumbent)
    incumbent_validation = validate_solution(instance, incumbent)
    if not incumbent_validation.is_feasible:
        raise ValueError('Incumbent is infeasible: ' + '; '.join(incumbent_validation.errors))

    valid_customer_ids = set(instance.customers_by_id)
    route_by_key: dict[tuple[int, tuple[int, ...]], Route] = {}
    parsed_files = 0
    skipped_files = 0
    for path in sorted(Path(args.solutions_dir).glob(args.pattern)):
        try:
            candidate = read_solution(path)
        except Exception:
            skipped_files += 1
            continue
        parsed_files += 1
        for route in candidate.routes:
            if (
                route.depot_id not in instance.depots_by_id
                or not route.customer_ids
                or not set(route.customer_ids) <= valid_customer_ids
                or len(route.customer_ids) != len(set(route.customer_ids))
                or route_load(instance, route) > instance.vehicle_capacity + 1e-9
            ):
                continue
            key = (route.depot_id, tuple(route.customer_ids))
            reverse_key = (route.depot_id, tuple(reversed(route.customer_ids)))
            canonical = min(key, reverse_key)
            if canonical not in route_by_key:
                route_by_key[canonical] = route

    for route in incumbent.routes:
        key = (route.depot_id, tuple(route.customer_ids))
        reverse_key = (route.depot_id, tuple(reversed(route.customer_ids)))
        route_by_key.setdefault(min(key, reverse_key), route)
    routes = list(route_by_key.values())
    incumbent_keys = {
        min(
            (route.depot_id, tuple(route.customer_ids)),
            (route.depot_id, tuple(reversed(route.customer_ids))),
        )
        for route in incumbent.routes
    }

    routes_by_customer: dict[int, list[int]] = defaultdict(list)
    routes_by_depot: dict[int, list[int]] = defaultdict(list)
    loads: list[float] = []
    costs: list[float] = []
    for route_index, route in enumerate(routes):
        loads.append(route_load(instance, route))
        costs.append(instance.route_fixed_cost + route_distance(instance, route))
        routes_by_depot[route.depot_id].append(route_index)
        for customer_id in route.customer_ids:
            routes_by_customer[customer_id].append(route_index)
    missing = sorted(valid_customer_ids - routes_by_customer.keys())
    if missing:
        raise RuntimeError(f'Route pool does not cover customers: {missing[:20]}')

    model = gp.Model('clrp_route_pool')
    model.Params.TimeLimit = max(1.0, args.time_limit)
    model.Params.MIPGap = 0.0
    x = [model.addVar(vtype=GRB.BINARY, name=f'x_{index}') for index in range(len(routes))]
    y = {
        depot.id: model.addVar(vtype=GRB.BINARY, name=f'y_{depot.id}')
        for depot in instance.depots
    }
    for customer_id in sorted(valid_customer_ids):
        model.addConstr(
            gp.quicksum(x[index] for index in routes_by_customer[customer_id]) == 1,
            name=f'cover_{customer_id}',
        )
    for depot in instance.depots:
        indices = routes_by_depot[depot.id]
        model.addConstr(
            gp.quicksum(loads[index] * x[index] for index in indices)
            <= depot.capacity * y[depot.id],
            name=f'capacity_{depot.id}',
        )
        model.addConstr(
            gp.quicksum(x[index] for index in indices)
            <= depot.vehicle_limit * y[depot.id],
            name=f'vehicles_{depot.id}',
        )
    model.setObjective(
        gp.quicksum(costs[index] * x[index] for index in range(len(routes)))
        + gp.quicksum(depot.opening_cost * y[depot.id] for depot in instance.depots),
        GRB.MINIMIZE,
    )
    for index, route in enumerate(routes):
        key = min(
            (route.depot_id, tuple(route.customer_ids)),
            (route.depot_id, tuple(reversed(route.customer_ids))),
        )
        x[index].Start = 1.0 if key in incumbent_keys else 0.0
    for depot_id, variable in y.items():
        variable.Start = 1.0 if depot_id in incumbent.opened_depot_ids else 0.0

    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError('Route-pool MIP found no feasible solution')
    selected = [routes[index] for index, variable in enumerate(x) if variable.X > 0.5]
    solution = Solution(instance.name, selected)
    validation = validate_solution(instance, solution)
    if not validation.is_feasible:
        raise RuntimeError('Route-pool output is infeasible: ' + '; '.join(validation.errors))
    write_solution(solution, Path(args.output), instance=instance)
    initial_cost = objective_cost(instance, incumbent)
    final_cost = objective_cost(instance, solution)
    print(f'solution: {args.output}')
    print('feasible: true')
    print(f'initial_cost: {initial_cost:.10f}')
    print(f'cost: {final_cost:.10f}')
    print(f'improvement: {initial_cost - final_cost:.10f}')
    print(f'parsed_files: {parsed_files}')
    print(f'skipped_files: {skipped_files}')
    print(f'pool_routes: {len(routes)}')
    print(f'selected_routes: {len(selected)}')
    print(f'depots_opened: {len(solution.opened_depot_ids)}')
    print(f'mip_gap: {model.MIPGap:.8f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
