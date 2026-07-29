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


def _remove_delta(instance, route: Route, position: int) -> float:
    depot = ('depot', route.depot_id)
    customer = ('customer', route.customer_ids[position])
    previous = depot if position == 0 else ('customer', route.customer_ids[position - 1])
    following = depot if position + 1 == len(route.customer_ids) else (
        'customer', route.customer_ids[position + 1]
    )
    return (
        distance(instance, previous, following)
        - distance(instance, previous, customer)
        - distance(instance, customer, following)
    )


def _insertion_delta(instance, route: Route, customer_id: int) -> float:
    depot = ('depot', route.depot_id)
    customer = ('customer', customer_id)
    best = float('inf')
    for position in range(len(route.customer_ids) + 1):
        previous = depot if position == 0 else ('customer', route.customer_ids[position - 1])
        following = depot if position == len(route.customer_ids) else (
            'customer', route.customer_ids[position]
        )
        best = min(
            best,
            distance(instance, previous, customer)
            + distance(instance, customer, following)
            - distance(instance, previous, following),
        )
    return best


def _insert_best(instance, depot_id: int, route: list[int], customer_id: int) -> None:
    probe = Route(depot_id, route)
    depot = ('depot', depot_id)
    customer = ('customer', customer_id)
    best: tuple[float, int] | None = None
    for position in range(len(route) + 1):
        previous = depot if position == 0 else ('customer', route[position - 1])
        following = depot if position == len(route) else ('customer', route[position])
        delta = (
            distance(instance, previous, customer)
            + distance(instance, customer, following)
            - distance(instance, previous, following)
        )
        candidate = (delta, position)
        if best is None or candidate < best:
            best = candidate
    assert best is not None
    route.insert(best[1], customer_id)


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Joint customer-to-route reassignment for depot-capacity repair'
    )
    parser.add_argument('instance')
    parser.add_argument('seed_solution')
    parser.add_argument('--output', required=True)
    parser.add_argument('--time-limit', type=float, default=180.0)
    parser.add_argument('--routes-per-depot', type=int, default=5)
    parser.add_argument('--move-penalty', type=float, default=0.0)
    args = parser.parse_args()

    instance = read_instance(args.instance)
    seed = read_solution(args.seed_solution)
    validation = validate_solution(instance, seed)
    only_depot_capacity = bool(validation.errors) and all(
        error.startswith('Depot ') and ' assigned demand ' in error and ' exceeds capacity ' in error
        for error in validation.errors
    )
    if not validation.is_feasible and not only_depot_capacity:
        raise ValueError('Seed has non-depot-capacity errors: ' + '; '.join(validation.errors))

    routes = seed.routes
    opened = sorted(seed.opened_depot_ids)
    routes_by_depot: dict[int, list[int]] = defaultdict(list)
    source: dict[int, tuple[int, int]] = {}
    for route_index, route in enumerate(routes):
        routes_by_depot[route.depot_id].append(route_index)
        for position, customer_id in enumerate(route.customer_ids):
            source[customer_id] = (route_index, position)

    # Each bin is a route tied to a depot. Existing bins preserve route count;
    # optional empty bins let capacity migrate between depots when fleet permits.
    bin_depot = [route.depot_id for route in routes]
    existing_count = len(bin_depot)
    for depot_id in opened:
        free_slots = instance.depots_by_id[depot_id].vehicle_limit - len(routes_by_depot[depot_id])
        bin_depot.extend([depot_id] * max(0, free_slots))
    bins_by_depot: dict[int, list[int]] = defaultdict(list)
    for bin_index, depot_id in enumerate(bin_depot):
        bins_by_depot[depot_id].append(bin_index)

    candidates: dict[int, list[int]] = {}
    assignment_cost: dict[tuple[int, int], float] = {}
    for customer in instance.customers:
        source_route, source_position = source[customer.id]
        remove = _remove_delta(instance, routes[source_route], source_position)
        selected: set[int] = {source_route}
        for depot_id in opened:
            scored: list[tuple[float, int]] = []
            for bin_index in bins_by_depot[depot_id]:
                if bin_index < existing_count:
                    insert = _insertion_delta(instance, routes[bin_index], customer.id)
                else:
                    depot = ('depot', depot_id)
                    node = ('customer', customer.id)
                    insert = distance(instance, depot, node) + distance(instance, node, depot)
                scored.append((insert, bin_index))
            selected.update(
                bin_index for _, bin_index in sorted(scored)[: max(1, args.routes_per_depot)]
            )
        candidates[customer.id] = sorted(selected)
        for bin_index in candidates[customer.id]:
            if bin_index == source_route:
                assignment_cost[customer.id, bin_index] = 0.0
            elif bin_index < existing_count:
                assignment_cost[customer.id, bin_index] = (
                    remove
                    + _insertion_delta(instance, routes[bin_index], customer.id)
                    + args.move_penalty
                )
            else:
                depot = ('depot', bin_depot[bin_index])
                node = ('customer', customer.id)
                assignment_cost[customer.id, bin_index] = (
                    remove
                    + distance(instance, depot, node)
                    + distance(instance, node, depot)
                    + args.move_penalty
                )

    model = gp.Model('customer_route_reassignment')
    model.Params.TimeLimit = max(1.0, args.time_limit)
    model.Params.MIPGap = 0.001
    x = {
        (customer_id, bin_index): model.addVar(
            vtype=GRB.BINARY, name=f'x_{customer_id}_{bin_index}'
        )
        for customer_id, bin_indices in candidates.items()
        for bin_index in bin_indices
    }
    z = {
        bin_index: model.addVar(vtype=GRB.BINARY, name=f'z_{bin_index}')
        for bin_index in range(len(bin_depot))
    }
    for customer in instance.customers:
        model.addConstr(
            gp.quicksum(x[customer.id, bin_index] for bin_index in candidates[customer.id]) == 1,
            name=f'assign_{customer.id}',
        )
    for bin_index in range(len(bin_depot)):
        eligible = [customer for customer in instance.customers if (customer.id, bin_index) in x]
        model.addConstr(
            gp.quicksum(customer.demand * x[customer.id, bin_index] for customer in eligible)
            <= instance.vehicle_capacity * z[bin_index],
            name=f'route_capacity_{bin_index}',
        )
    for depot_id in opened:
        model.addConstr(
            gp.quicksum(
                customer.demand * x[customer.id, bin_index]
                for customer in instance.customers
                for bin_index in bins_by_depot[depot_id]
                if (customer.id, bin_index) in x
            )
            <= instance.depots_by_id[depot_id].capacity,
            name=f'depot_capacity_{depot_id}',
        )
    model.setObjective(
        gp.quicksum(assignment_cost[key] * variable for key, variable in x.items())
        + instance.route_fixed_cost * gp.quicksum(
            z[bin_index] for bin_index in range(existing_count, len(bin_depot))
        ),
        GRB.MINIMIZE,
    )
    for (customer_id, bin_index), variable in x.items():
        variable.Start = 1.0 if source[customer_id][0] == bin_index else 0.0
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError('Gurobi found no feasible customer-to-route repair')

    assigned: dict[int, list[int]] = defaultdict(list)
    moved = 0
    for customer in instance.customers:
        chosen = next(
            bin_index for bin_index in candidates[customer.id]
            if x[customer.id, bin_index].X > 0.5
        )
        assigned[chosen].append(customer.id)
        moved += chosen != source[customer.id][0]

    result_routes: list[Route] = []
    for bin_index, customer_ids in sorted(assigned.items()):
        depot_id = bin_depot[bin_index]
        if bin_index < existing_count:
            assigned_set = set(customer_ids)
            ordered = [
                customer_id for customer_id in routes[bin_index].customer_ids
                if customer_id in assigned_set
            ]
            arriving = [
                customer_id for customer_id in customer_ids
                if source[customer_id][0] != bin_index
            ]
            for customer_id in sorted(
                arriving,
                key=lambda candidate: _insertion_delta(
                    instance, Route(depot_id, ordered), candidate
                ),
            ):
                _insert_best(instance, depot_id, ordered, customer_id)
        else:
            ordered = []
            for customer_id in sorted(customer_ids):
                _insert_best(instance, depot_id, ordered, customer_id)
        result_routes.append(Route(depot_id, ordered))

    solution = Solution(instance.name, result_routes)
    final_validation = validate_solution(instance, solution)
    if not final_validation.is_feasible:
        raise RuntimeError('Route-bin repair is infeasible: ' + '; '.join(final_validation.errors))
    write_solution(solution, Path(args.output), instance=instance)
    print(f'solution: {args.output}')
    print('feasible: true')
    print(f'initial_cost: {objective_cost(instance, seed):.10f}')
    print(f'cost: {objective_cost(instance, solution):.10f}')
    print(f'customers_moved: {moved}')
    print(f'routes: {len(solution.routes)}')
    print(f'depots_opened: {len(solution.opened_depot_ids)}')
    print(f'mip_gap: {model.MIPGap:.8f}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
