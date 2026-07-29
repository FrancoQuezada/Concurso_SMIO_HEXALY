from __future__ import annotations

import argparse
import math
from pathlib import Path

import gurobipy as gp
from gurobipy import GRB

from smio_clrp.core.distance import distance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


def _pack_routes(instance, depot_id: int, customer_ids: list[int]) -> list[Route]:
    # Best-fit decreasing obtains a small feasible fleet. The angle tie-break keeps
    # the initial bins somewhat geographical before OR-Tools intensification.
    depot = instance.depots_by_id[depot_id]
    ordered = sorted(
        customer_ids,
        key=lambda customer_id: (
            -instance.customers_by_id[customer_id].demand,
            math.atan2(
                float(instance.customers_by_id[customer_id].y) - float(depot.y),
                float(instance.customers_by_id[customer_id].x) - float(depot.x),
            ),
            customer_id,
        ),
    )
    bins: list[list[int]] = []
    loads: list[float] = []
    for customer_id in ordered:
        demand = instance.customers_by_id[customer_id].demand
        feasible = [i for i, load in enumerate(loads) if load + demand <= instance.vehicle_capacity + 1e-9]
        if feasible:
            index = min(feasible, key=lambda i: (instance.vehicle_capacity - loads[i] - demand, i))
            bins[index].append(customer_id)
            loads[index] += demand
        else:
            bins.append([customer_id])
            loads.append(demand)
    if len(bins) > depot.vehicle_limit:
        raise RuntimeError(
            f"Depot {depot_id} needs {len(bins)} FFD routes but its limit is {depot.vehicle_limit}"
        )
    return [Route(depot_id, _nearest_neighbor(instance, depot_id, group)) for group in bins]


def _nearest_neighbor(instance, depot_id: int, customer_ids: list[int]) -> list[int]:
    remaining = set(customer_ids)
    result: list[int] = []
    current = ("depot", depot_id)
    while remaining:
        selected = min(
            remaining,
            key=lambda customer_id: (distance(instance, current, ("customer", customer_id)), customer_id),
        )
        result.append(selected)
        remaining.remove(selected)
        current = ("customer", selected)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Capacitated MIP customer-to-depot assignment")
    parser.add_argument("instance")
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates", type=int, default=8)
    parser.add_argument("--time-limit", type=float, default=120.0)
    parser.add_argument("--warm-solution")
    args = parser.parse_args()

    instance = read_instance(args.instance)
    customers = instance.customers
    depots = instance.depots
    candidate_count = min(len(depots), max(1, args.candidates))
    warm_assignment: dict[int, int] = {}
    if args.warm_solution:
        warm = read_solution(args.warm_solution)
        validation = validate_solution(instance, warm)
        if not validation.is_feasible:
            raise ValueError("Warm solution is infeasible: " + "; ".join(validation.errors))
        warm_assignment = {
            customer_id: route.depot_id for route in warm.routes for customer_id in route.customer_ids
        }

    proxy: dict[tuple[int, int], float] = {}
    candidates_by_customer: dict[int, list[int]] = {}
    for customer in customers:
        costs = sorted(
            (
                distance(instance, ("depot", depot.id), ("customer", customer.id))
                + distance(instance, ("customer", customer.id), ("depot", depot.id)),
                depot.id,
            )
            for depot in depots
        )
        candidate_ids = [depot_id for _, depot_id in costs[:candidate_count]]
        if customer.id in warm_assignment and warm_assignment[customer.id] not in candidate_ids:
            candidate_ids.append(warm_assignment[customer.id])
        candidates_by_customer[customer.id] = candidate_ids
        for cost, depot_id in costs:
            if depot_id in candidate_ids:
                proxy[customer.id, depot_id] = cost

    model = gp.Model("customer_depot_assignment")
    model.Params.TimeLimit = max(1.0, args.time_limit)
    model.Params.MIPGap = 0.001
    x = {
        (customer.id, depot_id): model.addVar(vtype=GRB.BINARY, name=f"x_{customer.id}_{depot_id}")
        for customer in customers
        for depot_id in candidates_by_customer[customer.id]
    }
    y = {depot.id: model.addVar(vtype=GRB.BINARY, name=f"y_{depot.id}") for depot in depots}
    for customer in customers:
        model.addConstr(
            gp.quicksum(x[customer.id, depot_id] for depot_id in candidates_by_customer[customer.id]) == 1,
            name=f"assign_{customer.id}",
        )
    for depot in depots:
        assigned = [customer for customer in customers if (customer.id, depot.id) in x]
        model.addConstr(
            gp.quicksum(customer.demand * x[customer.id, depot.id] for customer in assigned)
            <= depot.capacity * y[depot.id],
            name=f"capacity_{depot.id}",
        )
    model.setObjective(
        gp.quicksum(proxy[key] * variable for key, variable in x.items())
        + gp.quicksum(depot.opening_cost * y[depot.id] for depot in depots),
        GRB.MINIMIZE,
    )
    for (customer_id, depot_id), variable in x.items():
        variable.Start = 1.0 if warm_assignment.get(customer_id) == depot_id else 0.0
    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError("Gurobi did not find a feasible customer assignment")

    assigned_by_depot: dict[int, list[int]] = {}
    for customer in customers:
        depot_id = next(
            depot_id for depot_id in candidates_by_customer[customer.id]
            if x[customer.id, depot_id].X > 0.5
        )
        assigned_by_depot.setdefault(depot_id, []).append(customer.id)

    routes = [
        route
        for depot_id, customer_ids in sorted(assigned_by_depot.items())
        for route in _pack_routes(instance, depot_id, customer_ids)
    ]
    solution = Solution(instance.name, routes)
    final_validation = validate_solution(instance, solution)
    if not final_validation.is_feasible:
        raise RuntimeError("Assigned solution is infeasible: " + "; ".join(final_validation.errors))
    write_solution(solution, Path(args.output), instance=instance)
    print(f"solution: {args.output}")
    print("feasible: true")
    print(f"cost: {objective_cost(instance, solution):.10f}")
    print(f"routes: {len(routes)}")
    print(f"depots_opened: {len(solution.opened_depot_ids)}")
    print(f"mip_gap: {model.MIPGap:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
