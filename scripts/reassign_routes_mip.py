from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Exact MIP reassignment of complete routes to depots")
    parser.add_argument("instance")
    parser.add_argument("seed_solution")
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--allow-overloaded-seed", action="store_true")
    args = parser.parse_args()

    instance = read_instance(args.instance)
    seed = read_solution(args.seed_solution)
    validation = validate_solution(instance, seed)
    if not validation.is_feasible:
        only_depot_capacity = all(
            error.startswith("Depot ") and " assigned demand " in error and " exceeds capacity " in error
            for error in validation.errors
        )
        if not args.allow_overloaded_seed or not only_depot_capacity:
            raise ValueError("Seed solution is infeasible: " + "; ".join(validation.errors))

    routes = seed.routes
    depots = instance.depots
    loads = [route_load(instance, route) for route in routes]
    depot_cost: dict[tuple[int, int], float] = {}
    for route_index, route in enumerate(routes):
        first = route.customer_ids[0]
        last = route.customer_ids[-1]
        for depot in depots:
            depot_cost[route_index, depot.id] = (
                distance(instance, ("depot", depot.id), ("customer", first))
                + distance(instance, ("customer", last), ("depot", depot.id))
            )

    model = gp.Model("route_depot_reassignment")
    model.Params.TimeLimit = max(1.0, args.time_limit)
    model.Params.MIPGap = 0.0
    x = {
        (route_index, depot.id): model.addVar(vtype=GRB.BINARY, name=f"x_{route_index}_{depot.id}")
        for route_index in range(len(routes))
        for depot in depots
        if loads[route_index] <= depot.capacity + 1e-9
    }
    y = {depot.id: model.addVar(vtype=GRB.BINARY, name=f"y_{depot.id}") for depot in depots}

    for route_index in range(len(routes)):
        model.addConstr(
            gp.quicksum(x[route_index, depot.id] for depot in depots if (route_index, depot.id) in x) == 1,
            name=f"assign_{route_index}",
        )
    for depot in depots:
        candidates = [route_index for route_index in range(len(routes)) if (route_index, depot.id) in x]
        model.addConstr(
            gp.quicksum(loads[r] * x[r, depot.id] for r in candidates) <= depot.capacity * y[depot.id],
            name=f"capacity_{depot.id}",
        )
        model.addConstr(
            gp.quicksum(x[r, depot.id] for r in candidates) <= depot.vehicle_limit * y[depot.id],
            name=f"vehicles_{depot.id}",
        )

    constant = instance.route_fixed_cost * len(routes)
    model.setObjective(
        constant
        + gp.quicksum(depot.opening_cost * y[depot.id] for depot in depots)
        + gp.quicksum(depot_cost[r, depot_id] * variable for (r, depot_id), variable in x.items()),
        GRB.MINIMIZE,
    )
    for (route_index, depot_id), variable in x.items():
        variable.Start = 1.0 if routes[route_index].depot_id == depot_id else 0.0
    for depot_id, variable in y.items():
        variable.Start = 1.0 if depot_id in seed.opened_depot_ids else 0.0

    model.optimize()
    if model.SolCount == 0:
        raise RuntimeError("Gurobi did not find a feasible route reassignment")

    reassigned: list[Route] = []
    changed = 0
    for route_index, route in enumerate(routes):
        depot_id = next(
            depot.id for depot in depots
            if (route_index, depot.id) in x and x[route_index, depot.id].X > 0.5
        )
        changed += depot_id != route.depot_id
        reassigned.append(Route(depot_id, list(route.customer_ids)))
    solution = Solution(instance.name, reassigned)
    final_validation = validate_solution(instance, solution)
    if not final_validation.is_feasible:
        raise RuntimeError("Reassigned solution is infeasible: " + "; ".join(final_validation.errors))
    write_solution(solution, Path(args.output), instance=instance)
    print(f"solution: {args.output}")
    print("feasible: true")
    print(f"cost: {objective_cost(instance, solution):.10f}")
    print(f"routes_reassigned: {changed}")
    print(f"depots_opened: {len(solution.opened_depot_ids)}")
    print(f"mip_gap: {model.MIPGap:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
