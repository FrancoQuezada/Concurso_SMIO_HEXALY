from __future__ import annotations

import argparse
import math
from pathlib import Path

import pyvrp

from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


DISTANCE_SCALE = 10


def main() -> int:
    parser = argparse.ArgumentParser(description="Hybrid Genetic Search for the large CLRP")
    parser.add_argument("instance")
    parser.add_argument("--output", required=True)
    parser.add_argument("--time-limit", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--initial-solution")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    instance = read_instance(args.instance)

    model = pyvrp.Model()
    locations = []
    for depot in instance.depots:
        locations.append(model.add_depot(float(depot.x), float(depot.y), name=f"d{depot.id}"))
    for customer in instance.customers:
        locations.append(
            model.add_client(
                float(customer.x),
                float(customer.y),
                delivery=int(round(customer.demand)),
                name=f"c{customer.id}",
            )
        )

    vehicle_type_by_depot: dict[int, int] = {}
    for depot_index, depot in enumerate(instance.depots):
        vehicle_type_by_depot[depot.id] = len(vehicle_type_by_depot)
        amortized_opening = depot.opening_cost / max(1, depot.vehicle_limit)
        model.add_vehicle_type(
            num_available=depot.vehicle_limit,
            capacity=int(round(instance.vehicle_capacity)),
            start_depot=locations[depot_index],
            end_depot=locations[depot_index],
            fixed_cost=int(round((instance.route_fixed_cost + amortized_opening) * DISTANCE_SCALE)),
            name=f"depot_{depot.id}",
        )

    coordinates = [
        (float(depot.x), float(depot.y)) for depot in instance.depots
    ] + [
        (float(customer.x), float(customer.y)) for customer in instance.customers
    ]
    matrix = instance.distance_matrix
    for from_index, frm in enumerate(locations):
        for to_index, to in enumerate(locations):
            if from_index == to_index:
                continue
            if matrix is not None:
                arc_cost = int(round(float(matrix[from_index, to_index]) * DISTANCE_SCALE))
            else:
                x1, y1 = coordinates[from_index]
                x2, y2 = coordinates[to_index]
                arc_cost = int(round(math.hypot(x1 - x2, y1 - y2) * DISTANCE_SCALE))
            model.add_edge(frm, to, distance=arc_cost)

    initial = None
    if args.initial_solution:
        seed = read_solution(args.initial_solution)
        validation = validate_solution(instance, seed)
        if not validation.is_feasible:
            raise ValueError("Initial solution is infeasible: " + "; ".join(validation.errors))
        data = model.data()
        pyvrp_routes = [
            pyvrp.Route(
                data,
                [instance.node_index[("customer", customer_id)] for customer_id in route.customer_ids],
                vehicle_type_by_depot[route.depot_id],
            )
            for route in seed.routes
        ]
        initial = pyvrp.Solution(data, pyvrp_routes)

    result = model.solve(
        pyvrp.stop.MaxRuntime(max(1.0, args.time_limit)),
        seed=args.seed,
        display=not args.quiet,
        initial_solution=initial,
    )
    if not result.is_feasible():
        raise RuntimeError("PyVRP did not find a feasible vehicle-routing solution")

    depot_count = len(instance.depots)
    routes = []
    for route in result.best.routes():
        depot_id = instance.depots[route.start_depot()].id
        customer_ids = [instance.customers[index - depot_count].id for index in route.visits()]
        routes.append(Route(depot_id, customer_ids))
    solution = Solution(instance.name, routes)
    write_solution(solution, Path(args.output), instance=instance)
    validation = validate_solution(instance, solution)
    print(f"solution: {args.output}")
    print(f"feasible: {str(validation.is_feasible).lower()}")
    print(f"cost: {objective_cost(instance, solution):.10f}")
    print(f"routes: {len(routes)}")
    print(f"depots_opened: {len(solution.opened_depot_ids)}")
    print(f"pyvrp_cost: {result.cost():.10f}")
    print(f"iterations: {result.num_iterations}")
    print(f"runtime_seconds: {result.runtime:.3f}")
    for error in validation.errors:
        print(f"error: {error}")
    return 0 if validation.is_feasible else 2


if __name__ == "__main__":
    raise SystemExit(main())
