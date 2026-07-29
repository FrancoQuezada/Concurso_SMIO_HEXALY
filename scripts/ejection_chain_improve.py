from __future__ import annotations

import argparse
from pathlib import Path

from smio_clrp.algorithms.common import route_load
from smio_clrp.algorithms.local_search.two_opt import improve_solution_two_opt
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


EPS = 1e-9


def _best_insertion(instance, route: Route, customer_id: int) -> Route:
    candidates = []
    for position in range(len(route.customer_ids) + 1):
        customers = list(route.customer_ids)
        customers.insert(position, customer_id)
        candidate = Route(route.depot_id, customers)
        candidates.append((route_distance(instance, candidate), position, candidate))
    return min(candidates, key=lambda item: (item[0], item[1]))[2]


def _replace_routes(
    solution: Solution,
    source_index: int,
    target_index: int,
    target: Route,
    receiver_index: int | None,
    receiver: Route | None,
) -> Solution:
    routes: list[Route] = []
    for index, route in enumerate(solution.routes):
        if index == source_index:
            continue
        if index == target_index:
            routes.append(target)
        elif receiver_index is not None and index == receiver_index:
            assert receiver is not None
            routes.append(receiver)
        else:
            routes.append(Route(route.depot_id, list(route.customer_ids)))
    return Solution(solution.instance_name, routes)


def improve_singleton_ejection(instance, solution: Solution) -> tuple[Solution, dict[str, object]]:
    best = solution
    best_cost = objective_cost(instance, solution)
    initial_cost = best_cost
    evaluated = 0

    singleton_indices = [
        index for index, route in enumerate(solution.routes) if len(route.customer_ids) == 1
    ]
    for source_index in singleton_indices:
        source = solution.routes[source_index]
        inserted_customer = source.customer_ids[0]
        inserted_demand = instance.customers_by_id[inserted_customer].demand

        for target_index, target in enumerate(solution.routes):
            if target_index == source_index:
                continue
            target_load = route_load(instance, target)
            ejection_options: list[int | None] = [None, *target.customer_ids]
            for ejected_customer in ejection_options:
                ejected_demand = (
                    0.0
                    if ejected_customer is None
                    else instance.customers_by_id[ejected_customer].demand
                )
                if target_load - ejected_demand + inserted_demand > instance.vehicle_capacity + EPS:
                    continue

                reduced_target = Route(
                    target.depot_id,
                    [customer for customer in target.customer_ids if customer != ejected_customer],
                )
                improved_target = _best_insertion(instance, reduced_target, inserted_customer)

                if ejected_customer is None:
                    candidate = _replace_routes(
                        solution, source_index, target_index, improved_target, None, None
                    )
                    evaluated += 1
                    validation = validate_solution(instance, candidate)
                    if validation.is_feasible and validation.cost + EPS < best_cost:
                        best = candidate
                        best_cost = validation.cost
                    continue

                for receiver_index, receiver in enumerate(solution.routes):
                    if receiver_index in {source_index, target_index}:
                        continue
                    if (
                        route_load(instance, receiver) + ejected_demand
                        > instance.vehicle_capacity + EPS
                    ):
                        continue
                    improved_receiver = _best_insertion(instance, receiver, ejected_customer)
                    candidate = _replace_routes(
                        solution,
                        source_index,
                        target_index,
                        improved_target,
                        receiver_index,
                        improved_receiver,
                    )
                    evaluated += 1
                    validation = validate_solution(instance, candidate)
                    if validation.is_feasible and validation.cost + EPS < best_cost:
                        best = candidate
                        best_cost = validation.cost

    if best_cost + EPS < initial_cost:
        best = improve_solution_two_opt(instance, best)
        best_cost = objective_cost(instance, best)
    return best, {
        "initial_cost": initial_cost,
        "best_cost": best_cost,
        "evaluated": evaluated,
        "singleton_routes": len(singleton_indices),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Eliminate singleton routes using one ejection")
    parser.add_argument("instance")
    parser.add_argument("solution")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    instance = read_instance(args.instance)
    solution = read_solution(args.solution)
    validation = validate_solution(instance, solution)
    if not validation.is_feasible:
        raise ValueError("Input solution is infeasible: " + "; ".join(validation.errors))

    improved, metadata = improve_singleton_ejection(instance, solution)
    final_validation = validate_solution(instance, improved)
    if not final_validation.is_feasible:
        raise RuntimeError("Output solution is infeasible: " + "; ".join(final_validation.errors))
    write_solution(improved, Path(args.output), instance=instance)
    print(f"solution: {args.output}")
    print("feasible: true")
    print(f"cost: {final_validation.cost:.10f}")
    print(f"routes: {len(improved.routes)}")
    for key, value in metadata.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
