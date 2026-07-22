#!/usr/bin/env python3
"""Greedy depot-closure post-processing: for each currently open depot, test removing
ALL its routes and reinserting those customers into the OTHER open depots (never
allowed to reopen the depot being tested), accept the closure if it's a net
improvement. Repeat until no single closure improves the solution.

Usage:
    python scripts/close_depots.py data/official/clrp-medium-01.txt results/competition/best.sol \
        --output results/competition/best_depotclosed.sol --max-passes 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from smio_clrp.algorithms.common import EPS, depot_loads, depot_route_counts, route_load
from smio_clrp.core.distance import distance
from smio_clrp.core.instance import Instance
from smio_clrp.core.solution import Route, Solution
from smio_clrp.evaluation.cost import objective_cost, route_distance
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance
from smio_clrp.io.solution_reader import read_solution
from smio_clrp.io.solution_writer import write_solution


def best_partial_insertion(instance: Instance, routes: list[Route], customer_id: int, forbidden_depot_id: int):
    demand = instance.customers_by_id[customer_id].demand
    opened_depot_ids = {route.depot_id for route in routes}
    loads = depot_loads(instance, routes)
    counts = depot_route_counts(routes)
    route_loads = [route_load(instance, route) for route in routes]

    best_cost = float("inf")
    best_move = None  # (route_index, position) or (-1, depot_id)

    for route_index, route in enumerate(routes):
        depot = instance.depots_by_id[route.depot_id]
        if route_loads[route_index] + demand > instance.vehicle_capacity + EPS:
            continue
        if loads.get(route.depot_id, 0.0) + demand > depot.capacity + EPS:
            continue
        for position in range(len(route.customer_ids) + 1):
            cost = insertion_delta_local(instance, route, customer_id, position)
            if cost < best_cost:
                best_cost = cost
                best_move = (route_index, position)

    if demand <= instance.vehicle_capacity + EPS:
        for depot in instance.depots:
            if depot.id == forbidden_depot_id:
                continue
            if loads.get(depot.id, 0.0) + demand > depot.capacity + EPS:
                continue
            if counts.get(depot.id, 0) + 1 > depot.vehicle_limit:
                continue
            opening_cost = 0.0 if depot.id in opened_depot_ids else depot.opening_cost
            cost = route_distance(instance, Route(depot.id, [customer_id])) + instance.route_fixed_cost + opening_cost
            if cost < best_cost:
                best_cost = cost
                best_move = (-1, depot.id)

    if best_move is None:
        return None

    route_index, position_or_depot = best_move
    if route_index == -1:
        return routes + [Route(position_or_depot, [customer_id])]
    updated = [Route(item.depot_id, list(item.customer_ids)) for item in routes]
    customers = list(updated[route_index].customer_ids)
    customers.insert(position_or_depot, customer_id)
    updated[route_index] = Route(routes[route_index].depot_id, customers)
    return updated


def insertion_delta_local(instance: Instance, route: Route, customer_id: int, position: int) -> float:
    customers = route.customer_ids
    size = len(customers)
    new_node = ("customer", customer_id)
    if size == 0:
        depot_node = ("depot", route.depot_id)
        return distance(instance, depot_node, new_node) + distance(instance, new_node, depot_node)
    prev_node = ("customer", customers[position - 1]) if position > 0 else ("depot", route.depot_id)
    next_node = ("customer", customers[position]) if position < size else ("depot", route.depot_id)
    removed_edge = distance(instance, prev_node, next_node)
    added_edges = distance(instance, prev_node, new_node) + distance(instance, new_node, next_node)
    return added_edges - removed_edge


def reinsert_customers(instance: Instance, routes: list[Route], customer_ids: list[int], forbidden_depot_id: int):
    current = [Route(route.depot_id, list(route.customer_ids)) for route in routes]
    for customer_id in sorted(customer_ids, key=lambda item: (-instance.customers_by_id[item].demand, item)):
        result = best_partial_insertion(instance, current, customer_id, forbidden_depot_id)
        if result is None:
            return None
        current = result
    return current


def try_close_depot(instance: Instance, solution: Solution, depot_id: int, current_cost: float):
    kept_routes = [route for route in solution.routes if route.depot_id != depot_id]
    released_customers = [cid for route in solution.routes if route.depot_id == depot_id for cid in route.customer_ids]
    if not released_customers:
        return None
    candidate_routes = reinsert_customers(instance, kept_routes, released_customers, depot_id)
    if candidate_routes is None:
        return None
    candidate = Solution(solution.instance_name, candidate_routes)
    validation = validate_solution(instance, candidate)
    if not validation.is_feasible:
        return None
    cost = objective_cost(instance, candidate)
    if cost + 1e-6 < current_cost:
        return candidate, cost
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance_path")
    parser.add_argument("solution_path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-passes", type=int, default=10)
    args = parser.parse_args()

    instance = read_instance(args.instance_path)
    solution = read_solution(args.solution_path)
    validation = validate_solution(instance, solution)
    if not validation.is_feasible:
        print(f"ERROR: initial solution infeasible: {validation.errors}")
        return 1
    cost = objective_cost(instance, solution)
    print(f"initial cost: {cost:.2f}, depots opened: {len({r.depot_id for r in solution.routes})}")

    t0 = time.perf_counter()
    for pass_num in range(1, args.max_passes + 1):
        open_depots = sorted({route.depot_id for route in solution.routes})
        loads = depot_loads(instance, solution.routes)
        open_depots.sort(key=lambda d: loads.get(d, 0.0))

        best_candidate = None
        best_cost = cost
        best_depot = None
        for depot_id in open_depots:
            result = try_close_depot(instance, solution, depot_id, best_cost)
            if result is not None:
                candidate, candidate_cost = result
                if candidate_cost < best_cost:
                    best_candidate, best_cost, best_depot = candidate, candidate_cost, depot_id

        if best_candidate is None:
            print(f"pass {pass_num}: no depot closure improves further, stopping")
            break

        solution = best_candidate
        cost = best_cost
        print(f"pass {pass_num}: closed depot {best_depot}, new cost={cost:.2f}, "
              f"depots opened={len({r.depot_id for r in solution.routes})}, elapsed={time.perf_counter()-t0:.1f}s")

    final_validation = validate_solution(instance, solution)
    print(f"FINAL: cost={cost:.2f} feasible={final_validation.is_feasible} "
          f"depots_opened={len({r.depot_id for r in solution.routes})} time={time.perf_counter()-t0:.1f}s")
    if final_validation.is_feasible:
        write_solution(solution, args.output, instance=instance)
        print(f"written: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
