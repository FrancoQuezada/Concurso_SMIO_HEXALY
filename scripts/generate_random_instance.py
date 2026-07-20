from __future__ import annotations

import argparse
import math
import random
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a reproducible SMIO-like Euclidean CLRP instance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--customers", type=int, default=200)
    parser.add_argument("--depots", type=int, default=10)
    args = parser.parse_args()
    if args.customers < 1 or args.depots < 1:
        parser.error("customers and depots must be positive")

    rng = random.Random(args.seed)
    width = 1000
    vehicle_capacity = 100
    route_fixed_cost = 80
    cluster_centers = [(180, 220), (780, 190), (250, 760), (760, 750)]

    customers: list[tuple[int, int, float, float]] = []
    for index in range(args.customers):
        if rng.random() < 0.75:
            center_x, center_y = rng.choice(cluster_centers)
            x = min(width, max(0, rng.gauss(center_x, 95)))
            y = min(width, max(0, rng.gauss(center_y, 95)))
        else:
            x = rng.uniform(0, width)
            y = rng.uniform(0, width)
        demand = rng.randint(5, 12) if rng.random() < 0.7 else rng.randint(18, 30)
        customers.append((1001 + index, demand, x, y))

    total_demand = sum(demand for _, demand, _, _ in customers)
    depot_capacity = math.ceil(total_demand * 1.25 / args.depots / 10) * 10
    max_vehicles = max(4, math.ceil(depot_capacity / vehicle_capacity) + 1)
    depots: list[tuple[int, int, int, int, float, float]] = []
    for index in range(args.depots):
        if index < len(cluster_centers):
            x, y = cluster_centers[index]
        else:
            x, y = rng.uniform(50, 950), rng.uniform(50, 950)
        opening_cost = rng.randint(650, 1250)
        capacity = depot_capacity + rng.randint(-20, 30)
        depots.append((index + 1, opening_cost, capacity, max_vehicles, x, y))

    lines = [
        f"# seed = {args.seed}",
        "# generator_version = 1.0.0",
        "# distribution = mixed (75% clustered + 25% uniform)",
        "# demand = bimodal; capacity_slack = moderate",
        "# Field order follows the current repository reader, not the official PDF order.",
        f"NAME : random_mixed_c{args.customers}_d{args.depots}_s{args.seed}",
        f"CUSTOMERS : {args.customers}",
        f"DEPOTS : {args.depots}",
        f"VEHICLE_CAPACITY : {vehicle_capacity}",
        f"ROUTE_FIXED_COST : {route_fixed_cost}",
        "DISTANCE_FORMAT : COORDS",
        "",
        "DEPOT_SECTION",
        "# id opening_cost capacity max_vehicles x y",
    ]
    lines.extend(f"{i} {cost} {capacity} {vehicles} {x:.1f} {y:.1f}" for i, cost, capacity, vehicles, x, y in depots)
    lines.extend(["", "CUSTOMER_SECTION", "# id demand x y"])
    lines.extend(f"{i} {demand} {x:.1f} {y:.1f}" for i, demand, x, y in customers)
    lines.append("EOF")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
