from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

NodeType = Literal["depot", "customer"]
NodeKey = tuple[NodeType, int]


@dataclass(frozen=True)
class Depot:
    id: int
    opening_cost: float
    capacity: float
    vehicle_limit: int
    x: float | None = None
    y: float | None = None


@dataclass(frozen=True)
class Customer:
    id: int
    demand: float
    x: float | None = None
    y: float | None = None


@dataclass
class Instance:
    name: str
    depots: list[Depot]
    customers: list[Customer]
    vehicle_capacity: float
    route_fixed_cost: float
    distance_format: Literal["COORDS", "FULL_MATRIX"]
    distance_matrix: np.ndarray | None = None
    depots_by_id: dict[int, Depot] = field(init=False)
    customers_by_id: dict[int, Customer] = field(init=False)
    node_index: dict[NodeKey, int] = field(init=False)

    def __post_init__(self) -> None:
        self.distance_format = self.distance_format.upper()  # type: ignore[assignment]
        if self.distance_format not in {"COORDS", "FULL_MATRIX"}:
            raise ValueError(f"Unsupported DISTANCE_FORMAT: {self.distance_format}")

        self.depots_by_id = _unique_by_id(self.depots, "depot")
        self.customers_by_id = _unique_by_id(self.customers, "customer")
        self.node_index = {}
        for idx, depot in enumerate(self.depots):
            self.node_index[("depot", depot.id)] = idx
        offset = len(self.depots)
        for idx, customer in enumerate(self.customers):
            self.node_index[("customer", customer.id)] = offset + idx

        if self.vehicle_capacity <= 0:
            raise ValueError("VEHICLE_CAPACITY must be positive")
        if self.route_fixed_cost < 0:
            raise ValueError("ROUTE_FIXED_COST must be non-negative")
        for depot in self.depots:
            if depot.opening_cost < 0:
                raise ValueError(f"Depot {depot.id} has negative opening cost")
            if depot.capacity < 0:
                raise ValueError(f"Depot {depot.id} has negative capacity")
            if depot.vehicle_limit < 0:
                raise ValueError(f"Depot {depot.id} has negative vehicle limit")
        for customer in self.customers:
            if customer.demand <= 0:
                raise ValueError(f"Customer {customer.id} demand must be positive")

        if self.distance_format == "COORDS":
            missing = [
                key
                for key, node in self.iter_nodes()
                if node.x is None or node.y is None
            ]
            if missing:
                raise ValueError(f"Missing coordinates for nodes: {missing}")
        else:
            expected = len(self.depots) + len(self.customers)
            if self.distance_matrix is None:
                raise ValueError("FULL_MATRIX instances require DISTANCE_SECTION")
            if self.distance_matrix.shape != (expected, expected):
                raise ValueError(
                    f"DISTANCE_SECTION must be {expected}x{expected}, "
                    f"got {self.distance_matrix.shape}"
                )
            if np.any(self.distance_matrix < 0):
                raise ValueError("Distances must be non-negative")

    def iter_nodes(self) -> list[tuple[NodeKey, Depot | Customer]]:
        return [
            *((("depot", depot.id), depot) for depot in self.depots),
            *((("customer", customer.id), customer) for customer in self.customers),
        ]


def _unique_by_id(nodes: list[Depot] | list[Customer], kind: str) -> dict[int, Depot] | dict[int, Customer]:
    by_id: dict[int, Depot] | dict[int, Customer] = {}
    for node in nodes:
        if node.id in by_id:
            raise ValueError(f"Duplicate {kind} id: {node.id}")
        by_id[node.id] = node
    return by_id
