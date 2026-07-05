from __future__ import annotations

import math
from typing import overload

from smio_clrp.core.instance import Instance, NodeKey


@overload
def node_key(kind: str, node_id: int) -> NodeKey: ...


@overload
def node_key(kind: NodeKey, node_id: None = None) -> NodeKey: ...


def node_key(kind: str | NodeKey, node_id: int | None = None) -> NodeKey:
    if isinstance(kind, tuple):
        node_type, parsed_id = kind
    else:
        if node_id is None:
            raise ValueError("node_id is required when kind is not a NodeKey")
        aliases = {"d": "depot", "depot": "depot", "c": "customer", "customer": "customer"}
        try:
            node_type = aliases[kind.lower()]
        except KeyError as exc:
            raise ValueError(f"Unknown node type: {kind}") from exc
        parsed_id = node_id
    if node_type not in {"depot", "customer"}:
        raise ValueError(f"Unknown node type: {node_type}")
    return (node_type, int(parsed_id))  # type: ignore[return-value]


def distance(instance: Instance, from_node: NodeKey, to_node: NodeKey) -> float:
    """Return the directed distance between two typed nodes."""
    from_key = node_key(from_node)
    to_key = node_key(to_node)

    if instance.distance_format == "FULL_MATRIX":
        try:
            return float(instance.distance_matrix[instance.node_index[from_key], instance.node_index[to_key]])  # type: ignore[index]
        except KeyError as exc:
            raise ValueError(f"Unknown node in distance lookup: {exc}") from exc

    from_obj = _get_node(instance, from_key)
    to_obj = _get_node(instance, to_key)
    if from_obj.x is None or from_obj.y is None or to_obj.x is None or to_obj.y is None:
        raise ValueError("COORDS distance requires coordinates on both nodes")
    return round(math.hypot(from_obj.x - to_obj.x, from_obj.y - to_obj.y), 1)


def _get_node(instance: Instance, key: NodeKey):
    kind, node_id = key
    if kind == "depot":
        try:
            return instance.depots_by_id[node_id]
        except KeyError as exc:
            raise ValueError(f"Unknown depot id: {node_id}") from exc
    try:
        return instance.customers_by_id[node_id]
    except KeyError as exc:
        raise ValueError(f"Unknown customer id: {node_id}") from exc
