from __future__ import annotations

from dataclasses import dataclass

from smio_clrp.core.distance import distance
from smio_clrp.core.instance import Instance


EPS = 1e-9


@dataclass(frozen=True)
class ClusterResult:
    """A capacity-feasible partition of customers into candidate routes."""

    requested_method: str
    effective_method: str
    clusters: list[list[int]]
    loads: list[float]
    iterations: int
    metadata: dict[str, object]


def cluster_customers(
    instance: Instance,
    method: str = "auto",
    num_clusters: int | None = None,
    max_iterations: int = 20,
) -> ClusterResult:
    """Cluster customers while keeping every cluster within vehicle capacity.

    ``kmeans`` uses geometric centroids and is therefore applicable to COORDS.
    ``kmedoids`` uses a precomputed symmetric dissimilarity and works for both
    formats.  Directed matrix arcs are symmetrized only for clustering with
    ``(d(i, j) + d(j, i)) / 2``; routing itself keeps the original direction.
    """
    normalized = method.lower()
    if normalized not in {"auto", "kmeans", "kmedoids"}:
        raise ValueError("cluster_method must be one of: auto, kmeans, kmedoids")
    if max_iterations < 0:
        raise ValueError("cluster_iterations must be non-negative")
    if any(customer.demand > instance.vehicle_capacity + EPS for customer in instance.customers):
        raise ValueError("A customer demand exceeds VEHICLE_CAPACITY")

    seed_clusters = _first_fit_decreasing_clusters(instance)
    target_count = len(seed_clusters) if num_clusters is None else num_clusters
    if target_count < len(seed_clusters):
        raise ValueError(
            f"cluster_count={target_count} is too small for vehicle capacity; at least {len(seed_clusters)} is required"
        )
    if target_count > len(instance.customers):
        raise ValueError("cluster_count cannot exceed the number of customers")
    while len(seed_clusters) < target_count:
        _split_largest_cluster(instance, seed_clusters)

    effective = normalized
    fallback_reason: str | None = None
    if normalized == "auto":
        effective = "kmeans" if instance.distance_format == "COORDS" else "kmedoids"
    elif normalized == "kmeans" and instance.distance_format == "FULL_MATRIX":
        effective = "kmedoids"
        fallback_reason = "K-Means requires coordinates; K-Medoids was used for FULL_MATRIX"

    if effective == "kmeans":
        clusters, iterations = _capacitated_kmeans(instance, seed_clusters, max_iterations)
    else:
        clusters, iterations = _capacitated_kmedoids(instance, seed_clusters, max_iterations)
    loads = [_cluster_load(instance, cluster) for cluster in clusters]
    metadata: dict[str, object] = {
        "distance_format": instance.distance_format,
        "num_clusters": len(clusters),
        "capacity_feasible": all(load <= instance.vehicle_capacity + EPS for load in loads),
    }
    if fallback_reason is not None:
        metadata["fallback_reason"] = fallback_reason
    return ClusterResult(normalized, effective, clusters, loads, iterations, metadata)


def _first_fit_decreasing_clusters(instance: Instance) -> list[list[int]]:
    clusters: list[list[int]] = []
    loads: list[float] = []
    for customer in sorted(instance.customers, key=lambda item: (-item.demand, item.id)):
        feasible = [index for index, load in enumerate(loads) if load + customer.demand <= instance.vehicle_capacity + EPS]
        if feasible:
            index = min(feasible, key=lambda item: (instance.vehicle_capacity - loads[item] - customer.demand, item))
            clusters[index].append(customer.id)
            loads[index] += customer.demand
        else:
            clusters.append([customer.id])
            loads.append(customer.demand)
    return clusters


def _split_largest_cluster(instance: Instance, clusters: list[list[int]]) -> None:
    candidates = [index for index, cluster in enumerate(clusters) if len(cluster) > 1]
    if not candidates:
        raise ValueError("Cannot create more non-empty clusters than customers")
    index = max(candidates, key=lambda item: (_cluster_load(instance, clusters[item]), len(clusters[item]), -item))
    customer_id = max(clusters[index], key=lambda item: (instance.customers_by_id[item].demand, item))
    clusters[index].remove(customer_id)
    clusters.append([customer_id])


def _capacitated_kmeans(instance: Instance, clusters: list[list[int]], max_iterations: int) -> tuple[list[list[int]], int]:
    centers = [_centroid(instance, cluster) for cluster in clusters]
    return _improve_partition(
        instance,
        clusters,
        centers,
        lambda customer_id, center: _squared_euclidean(instance, customer_id, center),
        max_iterations,
        _centroid,
    )


def _capacitated_kmedoids(instance: Instance, clusters: list[list[int]], max_iterations: int) -> tuple[list[list[int]], int]:
    medoids = [_medoid(instance, cluster) for cluster in clusters]

    def score(customer_id: int, medoid_id: int) -> float:
        return _symmetric_distance(instance, customer_id, medoid_id)

    def representative(current: Instance, cluster: list[int]) -> int:
        return _medoid(current, cluster)

    return _improve_partition(instance, clusters, medoids, score, max_iterations, representative)


def _improve_partition(instance: Instance, initial: list[list[int]], representatives: list[object], score, max_iterations: int, update) -> tuple[list[list[int]], int]:
    clusters = [list(cluster) for cluster in initial]
    loads = [_cluster_load(instance, cluster) for cluster in clusters]
    completed_iterations = 0
    for _ in range(max_iterations):
        moved = False
        for source_index, cluster in enumerate(list(clusters)):
            for customer_id in sorted(list(cluster), key=lambda item: (-instance.customers_by_id[item].demand, item)):
                if len(clusters[source_index]) == 1:
                    continue
                demand = instance.customers_by_id[customer_id].demand
                current_score = score(customer_id, representatives[source_index])
                targets = [
                    target_index
                    for target_index in range(len(clusters))
                    if target_index != source_index and loads[target_index] + demand <= instance.vehicle_capacity + EPS
                ]
                if not targets:
                    continue
                best_target = min(targets, key=lambda item: (score(customer_id, representatives[item]), item))
                target_score = score(customer_id, representatives[best_target])
                if target_score + EPS >= current_score:
                    continue
                clusters[source_index].remove(customer_id)
                clusters[best_target].append(customer_id)
                loads[source_index] -= demand
                loads[best_target] += demand
                representatives[source_index] = update(instance, clusters[source_index])
                representatives[best_target] = update(instance, clusters[best_target])
                moved = True
        completed_iterations += 1
        if not moved:
            break
    return clusters, completed_iterations


def _centroid(instance: Instance, cluster: list[int]) -> tuple[float, float]:
    customers = [instance.customers_by_id[customer_id] for customer_id in cluster]
    return (
        sum(float(customer.x) for customer in customers) / len(customers),
        sum(float(customer.y) for customer in customers) / len(customers),
    )


def _squared_euclidean(instance: Instance, customer_id: int, center: tuple[float, float]) -> float:
    customer = instance.customers_by_id[customer_id]
    return (float(customer.x) - center[0]) ** 2 + (float(customer.y) - center[1]) ** 2


def _medoid(instance: Instance, cluster: list[int]) -> int:
    return min(
        cluster,
        key=lambda candidate: (sum(_symmetric_distance(instance, candidate, other) for other in cluster), candidate),
    )


def _symmetric_distance(instance: Instance, first: int, second: int) -> float:
    if first == second:
        return 0.0
    return (
        distance(instance, ("customer", first), ("customer", second))
        + distance(instance, ("customer", second), ("customer", first))
    ) / 2.0


def _cluster_load(instance: Instance, cluster: list[int]) -> float:
    return sum(instance.customers_by_id[customer_id].demand for customer_id in cluster)
