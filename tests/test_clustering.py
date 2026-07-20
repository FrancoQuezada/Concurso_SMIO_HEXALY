from smio_clrp.algorithms.base import SolverConfig
from smio_clrp.algorithms.clustering import ClusteredConstructiveSolver, ClusteredHybridSolver, cluster_customers
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.io.instance_reader import read_instance


def test_auto_uses_kmeans_for_coordinate_instances_and_respects_vehicle_capacity():
    instance = read_instance("data/samples/tiny_coords.txt")

    result = cluster_customers(instance, method="auto")

    assert result.effective_method == "kmeans"
    assert sorted(customer_id for cluster in result.clusters for customer_id in cluster) == sorted(
        customer.id for customer in instance.customers
    )
    assert all(load <= instance.vehicle_capacity for load in result.loads)


def test_auto_uses_kmedoids_for_full_matrix_instances():
    instance = read_instance("data/samples/tiny_full_matrix.txt")

    result = cluster_customers(instance, method="auto")

    assert result.effective_method == "kmedoids"
    assert result.metadata["capacity_feasible"]


def test_kmeans_request_on_full_matrix_falls_back_to_kmedoids():
    instance = read_instance("data/samples/tiny_full_matrix.txt")

    result = cluster_customers(instance, method="kmeans")

    assert result.effective_method == "kmedoids"
    assert "fallback_reason" in result.metadata


def test_clustered_constructive_returns_feasible_solution_for_both_formats():
    for path in ("data/samples/tiny_coords.txt", "data/samples/tiny_full_matrix.txt"):
        instance = read_instance(path)
        result = ClusteredConstructiveSolver(SolverConfig(seed=7)).solve(instance)

        assert result.solution is not None
        assert result.feasible
        assert validate_solution(instance, result.solution).is_feasible


def test_clustered_hybrid_returns_a_feasible_solution():
    instance = read_instance("data/samples/tiny_coords.txt")
    result = ClusteredHybridSolver(
        SolverConfig(
            seed=3,
            metadata={"max_iterations": 20, "fixopt_iterations": 10, "cluster_iterations": 10},
        )
    ).solve(instance)

    assert result.solution is not None
    assert result.feasible
    assert validate_solution(instance, result.solution).is_feasible
