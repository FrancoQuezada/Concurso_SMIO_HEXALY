from smio_clrp.algorithms.base import SolverConfig
from smio_clrp.algorithms.clustering import ClusteredConstructiveSolver
from smio_clrp.algorithms.local_search.solver import ConstructiveLocalSearchSolver
from smio_clrp.algorithms.local_search.depot_reassignment import reassign_route_depot
from smio_clrp.algorithms.local_search.two_opt_star import inter_route_two_opt_star
from smio_clrp.algorithms.vns import ClusteredHybridALNSVNSSolver, HybridALNSVNSSolver, VNSSolver
from smio_clrp.evaluation.cost import objective_cost
from smio_clrp.evaluation.validator import validate_solution
from smio_clrp.experiments.solvers import make_solver
from smio_clrp.io.instance_reader import read_instance


def _initial_solution():
    instance = read_instance("data/samples/tiny_coords.txt")
    result = ConstructiveLocalSearchSolver(
        SolverConfig(seed=1, metadata={"num_starts": 4, "max_iterations": 10})
    ).solve(instance)
    assert result.solution is not None
    return instance, result.solution


def test_vns_returns_feasible_non_worsening_solution():
    instance, initial = _initial_solution()
    result = VNSSolver(
        initial,
        SolverConfig(
            seed=1,
            metadata={"vns_iterations": 8, "vns_local_search_iterations": 3},
        ),
    ).solve(instance)

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible
    assert result.cost is not None
    assert result.cost <= objective_cost(instance, initial)


def test_two_opt_star_preserves_feasibility_and_never_worsens():
    instance, initial = _initial_solution()

    result = inter_route_two_opt_star(instance, initial)

    assert validate_solution(instance, result).is_feasible
    assert objective_cost(instance, result) <= objective_cost(instance, initial)


def test_route_depot_reassignment_preserves_feasibility_and_never_worsens():
    instance, initial = _initial_solution()

    result = reassign_route_depot(instance, initial)

    assert validate_solution(instance, result).is_feasible
    assert objective_cost(instance, result) <= objective_cost(instance, initial)


def test_vns_is_deterministic_for_fixed_seed():
    instance, initial = _initial_solution()
    config = SolverConfig(seed=4, metadata={"vns_iterations": 8, "vns_local_search_iterations": 3})

    first = VNSSolver(initial, config).solve(instance)
    second = VNSSolver(initial, config).solve(instance)

    assert first.cost == second.cost
    assert first.solution == second.solution


def test_hybrid_algorithm_selects_alns_vns_solver():
    solver = make_solver(
        "halns",
        seed=1,
        parameters={"max_iterations": 5, "vns_iterations": 5},
    )

    assert isinstance(solver, HybridALNSVNSSolver)


def test_cluster_first_returns_feasible_solution():
    instance = read_instance("data/samples/tiny_coords.txt")

    result = ClusteredConstructiveSolver(SolverConfig(seed=1)).solve(instance)

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible


def test_clustered_hybrid_algorithm_selects_clustered_solver():
    solver = make_solver(
        "clustered_halns",
        seed=1,
        parameters={"max_iterations": 5, "vns_iterations": 5},
    )

    assert isinstance(solver, ClusteredHybridALNSVNSSolver)


def test_clustered_hybrid_returns_feasible_solution():
    instance = read_instance("data/samples/tiny_coords.txt")
    result = ClusteredHybridALNSVNSSolver(
        SolverConfig(
            seed=1,
            metadata={"max_iterations": 8, "vns_iterations": 5, "vns_local_search_iterations": 2},
        )
    ).solve(instance)

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible
    assert result.metadata["cluster_initial_cost"] is not None


def test_alns_vns_hybrid_returns_feasible_solution():
    instance = read_instance("data/samples/tiny_full_matrix.txt")
    result = HybridALNSVNSSolver(
        SolverConfig(
            seed=1,
            metadata={
                "num_starts": 4,
                "max_iterations": 10,
                "vns_iterations": 8,
                "vns_local_search_iterations": 3,
            },
        )
    ).solve(instance)

    assert result.solution is not None
    assert validate_solution(instance, result.solution).is_feasible
    assert result.metadata["vns_best_cost"] <= result.metadata["alns_best_cost"]
    assert "vns_stagnation_kicks" in result.metadata
