# Architecture

This repository uses a `src` layout with one importable package: `smio_clrp`.

## Package Structure

- `core/`: immutable-ish problem data structures, typed node identifiers, route and solution models.
- `io/`: readers and writers for official-style instance and solution text files.
- `evaluation/`: objective computation and feasibility validation.
- `algorithms/base.py`: common `Solver`, `SolverConfig`, and `SolverResult` interfaces.
- `algorithms/constructive/`: deterministic constructive heuristics and multistart construction.
- `algorithms/local_search/`: directed 2-opt, relocate, swap, route reinsertion, and local-search driver.
- `algorithms/alns/`: ALNS solver, destroy and repair operators, acceptance criteria, adaptive operator selection, and config/state models.
- `algorithms/fixopt/`: fix-and-optimize and hybrid solver skeletons.
- `experiments/`: single-instance and batch runners.
- `utils/`: logging, seeding, and timing helpers.

## Data Model

Depot IDs and customer IDs are kept exactly as provided by the input. Because official files can use the same numeric ID in both domains, internal distance lookup uses typed node keys:

```python
("depot", depot_id)
("customer", customer_id)
```

For `FULL_MATRIX`, the matrix order is always depots first, then customers, matching the order in the parsed sections. Matrix distances are directed and no symmetry or triangle inequality is assumed.

## Adding Algorithms

New algorithms should subclass `Solver` and return `SolverResult`. The solver should validate any produced solution before reporting success. Put shared construction code in small helper functions and keep algorithm-specific parameters in `SolverConfig.metadata` or a dedicated config dataclass when the algorithm matures.

Recommended future flow:

1. Build a feasible solution with a constructive method.
2. Improve it with local search or ALNS operators.
3. Intensify selected neighborhoods with restricted subproblems.
4. Return a validated `SolverResult` with metadata needed for reproducibility.

## ALNS Flow

`ALNSSolver` builds an initial solution with `constructive_ls`, then iterates:

1. Select destroy and repair operators by adaptive roulette weights.
2. Remove customers/routes/depots from the incumbent.
3. Reinsert missing customers with greedy, regret, or noisy repair.
4. Validate the candidate.
5. Accept by the configured acceptance rule.
6. Apply local search on new best solutions by default.
7. Update operator rewards and keep the best feasible solution found.

All distance-sensitive operators use existing directed cost functions, so `FULL_MATRIX` asymmetry is preserved.
