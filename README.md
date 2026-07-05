# SMIO-Hexaly CLRP Solver

Base architecture for the SMIO-Hexaly Location Routing Challenge Monterrey 2026.

The repository targets the Capacitated Location Routing Problem (CLRP): choose which depots to open, assign customers to depot-based routes, respect depot capacities, vehicle limits and vehicle capacity, then minimize depot opening costs, fixed route dispatching costs and routing distance.

Official instances are not available yet. The included files under `data/samples/` are artificial development samples only.

## Installation

```bash
python -m pip install -e ".[dev]"
```

Python 3.10 or newer is required. The base package only depends on `numpy`. Optional extras reserve future integrations:

```bash
python -m pip install -e ".[exact,alns,clustering,dev]"
```

## Tests

```bash
pytest -q
```

## Algorithms

Available constructive and improvement methods:

- `greedy_nearest_depot`: deterministic feasibility-first baseline.
- `savings`: feasible depot assignment followed by directed Clarke-Wright-style route merging per depot.
- `regret`: regret insertion with `--regret-k 2` or `--regret-k 3`.
- `multistart`: deterministic wrapper that tries several constructive variants and returns the best feasible solution.
- `constructive_ls`: multistart construction followed by local search.

Local search currently includes directed intra-route 2-opt, customer relocation, customer swaps, and route reinsertion. All final solutions are validated before being reported.

## CLI

Parse an instance:

```bash
clrp parse data/samples/tiny_coords.txt
```

Solve a sample with the baseline constructive heuristic:

```bash
clrp solve data/samples/tiny_coords.txt --algorithm greedy_nearest_depot --output solutions/tiny_coords.sol --seed 1
```

Solve with constructive multistart plus local search:

```bash
clrp solve data/samples/tiny_coords.txt --algorithm constructive_ls --output solutions/tiny_coords_ls.sol --seed 1 --num-starts 20 --time-limit 10
```

Use regret insertion:

```bash
clrp solve data/samples/tiny_full_matrix.txt --algorithm regret --regret-k 3 --output solutions/tiny_full_matrix_regret.sol
```

Useful solve options:

- `--algorithm`: `greedy_nearest_depot`, `savings`, `regret`, `multistart`, or `constructive_ls`.
- `--seed`: deterministic seed for wrappers and future randomized variants.
- `--num-starts`: number of constructive starts for multistart and constructive local search.
- `--time-limit`: optional time limit in seconds.
- `--regret-k`: regret level, either `2` or `3`.
- `--local-search`: shortcut that routes solving through `constructive_ls`.
- `--max-iterations`: local-search iteration cap.

Validate a solution:

```bash
clrp validate data/samples/tiny_coords.txt solutions/tiny_coords.sol
```

Print recomputed cost:

```bash
clrp cost data/samples/tiny_coords.txt solutions/tiny_coords.sol
```

Run a batch:

```bash
clrp batch-solve data/samples --output-dir solutions --algorithm constructive_ls --seed 1 --num-starts 20
```

## Current Limitations

- The current constructive and local-search methods are still simple feasibility-first heuristics, not a competitive final method.
- ALNS and fix-and-optimize solvers remain skeletons.
- The parser supports the documented official-style format and the artificial samples, but may need minor adjustments once official instances are released.
- No symmetry, triangle inequality or zero-based contiguous IDs are assumed.

## Next Steps

1. Improve constructive diversification and insertion scoring.
2. Add ALNS destroy/repair operators.
3. Add optional exact restricted subproblems.
4. Expand experiment tracking and submission tooling.
