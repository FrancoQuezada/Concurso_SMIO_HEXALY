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

## CLI

Parse an instance:

```bash
clrp parse data/samples/tiny_coords.txt
```

Solve a sample with the baseline constructive heuristic:

```bash
clrp solve data/samples/tiny_coords.txt --algorithm greedy_nearest_depot --output solutions/tiny_coords.sol --seed 1
```

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
clrp batch-solve data/samples --output-dir solutions --algorithm greedy_nearest_depot --seed 1
```

## Current Limitations

- `greedy_nearest_depot` is a deterministic feasibility-oriented baseline, not a competitive final method.
- ALNS, local search, fix-and-optimize and hybrid solvers are skeletons.
- The parser supports the documented official-style format and the artificial samples, but may need minor adjustments once official instances are released.
- No symmetry, triangle inequality or zero-based contiguous IDs are assumed.

## Next Steps

1. Add stronger construction and route insertion heuristics.
2. Implement local-search neighborhoods.
3. Add ALNS destroy/repair operators.
4. Add optional exact restricted subproblems.
5. Expand experiment tracking and submission tooling.
