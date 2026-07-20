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
python -m pip install -e ".[exact,alns,dev]"
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
- `alns`: Adaptive Large Neighborhood Search seeded by `constructive_ls`.
- `fixopt`: restricted Fix-and-Optimize intensification over selected depot/route/customer neighborhoods.
- `hybrid`: ALNS followed by Fix-and-Optimize.
- `halns`: cooperative ALNS followed by Variable Neighborhood Search (VNS).
- `clustered`: capacity-aware customer clustering followed by route/depot construction.
- `clustered_hybrid`: clustering, then ALNS, then Fix-and-Optimize.
- `clustered_halns`: capacity-aware clustering, then cooperative ALNS+VNS.

Local search currently includes directed intra-route 2-opt, customer relocation, customer swaps, and route reinsertion. All final solutions are validated before being reported.

The ALNS flow repeatedly destroys part of the incumbent solution, repairs it with feasible insertion, accepts or rejects the candidate, and updates adaptive operator weights. Destroy operators include random customer removal, worst customer removal, Shaw-related removal, route removal, and depot removal. Repair operators include greedy, regret-2, regret-3, and deterministic noisy regret repair. Acceptance options are accept-if-better, simulated annealing, and record-to-record; simulated annealing is the default.

Fix-and-Optimize releases a small neighborhood, keeps the rest of the solution fixed, and rebuilds the released customers with a restricted backend. Neighborhoods include depot, route, boundary customer, expensive customer, and route-pair neighborhoods. The default backend is `auto`: it uses a lazy optional Gurobi hook if available, otherwise falls back to the validated heuristic backend. This is a restricted matheuristic layer, not a full CLRP MIP.

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

Run ALNS:

```bash
clrp solve data/samples/tiny_coords.txt --algorithm alns --output solutions/tiny_coords_alns.sol --seed 1 --num-starts 20 --max-iterations 500 --time-limit 10
```

Run Fix-and-Optimize:

```bash
clrp solve data/samples/tiny_coords.txt --algorithm fixopt --output solutions/tiny_coords_fixopt.sol --seed 1 --fixopt-iterations 50 --fixopt-time-limit 10 --fixopt-backend heuristic
```

Run the hybrid ALNS + VNS solver:

```bash
clrp solve data/samples/tiny_coords.txt --algorithm hybrid --output solutions/tiny_coords_hybrid.sol --seed 1 --num-starts 20 --max-iterations 500 --time-limit 20 --hybrid-alns-fraction 0.75 --vns-iterations 50 --vns-shake-fractions 0.05,0.10,0.20,0.30
```

Run clustering, ALNS and FixOpt in sequence:

```bash
clrp solve data/samples/tiny_coords.txt --algorithm clustered_hybrid --cluster-method auto --output solutions/tiny_coords_clustered_hybrid.sol --seed 1 --max-iterations 500 --time-limit 20 --fixopt-iterations 50 --fixopt-time-limit 10
```

`--cluster-method auto` selects capacity-aware K-Means for `COORDS` instances and capacity-aware K-Medoids for `FULL_MATRIX` instances. K-Medoids uses the average of both directed arcs only while forming clusters; all route evaluation continues to use the original directed distances. Asking for `kmeans` on `FULL_MATRIX` safely falls back to K-Medoids and records that decision in the solver metadata.

Useful solve options:

- `--algorithm`: `greedy_nearest_depot`, `savings`, `regret`, `multistart`, `constructive_ls`, `alns`, `fixopt`, `hybrid`, `clustered`, or `clustered_hybrid`.
- `--seed`: deterministic seed for wrappers and future randomized variants.
- `--num-starts`: number of constructive starts for multistart, constructive local search, and ALNS initialization.
- `--time-limit`: optional time limit in seconds.
- `--regret-k`: regret level, either `2` or `3`.
- `--local-search`: shortcut that routes solving through `constructive_ls`.
- `--max-iterations`: local-search iteration cap for constructive local search, and ALNS iteration cap for `alns`.
- `--destroy-fraction-min` / `--destroy-fraction-max`: ALNS removal fraction bounds.
- `--initial-temperature` and `--cooling-rate`: simulated annealing controls.
- `--local-search-frequency`: `best`, `never`, or an accepted-move frequency for ALNS local search.
- `--verbose`: reserved ALNS verbosity flag.
- `--fixopt-iterations` and `--fixopt-time-limit`: FixOpt iteration and time budgets.
- `--hybrid-alns-fraction`: fraction of the hybrid global time budget reserved for ALNS (default `0.75`); VNS receives the remaining time.
- `--vns-iterations`: maximum VNS shake-and-descent iterations in the hybrid phase.
- `--vns-local-search-iterations`: local descent limit after each VNS shake.
- `--vns-shake-fractions`: comma-separated neighborhood strengths used by VNS.
- `--vns-stagnation-threshold`: ALNS iterations without improvement before a cooperative VNS kick.
- `--vns-stagnation-iterations`: short VNS iteration budget used by each stagnation kick.
- `--fixopt-backend`: `auto`, `heuristic`, or `mip`; `mip` requires optional `gurobipy`.
- `--max-customers-per-subproblem` / `--max-routes-per-subproblem`: restricted subproblem size limits.
- `--mip-time-limit`: optional restricted MIP backend budget.
- `--neighborhood-types`: comma-separated FixOpt neighborhood sequence.
- `--cluster-method`: `auto`, `kmeans`, or `kmedoids` for `clustered` and `clustered_hybrid`.
- `--cluster-count`: optional number of route clusters; it cannot be below the capacity-based minimum.
- `--cluster-iterations`: maximum improvement passes for the capacity-aware clustering phase.

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
clrp batch-solve data/samples --output-dir results/runs/smoke_constructive --algorithm constructive_ls --seed 1 --num-starts 20
```

Run a benchmark config:

```bash
python scripts/run_benchmark.py --config configs/smoke_samples.json
```

Update local best solutions:

```bash
python scripts/update_best_registry.py --instance-dir data/samples --run-dir results/runs/smoke_constructive --best-dir results/best
```

Build a validated bundle:

```bash
python scripts/build_submission_bundle.py --instance-dir data/samples --solution-dir results/best --output submissions/smoke_bundle.zip
```

## Current Limitations

- The current constructive, local-search, ALNS, and FixOpt methods are feasibility-first implementations, not a tuned final competition method.
- The optional MIP backend is a conservative restricted hook and falls back to heuristic reconstruction in this version; it is not a full exact CLRP model.
- The parser supports the documented official-style format and the artificial samples, but may need minor adjustments once official instances are released.
- No symmetry, triangle inequality or zero-based contiguous IDs are assumed.
- `data/samples` contains development samples only. Future official instances should go under `data/official`.

## Next Steps

1. Calibrate ALNS and FixOpt parameters on official instances when available.
2. Add performance profiling and parameter tuning workflows.
3. Prepare final challenge submission checks once official rules are published.
