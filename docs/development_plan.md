# Development Plan

## Phase 0: Infrastructure Completed

Implement parser, writer, data models, cost evaluation, validation, CLI, tests, artificial samples, and a simple feasible constructive heuristic.

## Phase 1: Constructive Heuristics and Local Search Implemented

Implemented components:

- `savings`: feasible depot assignment plus directed Clarke-Wright-style route merging per depot. For asymmetric matrices, merge savings are computed by comparing the directed before/after route costs of candidate concatenations, rather than using a symmetric closed-form saving.
- `regret`: regret-2 and regret-3 insertion with feasible insertions into existing routes or new depot routes.
- `multistart`: deterministic wrapper that evaluates constructive variants and keeps the best feasible solution.
- `constructive_ls`: multistart construction followed by local search.
- Local search operators: directed intra-route 2-opt, customer relocation, customer swap, and route reinsertion.

All solver outputs are validated before being reported.

## Phase 2: ALNS Implemented

Implemented components:

- `alns`: starts from `constructive_ls`, iterates destroy/repair, validates candidates, and keeps the best feasible solution.
- Destroy operators: random customer removal, worst customer removal, Shaw-related removal, route removal, and depot removal.
- Repair operators: greedy, regret-2, regret-3, and deterministic noise repair.
- Acceptance criteria: accept-if-better, simulated annealing, and record-to-record.
- Adaptive roulette-wheel operator selection with segment weight updates and per-operator statistics.
- Local search integration on new best solutions by default, with optional accepted-move frequency.

All operators reuse the existing objective and validator and keep compatibility with asymmetric `FULL_MATRIX` distances.

## Phase 3: Fix-and-Optimize

Future work: build restricted subproblems over selected depots, customers, or routes. Keep Gurobi optional and isolate exact-solver dependencies behind optional extras.

## Phase 4: Experiments

Add benchmark runners, parameter sweeps, reproducibility metadata, structured logs, and summary reports.

## Phase 5: Submission Workflow

Add final solution validation gates, cost cross-checks, artifact naming, metadata capture, and scripts for producing challenge-ready submissions.
