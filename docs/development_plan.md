# Development Plan

## Phase 0: Infrastructure

Implement parser, writer, data models, cost evaluation, validation, CLI, tests, artificial samples, and a simple feasible constructive heuristic.

## Phase 1: Constructive Heuristics and Local Search

Add stronger depot assignment logic, route insertion policies, intra-route 2-opt, inter-route relocate/swap moves, and capacity-aware repair steps.

## Phase 2: ALNS

Implement destroy and repair operators, adaptive weights, acceptance criteria, temperature schedules, and solution pools. Keep all operators compatible with asymmetric distances.

## Phase 3: Fix-and-Optimize

Build restricted subproblems over selected depots, customers, or routes. Keep Gurobi optional and isolate exact-solver dependencies behind optional extras.

## Phase 4: Experiments

Add benchmark runners, parameter sweeps, reproducibility metadata, structured logs, and summary reports.

## Phase 5: Submission Workflow

Add final solution validation gates, cost cross-checks, artifact naming, metadata capture, and scripts for producing challenge-ready submissions.
