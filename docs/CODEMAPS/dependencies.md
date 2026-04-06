<!-- Generated: 2026-04-06 | Files scanned: 2 | Token estimate: ~150 -->
# Dependencies

## Python Packages

| Package | Used By | Purpose |
|---------|---------|---------|
| gurobipy | solver.py | Gurobi MIP solver (ILP formulation) |
| numpy | solver_heuristic.py | Fast array ops, RNG, neighbor sorting |

## Standard Library
- sys (CLI args)
- math (exp for SA acceptance)
- time (monotonic clock for time budget)

## External Requirements
- **Gurobi license** required for solver.py (commercial optimizer)
