<!-- Generated: 2026-04-06 | Files scanned: 2 | Token estimate: ~600 -->
# Architecture — CVRPTW Solver Suite

## Overview
Single-app Python project: two standalone solvers for the Capacitated Vehicle Routing Problem with Time Windows (CVRPTW).

```
Input File ──► parse_input() ──► Solver ──► validate() ──► write_output() ──► Output File
                                   │
                        ┌──────────┴──────────┐
                        │                     │
                  solver.py              solver_heuristic.py
                  (ILP/Optimal)          (ALNS/Heuristic)
```

## Solvers

| Solver | File | Method | Dependency |
|--------|------|--------|------------|
| SP_CC_O (Optimal) | solver.py (289 lines) | 2-index MTZ ILP formulation | gurobipy |
| SP_CC_T / SP_CC_R (Heuristic) | solver_heuristic.py (845 lines) | ALNS + Simulated Annealing | numpy |

## Data Flow
1. **Parse**: Read N customers, K vans, Q capacity, Gamma van-cost, demands, time windows, travel/cost matrices
2. **Solve**: ILP (exact) or ALNS (metaheuristic with destroy/repair operators)
3. **Validate**: Check capacity, time windows, visit uniqueness, objective correctness
4. **Output**: `<objective> <num_routes>\n` then per-route: `<num_stops> <cust> <arrival> ...`

## CLI Interface
```
python solver.py <input> <output> [time_limit=300]
python solver_heuristic.py <input> <output> [time_limit=300]
```

## Test Data
- `data/CC-O/` — Small instances for optimal solver (public-1, public-2)
- `data/CC-T/` — Larger instances for heuristic solver (public-random-{1,2,10,12})
