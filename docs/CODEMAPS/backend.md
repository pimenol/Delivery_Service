<!-- Generated: 2026-04-06 | Files scanned: 2 | Token estimate: ~800 -->
# Backend — Solver Details

## solver.py — Optimal ILP Solver (SP_CC_O)

### Functions
```
parse_input(filepath) → (N, K, Q, Gamma, s, tw_lo, tw_hi, T, C)
validate(N, K, Q, ..., obj_val, routes) → (bool, str)
write_output(filepath, obj_val, routes)
solve_ilp(N, K, Q, ..., time_limit=300) → (obj_val, routes)
main() — CLI entry point
```

### ILP Model (Gurobi)
- **Variables**: x[i,j] binary (arc selection), tau[i] continuous (arrival time), w[i] continuous (cumulative load)
- **Objective**: min Σ C[i][j]·x[i,j] + Gamma · Σ x[0,j]
- **Constraints**: visit-once (C1), flow conservation (C2), depot balance (C3), MTZ time (C4), MTZ capacity (C5)
- **Preprocessing**: Infeasible arcs pruned via time window checks; tight Big-M per arc
- **Params**: MIPGap=0, IntegralityFocus=1, Presolve=2, Cuts=2, MIPFocus=2

## solver_heuristic.py — ALNS Heuristic (SP_CC_T / SP_CC_R)

### Core Classes
```
Route        — customers[], load, cost, earliest[], fts[], waiting[]
             — can_insert(pos, u) O(1), insert(), remove_at(), recompute() O(m)
Solution     — routes[], total_cost, num_vans
             — find_customer(c), all_customers(), copy()
```

### Algorithm Pipeline
```
construct_regret2() → initial solution
    │
    ▼
ALNS loop (until time_limit - 1.5s):
    select destroy op (adaptive weights)
    select repair op  (adaptive weights)
    destroy → repair → SA acceptance
    every 50 iters: local_search()
    every 500 iters: try_reduce_vans()
    every 100 iters: update adaptive weights
    │
    ▼
Final polish: local_search() + try_reduce_vans()
```

### Destroy Operators (4)
| Operator | Strategy |
|----------|----------|
| random_removal | Uniform random selection |
| worst_removal | Remove highest-saving customers (randomized) |
| shaw_removal | Remove related customers (cost + TW + demand similarity) |
| string_removal | SISRs-style contiguous subsequence removal |

### Repair Operators (3)
| Operator | Strategy |
|----------|----------|
| greedy_insertion | Cheapest feasible position |
| regret2_insertion | Regret-2 heuristic |
| noisy_greedy_insertion | Greedy + random noise for diversification |

### Local Search Moves
- Intra-route relocate (or-opt(1))
- Inter-route relocate
- Inter-route swap

### SA Parameters
- Initial temp: calibrated so 5% worse accepted at 50%
- Cooling: 0.99975, then recalibrated at iter 200 based on elapsed time
- Adaptive weights: SIGMA_1=33, SIGMA_2=9, SIGMA_3=13, decay=0.1, segment=100
