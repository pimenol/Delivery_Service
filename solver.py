#!/usr/bin/env python3
import sys
import gurobipy as g


def parse_input(filepath):
    with open(filepath, 'r') as f:
        tokens = f.read().split()
    idx = 0

    def next_int():
        nonlocal idx
        val = int(tokens[idx]); idx += 1; return val

    def next_float():
        nonlocal idx
        val = float(tokens[idx]); idx += 1; return val

    N = next_int()
    K = next_int()
    Q = next_int()
    Gamma = next_float()

    s = [0] * (N + 1)
    tw_lo = [0.0] * (N + 1)
    tw_hi = [0.0] * (N + 1)
    for i in range(1, N + 1):
        s[i] = next_int()
        tw_lo[i] = next_float()
        tw_hi[i] = next_float()

    T = []
    for i in range(N + 1):
        T.append([next_float() for _ in range(N + 1)])

    C = []
    for i in range(N + 1):
        C.append([next_float() for _ in range(N + 1)])

    tw_hi[0] = max(tw_hi[i] + T[i][0] for i in range(1, N + 1))

    return N, K, Q, Gamma, s, tw_lo, tw_hi, T, C


def validate(N, K, Q, Gamma, s, tw_lo, tw_hi, T, C, obj_val, routes):
    visited = set()
    computed_cost = 0.0

    for ri, route in enumerate(routes):
        load = sum(s[cust] for cust, _ in route)
        if load > Q:
            return False, f"Route {ri}: capacity {load} > {Q}"

        prev_node = 0
        prev_departure = 0.0
        for ci, (cust, arr_time) in enumerate(route):
            if cust in visited:
                return False, f"Customer {cust} visited more than once"
            visited.add(cust)

            earliest_arrival = prev_departure + T[prev_node][cust]
            if arr_time < earliest_arrival - 1e-6:
                return False, f"Route {ri}, stop {ci}: arrival {arr_time} < earliest {earliest_arrival}"
            if arr_time < tw_lo[cust] - 1e-6:
                return False, f"Route {ri}, stop {ci}: arrival {arr_time} < tw_lo {tw_lo[cust]}"
            if arr_time > tw_hi[cust] + 1e-6:
                return False, f"Route {ri}, stop {ci}: arrival {arr_time} > tw_hi {tw_hi[cust]}"

            prev_node = cust
            prev_departure = arr_time

        prev = 0
        for cust, _ in route:
            computed_cost += C[prev][cust]
            prev = cust
        computed_cost += C[prev][0]

    if visited != set(range(1, N + 1)):
        missing = set(range(1, N + 1)) - visited
        return False, f"Customers not visited: {missing}"
    if len(routes) > K:
        return False, f"Too many vans: {len(routes)} > {K}"

    computed_cost += Gamma * len(routes)
    if abs(computed_cost - obj_val) > 1e-3:
        return False, f"Objective mismatch: computed {computed_cost} != reported {obj_val}"

    return True, "Valid"


def write_output(filepath, obj_val, routes):
    """Write solution in compact format."""
    lines = []
    if obj_val == int(obj_val):
        lines.append(f"{obj_val:.1f} {len(routes)}")
    else:
        lines.append(f"{obj_val} {len(routes)}")

    for route in routes:
        parts = [str(len(route))]
        for cust, arr_time in route:
            if arr_time == int(arr_time):
                parts.append(f"{cust} {int(arr_time)}")
            else:
                parts.append(f"{cust} {arr_time}")
        lines.append(' '.join(parts))

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def solve_ilp(N, K, Q, Gamma, s, tw_lo, tw_hi, T, C, time_limit=300):
    """Solve CVRPTW optimally using 2-index MTZ formulation."""
    model = g.Model("CVRPTW")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0
    model.Params.MIPGapAbs = 0
    model.Params.TimeLimit = time_limit
    model.Params.IntegralityFocus = 1
    model.Params.Presolve = 2
    model.Params.Cuts = 2
    model.Params.MIPFocus = 2

    V = range(N + 1)
    Cust = range(1, N + 1)

    # Precompute feasible arc set — eliminate arcs that violate time windows
    # Arc (i,j) feasible if: earliest_departure_i + T[i][j] <= tw_hi[j]
    # Service time = 0, so earliest_departure_i = tw_lo[i] (or 0 for depot)
    A = []
    A_set = set()
    for i in V:
        for j in V:
            if i == j:
                continue
            ed_i = tw_lo[i] if i > 0 else 0
            if ed_i + T[i][j] <= tw_hi[j] + 1e-9:
                A.append((i, j))
                A_set.add((i, j))

    # Big-M per arc: M_ij = max(0, tw_hi[i] + T[i][j] - tw_lo[j])
    M = {}
    for (i, j) in A:
        ub_i = tw_hi[i] if i > 0 else tw_hi[0]
        lb_j = tw_lo[j] if j > 0 else 0
        M[i, j] = max(0, ub_i + T[i][j] - lb_j)

    # ─── Decision variables ───

    # x[i,j] binary — any vehicle traverses arc (i,j)
    x = {}
    for (i, j) in A:
        x[i, j] = model.addVar(vtype=g.GRB.BINARY, name=f"x_{i}_{j}")

    # tau[i] continuous — service start time at node i
    tau = {}
    for i in V:
        lb = tw_lo[i] if i > 0 else 0
        ub = tw_hi[i] if i > 0 else 0  # depot fixed at time 0
        tau[i] = model.addVar(lb=lb, ub=ub, vtype=g.GRB.CONTINUOUS, name=f"tau_{i}")

    # w[i] continuous — cumulative load upon arriving at customer i
    w = {}
    w[0] = model.addVar(lb=0, ub=0, vtype=g.GRB.CONTINUOUS, name="w_0")
    for i in Cust:
        w[i] = model.addVar(lb=s[i], ub=Q, vtype=g.GRB.CONTINUOUS, name=f"w_{i}")

    model.update()

    # Branch priority: arc variables first
    for key in x:
        x[key].BranchPriority = 10

    # ─── Objective ───
    # Travel cost + Gamma per van (van count = sum of x[0,j])
    model.setObjective(
        g.quicksum(C[i][j] * x[i, j] for (i, j) in A) +
        Gamma * g.quicksum(x[0, j] for j in Cust if (0, j) in A_set),
        g.GRB.MINIMIZE
    )

    # ─── Constraints ───

    # C1: Visit each customer exactly once (in-degree = 1)
    for j in Cust:
        model.addConstr(
            g.quicksum(x[i, j] for i in V if (i, j) in A_set) == 1,
            name=f"visit_{j}"
        )

    # C2: Flow conservation at each customer
    for j in Cust:
        model.addConstr(
            g.quicksum(x[i, j] for i in V if (i, j) in A_set) ==
            g.quicksum(x[j, k] for k in V if (j, k) in A_set),
            name=f"flow_{j}"
        )

    # C3: Depot flow balance + vehicle limit
    model.addConstr(
        g.quicksum(x[0, j] for j in Cust if (0, j) in A_set) ==
        g.quicksum(x[j, 0] for j in Cust if (j, 0) in A_set),
        name="depot_balance"
    )
    model.addConstr(
        g.quicksum(x[0, j] for j in Cust if (0, j) in A_set) <= K,
        name="max_vans"
    )

    # C4: Time precedence (MTZ subtour elimination), only for customer nodes
    for (i, j) in A:
        if j == 0:
            continue  # no time constraint on return to depot
        model.addConstr(
            tau[i] + T[i][j] - M[i, j] * (1 - x[i, j]) <= tau[j],
            name=f"time_{i}_{j}"
        )

    # C5: Capacity tracking (MTZ-style)
    for (i, j) in A:
        if j == 0:
            continue  # no capacity needed for depot return
        model.addConstr(
            w[j] >= w[i] + s[j] - Q * (1 - x[i, j]),
            name=f"cap_{i}_{j}"
        )

    # ─── Solve ───
    model.optimize()

    if model.Status == g.GRB.INFEASIBLE:
        print("Model is infeasible!", file=sys.stderr)
        sys.exit(1)

    if model.SolCount == 0:
        print("No solution found!", file=sys.stderr)
        sys.exit(1)

    # ─── Extract solution ───
    obj_val = model.ObjVal

    # Build adjacency from solution arcs
    adj = {}
    for (i, j) in A:
        if x[i, j].X > 0.5:
            adj[i] = j

    # Reconstruct routes by following chains from depot
    routes = []
    for j in Cust:
        if (0, j) not in A_set:
            continue
        if x[0, j].X < 0.5:
            continue
        # New route starting with customer j
        route = []
        current = j
        while current != 0:
            arrival = tau[current].X
            arrival = max(arrival, tw_lo[current])
            route.append((current, arrival))
            current = adj.get(current, 0)
        routes.append(route)

    return obj_val, routes


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    time_limit = float(sys.argv[3]) if len(sys.argv) > 3 else 300

    N, K, Q, Gamma, s, tw_lo, tw_hi, T, C = parse_input(input_file)
    obj_val, routes = solve_ilp(N, K, Q, Gamma, s, tw_lo, tw_hi, T, C, time_limit)

    valid, msg = validate(N, K, Q, Gamma, s, tw_lo, tw_hi, T, C, obj_val, routes)
    if not valid:
        print(f"WARNING: Solution validation failed: {msg}", file=sys.stderr)

    write_output(output_file, obj_val, routes)


if __name__ == "__main__":
    main()
