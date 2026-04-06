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

    # Tighten time windows based on depot travel
    for i in range(1, N + 1):
        tw_lo[i] = max(tw_lo[i], T[0][i])
        tw_hi[i] = min(tw_hi[i], tw_hi[0] - T[i][0])
        
    return N, K, Q, Gamma, s, tw_lo, tw_hi, T, C


def write_output(filepath, obj_val, routes):
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


def solve(N, K, Q, Gamma, s, tw_lo, tw_hi, T, C, time_limit=300):
    model = g.Model()
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = max(1, time_limit - 5)

    V = range(N + 1)
    Cust = range(1, N + 1)

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

    x = {}
    for (i, j) in A:
        x[i, j] = model.addVar(vtype=g.GRB.BINARY, name=f"x_{i}_{j}")

    # tau[i] — service start time
    tau = {}
    for i in V:
        lb = tw_lo[i] if i > 0 else 0
        ub = tw_hi[i] if i > 0 else 0
        tau[i] = model.addVar(lb=lb, ub=ub, vtype=g.GRB.INTEGER, name=f"tau_{i}")

    # w[i] — cumulative load; INTEGER since demands and capacity are int
    w = {}
    w[0] = model.addVar(lb=0, ub=0, vtype=g.GRB.INTEGER, name="w_0")
    for i in Cust:
        w[i] = model.addVar(lb=s[i], ub=Q, vtype=g.GRB.INTEGER, name=f"w_{i}")

    # Branch priority: arc variables first
    for key in x:
        x[key].BranchPriority = 10

    model.setObjective(
        g.quicksum(C[i][j] * x[i, j] for (i, j) in A) +
        Gamma * g.quicksum(x[0, j] for j in Cust if (0, j) in A_set),
        g.GRB.MINIMIZE
    )

    # Visit each customer exactly once (in-degree = 1)
    for j in Cust:
        model.addConstr(
            g.quicksum(x[i, j] for i in V if (i, j) in A_set) == 1,
            name=f"visit_{j}"
        )

    # Flow conservation at each customer
    for j in Cust:
        model.addConstr(
            g.quicksum(x[i, j] for i in V if (i, j) in A_set) ==
            g.quicksum(x[j, k] for k in V if (j, k) in A_set),
            name=f"flow_{j}"
        )

    # Depot flow balance + vehicle limit
    model.addConstr(
        g.quicksum(x[0, j] for j in Cust if (0, j) in A_set) ==
        g.quicksum(x[j, 0] for j in Cust if (j, 0) in A_set),
        name="depot_balance"
    )
    model.addConstr(
        g.quicksum(x[0, j] for j in Cust if (0, j) in A_set) <= K,
        name="max_vans"
    )

    # Minimum vehicles lower bound
    total_demand = sum(s[i] for i in Cust)
    min_vehicles = -(-total_demand // Q)  # ceiling division
    model.addConstr(
        g.quicksum(x[0, j] for j in Cust if (0, j) in A_set) >= min_vehicles,
        name="min_vans"
    )

    # Time precedence (skip arcs returning to depot)
    for (i, j) in A:
        if j == 0:
            continue
        model.addConstr(
            tau[i] + T[i][j] - M[i, j] * (1 - x[i, j]) <= tau[j],
            name=f"time_{i}_{j}"
        )

    # Capacity propagation
    for (i, j) in A:
        if j == 0:
            continue
        model.addConstr(
            w[j] >= w[i] + s[j] - Q * (1 - x[i, j]),
            name=f"cap_{i}_{j}"
        )

    model.optimize()

    if model.Status == g.GRB.INFEASIBLE:
        return None, None

    if model.SolCount == 0:
        return None, None

    obj_val = model.ObjVal

    adj = {}
    for (i, j) in A:
        if x[i, j].X > 0.5:
            adj[i] = j

    routes = []
    for j in Cust:
        if (0, j) not in A_set:
            continue
        if x[0, j].X < 0.5:
            continue
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
    obj_val, routes = solve(N, K, Q, Gamma, s, tw_lo, tw_hi, T, C, time_limit)

    if obj_val is None:
        with open(output_file, 'w') as f:
            f.write('-1')
    else:
        write_output(output_file, obj_val, routes)


if __name__ == "__main__":
    main()
