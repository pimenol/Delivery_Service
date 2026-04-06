#!/usr/bin/env python3
"""CVRPTW Optimal ILP Solver (SP_CC_O) using Gurobi."""
import sys
import gurobipy as g


def parse_input(filepath):
    """Parse CVRPTW instance from file. Returns problem data."""
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

    # Customer data (1-indexed: customers 1..N)
    s = [0] * (N + 1)       # parcel sizes, s[0]=0 for depot
    tw_lo = [0.0] * (N + 1) # time window lower bounds
    tw_hi = [0.0] * (N + 1) # time window upper bounds
    for i in range(1, N + 1):
        s[i] = next_int()
        tw_lo[i] = next_float()
        tw_hi[i] = next_float()

    # Travel time matrix (N+1) x (N+1)
    T = []
    for i in range(N + 1):
        row = [next_float() for _ in range(N + 1)]
        T.append(row)

    # Cost matrix (N+1) x (N+1)
    C = []
    for i in range(N + 1):
        row = [next_float() for _ in range(N + 1)]
        C.append(row)

    # Depot time window: can depart at time 0, return by latest feasible time
    tw_hi[0] = max(tw_hi[i] + T[i][0] for i in range(1, N + 1))

    return N, K, Q, Gamma, s, tw_lo, tw_hi, T, C


def validate(N, K, Q, Gamma, s, tw_lo, tw_hi, T, C, obj_val, routes):
    """Validate a CVRPTW solution. Returns (is_valid, message)."""
    visited = set()
    computed_cost = 0.0

    for ri, route in enumerate(routes):
        # Check capacity
        load = sum(s[cust] for cust, _ in route)
        if load > Q:
            return False, f"Route {ri}: capacity {load} > {Q}"

        # Check time windows and travel time consistency
        prev_node = 0
        prev_departure = 0.0
        for ci, (cust, arr_time) in enumerate(route):
            if cust in visited:
                return False, f"Customer {cust} visited more than once"
            visited.add(cust)

            # Arrival must be >= departure from prev + travel time
            earliest_arrival = prev_departure + T[prev_node][cust]
            if arr_time < earliest_arrival - 1e-6:
                return False, f"Route {ri}, stop {ci}: arrival {arr_time} < earliest {earliest_arrival}"

            # Time window check
            if arr_time < tw_lo[cust] - 1e-6:
                return False, f"Route {ri}, stop {ci}: arrival {arr_time} < tw_lo {tw_lo[cust]}"
            if arr_time > tw_hi[cust] + 1e-6:
                return False, f"Route {ri}, stop {ci}: arrival {arr_time} > tw_hi {tw_hi[cust]}"

            prev_node = cust
            prev_departure = arr_time  # delivery time = arrival time (service time = 0)

        # Add travel costs for this route
        prev = 0
        for cust, _ in route:
            computed_cost += C[prev][cust]
            prev = cust
        computed_cost += C[prev][0]  # return to depot

    # Check all customers visited
    if visited != set(range(1, N + 1)):
        missing = set(range(1, N + 1)) - visited
        return False, f"Customers not visited: {missing}"

    # Check van count
    if len(routes) > K:
        return False, f"Too many vans: {len(routes)} > {K}"

    # Check objective
    computed_cost += Gamma * len(routes)
    if abs(computed_cost - obj_val) > 1e-3:
        return False, f"Objective mismatch: computed {computed_cost} != reported {obj_val}"

    return True, "Valid"


def write_output(filepath, obj_val, routes):
    """Write solution in compact format."""
    lines = []
    # Format objective: use .1f if it has a fractional part, else int-like
    if obj_val == int(obj_val):
        lines.append(f"{obj_val:.1f} {len(routes)}")
    else:
        lines.append(f"{obj_val} {len(routes)}")

    for route in routes:
        parts = [str(len(route))]
        for cust, arr_time in route:
            # Format arrival time as int if whole number
            if arr_time == int(arr_time):
                parts.append(f"{cust} {int(arr_time)}")
            else:
                parts.append(f"{cust} {arr_time}")
        lines.append(' '.join(parts))

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def solve_ilp(N, K, Q, Gamma, s, tw_lo, tw_hi, T, C, time_limit=300):
    """Solve CVRPTW optimally using Gurobi ILP."""
    model = g.Model("CVRPTW")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 0
    model.Params.TimeLimit = time_limit

    nodes = range(N + 1)      # 0 = depot, 1..N = customers
    customers = range(1, N + 1)
    vans = range(K)

    # Precompute feasible arcs (arc i->j is feasible if earliest arrival at j <= tw_hi[j])
    # For customer i: earliest departure = tw_lo[i], so earliest arrival at j = tw_lo[i] + T[i][j]
    # For depot (i=0): earliest departure = 0, so earliest arrival at j = T[0][j]
    feasible = {}
    for i in nodes:
        for j in nodes:
            if i == j:
                continue
            earliest_dep_i = tw_lo[i] if i > 0 else 0
            if earliest_dep_i + T[i][j] <= tw_hi[j] + 1e-9:
                feasible[i, j] = True

    # Decision variables
    # x[k,i,j] = 1 if van k travels from i to j
    x = {}
    for k in vans:
        for (i, j) in feasible:
            x[k, i, j] = model.addVar(vtype=g.GRB.BINARY, name=f"x_{k}_{i}_{j}")

    # t[k,i] = arrival time of van k at node i
    t = {}
    for k in vans:
        for i in nodes:
            lb = tw_lo[i] if i > 0 else 0
            ub = tw_hi[i] if i > 0 else tw_hi[0]
            t[k, i] = model.addVar(lb=lb, ub=ub, vtype=g.GRB.CONTINUOUS, name=f"t_{k}_{i}")

    # y[k] = 1 if van k is used
    y = {}
    for k in vans:
        y[k] = model.addVar(vtype=g.GRB.BINARY, name=f"y_{k}")

    model.update()

    # Objective: minimize travel cost + van usage cost
    model.setObjective(
        g.quicksum(C[i][j] * x[k, i, j] for (k, i, j) in x) +
        Gamma * g.quicksum(y[k] for k in vans),
        g.GRB.MINIMIZE
    )

    # Constraint 1: Each customer visited exactly once
    for i in customers:
        model.addConstr(
            g.quicksum(x[k, j, i] for k in vans for j in nodes
                       if (k, j, i) in x) == 1,
            name=f"visit_{i}"
        )

    # Constraint 2: Flow conservation for each van at each customer
    for k in vans:
        for i in customers:
            model.addConstr(
                g.quicksum(x[k, j, i] for j in nodes if (k, j, i) in x) ==
                g.quicksum(x[k, i, j] for j in nodes if (k, i, j) in x),
                name=f"flow_{k}_{i}"
            )

    # Constraint 3: Depot flow — each van leaves and returns at most once
    for k in vans:
        model.addConstr(
            g.quicksum(x[k, 0, j] for j in customers if (k, 0, j) in x) == y[k],
            name=f"depot_out_{k}"
        )
        model.addConstr(
            g.quicksum(x[k, i, 0] for i in customers if (k, i, 0) in x) == y[k],
            name=f"depot_in_{k}"
        )

    # Constraint 4: Capacity per van
    for k in vans:
        model.addConstr(
            g.quicksum(
                s[i] * g.quicksum(x[k, j, i] for j in nodes if (k, j, i) in x)
                for i in customers
            ) <= Q,
            name=f"cap_{k}"
        )

    # Constraint 5: Time propagation with Big-M
    # Skip arcs returning to depot (j=0) — t[k,0] is departure time, not return time
    for (k, i, j) in x:
        if j == 0:
            continue  # no time constraint on return to depot
        # M[i][j] = max(0, tw_hi[i] + T[i][j] - tw_lo[j])
        ub_i = tw_hi[i] if i > 0 else tw_hi[0]
        lb_j = tw_lo[j]
        M = max(0, ub_i + T[i][j] - lb_j)
        model.addConstr(
            t[k, j] >= t[k, i] + T[i][j] - M * (1 - x[k, i, j]),
            name=f"time_{k}_{i}_{j}"
        )

    # Constraint 6: Symmetry breaking — order vans
    for k in range(K - 1):
        model.addConstr(y[k] >= y[k + 1], name=f"sym_{k}")

    # Solve
    model.optimize()

    if model.Status == g.GRB.INFEASIBLE:
        print("Model is infeasible!", file=sys.stderr)
        sys.exit(1)

    if model.SolCount == 0:
        print("No solution found!", file=sys.stderr)
        sys.exit(1)

    # Extract solution
    obj_val = model.ObjVal
    routes = []
    for k in vans:
        if y[k].X < 0.5:
            continue
        # Follow the route from depot
        route = []
        current = 0
        while True:
            next_node = None
            for j in nodes:
                if (k, current, j) in x and x[k, current, j].X > 0.5:
                    next_node = j
                    break
            if next_node is None or next_node == 0:
                break
            arrival = t[k, next_node].X
            # Snap arrival to tw_lo if it's just below due to floating point
            if arrival < tw_lo[next_node] - 1e-9:
                arrival = tw_lo[next_node]
            arrival = max(arrival, tw_lo[next_node])
            route.append((next_node, arrival))
            current = next_node
        if route:
            routes.append(route)

    return obj_val, routes


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    time_limit = float(sys.argv[3]) if len(sys.argv) > 3 else 300

    N, K, Q, Gamma, s, tw_lo, tw_hi, T, C = parse_input(input_file)
    obj_val, routes = solve_ilp(N, K, Q, Gamma, s, tw_lo, tw_hi, T, C, time_limit)

    # Validate before writing
    valid, msg = validate(N, K, Q, Gamma, s, tw_lo, tw_hi, T, C, obj_val, routes)
    if not valid:
        print(f"WARNING: Solution validation failed: {msg}", file=sys.stderr)

    write_output(output_file, obj_val, routes)


if __name__ == "__main__":
    main()
