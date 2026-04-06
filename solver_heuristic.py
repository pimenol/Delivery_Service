#!/usr/bin/env python3
"""CVRPTW Heuristic Solver (SP_CC_T / SP_CC_R) using ALNS."""
import sys
import math
import random
import time

# ─── Global problem data ───
N = K = Q = 0
Gamma = 0.0
s = []
tw_lo = []
tw_hi = []
T = []
C = []

start_time = 0.0
TIME_LIMIT = 300.0


def elapsed():
    return time.monotonic() - start_time


def time_remaining():
    return TIME_LIMIT - elapsed()


def parse_input(filepath):
    global N, K, Q, Gamma, s, tw_lo, tw_hi, T, C
    with open(filepath, 'r') as f:
        tokens = f.read().split()
    idx = 0

    def next_val():
        nonlocal idx
        val = tokens[idx]; idx += 1; return val

    N = int(next_val())
    K = int(next_val())
    Q = int(next_val())
    Gamma = float(next_val())

    s = [0] * (N + 1)
    tw_lo = [0.0] * (N + 1)
    tw_hi = [0.0] * (N + 1)
    for i in range(1, N + 1):
        s[i] = int(next_val())
        tw_lo[i] = float(next_val())
        tw_hi[i] = float(next_val())

    T = []
    for i in range(N + 1):
        row = [float(next_val()) for _ in range(N + 1)]
        T.append(row)

    C = []
    for i in range(N + 1):
        row = [float(next_val()) for _ in range(N + 1)]
        C.append(row)


# ─── Route class ───

class Route:
    __slots__ = ['customers', 'load', 'cost', 'arrival']

    def __init__(self, customers=None):
        self.customers = customers if customers is not None else []
        self.load = 0
        self.cost = 0.0
        self.arrival = []
        if self.customers:
            self.recompute()

    def recompute(self):
        self.load = 0
        self.cost = 0.0
        self.arrival = []
        prev = 0
        dep = 0.0
        for c in self.customers:
            self.load += s[c]
            arr = dep + T[prev][c]
            arr = max(arr, tw_lo[c])
            self.arrival.append(arr)
            self.cost += C[prev][c]
            dep = arr
            prev = c
        self.cost += C[prev][0]

    def is_feasible(self):
        if self.load > Q:
            return False
        for k, c in enumerate(self.customers):
            if self.arrival[k] > tw_hi[c] + 1e-9:
                return False
        return True

    def check_insert(self, cust, pos):
        """Check if inserting cust at pos is feasible. Returns (delta_cost, feasible)."""
        if self.load + s[cust] > Q:
            return float('inf'), False

        m = len(self.customers)

        # Predecessor departure
        if pos == 0:
            prev_node = 0
            prev_dep = 0.0
        else:
            prev_node = self.customers[pos - 1]
            prev_dep = self.arrival[pos - 1]

        # Arrival at new customer
        arr_c = prev_dep + T[prev_node][cust]
        arr_c = max(arr_c, tw_lo[cust])
        if arr_c > tw_hi[cust] + 1e-9:
            return float('inf'), False

        dep_c = arr_c

        # Cost delta
        if pos < m:
            next_node = self.customers[pos]
            old_cost_seg = C[prev_node][next_node]
            new_cost_seg = C[prev_node][cust] + C[cust][next_node]

            # Check push-forward propagation on subsequent customers
            new_arr_next = dep_c + T[cust][next_node]
            new_arr_next = max(new_arr_next, tw_lo[next_node])
            push = new_arr_next - self.arrival[pos]
            if push > 1e-9:
                for k in range(pos, m):
                    node_k = self.customers[k]
                    new_arr_k = self.arrival[k] + push
                    if new_arr_k > tw_hi[node_k] + 1e-9:
                        return float('inf'), False
                    # Absorption by wait time
                    new_dep_k = max(new_arr_k, tw_lo[node_k])
                    old_dep_k = self.arrival[k]  # arrival = max(physical_arrival, tw_lo), so departure = arrival
                    if k + 1 < m:
                        next_k = self.customers[k + 1]
                        old_arr_next = self.arrival[k + 1]
                        new_arr_next_k = new_dep_k + T[node_k][next_k]
                        new_arr_next_k = max(new_arr_next_k, tw_lo[next_k])
                        push = new_arr_next_k - old_arr_next
                        if push <= 1e-9:
                            break
            # Also need to update return-to-depot cost if last segment changes
            delta = new_cost_seg - old_cost_seg
        else:
            # Inserting at end
            old_cost_seg = C[prev_node][0]
            new_cost_seg = C[prev_node][cust] + C[cust][0]
            delta = new_cost_seg - old_cost_seg

        return delta, True

    def insert(self, cust, pos):
        self.customers.insert(pos, cust)
        self.recompute()

    def remove(self, pos):
        cust = self.customers.pop(pos)
        self.recompute()
        return cust

    def copy(self):
        r = Route.__new__(Route)
        r.customers = self.customers[:]
        r.load = self.load
        r.cost = self.cost
        r.arrival = self.arrival[:]
        return r


# ─── Solution class ───

class Solution:
    def __init__(self, routes=None):
        self.routes = routes if routes is not None else []
        self.unassigned = set()

    def objective(self):
        return sum(r.cost for r in self.routes) + Gamma * len(self.routes)

    def copy(self):
        sol = Solution()
        sol.routes = [r.copy() for r in self.routes]
        sol.unassigned = set(self.unassigned)
        return sol


# ─── Construction Heuristic (Solomon I1) ───

def construct_initial():
    sol = Solution()
    sol.unassigned = set(range(1, N + 1))

    while sol.unassigned:
        # Seed: farthest unassigned customer from depot (by cost)
        seed = max(sol.unassigned, key=lambda c: C[0][c])
        route = Route([seed])
        sol.unassigned.remove(seed)

        # Greedy insertion
        improved = True
        while improved:
            improved = False
            best_cust, best_pos, best_delta = None, None, float('inf')
            for c in sol.unassigned:
                for pos in range(len(route.customers) + 1):
                    delta, feas = route.check_insert(c, pos)
                    if feas and delta < best_delta:
                        best_cust, best_pos, best_delta = c, pos, delta
            if best_cust is not None:
                route.insert(best_cust, best_pos)
                sol.unassigned.remove(best_cust)
                improved = True

        sol.routes.append(route)

    return sol


# ─── Destroy Operators ───

def get_destroy_size():
    lo = max(1, int(0.1 * N))
    hi = max(lo + 1, min(int(0.4 * N), N))
    return random.randint(lo, hi)


def random_removal(sol):
    q = get_destroy_size()
    # Collect all customers in routes
    all_custs = []
    for ri, r in enumerate(sol.routes):
        for ci, c in enumerate(r.customers):
            all_custs.append((ri, ci, c))
    random.shuffle(all_custs)
    removed = []
    to_remove = {}  # ri -> sorted list of positions to remove (descending)
    for ri, ci, c in all_custs[:q]:
        removed.append(c)
        to_remove.setdefault(ri, []).append(ci)
    # Remove in reverse order to maintain indices
    for ri in to_remove:
        for ci in sorted(to_remove[ri], reverse=True):
            sol.routes[ri].remove(ci)
    sol.unassigned.update(removed)
    # Clean empty routes
    sol.routes = [r for r in sol.routes if r.customers]
    return removed


def worst_removal(sol):
    q = get_destroy_size()
    # Compute removal cost for each customer
    savings = []
    for ri, r in enumerate(sol.routes):
        custs = r.customers
        m = len(custs)
        for ci in range(m):
            c = custs[ci]
            prev = 0 if ci == 0 else custs[ci - 1]
            nxt = 0 if ci == m - 1 else custs[ci + 1]
            # Cost saving if removing c
            old_cost = C[prev][c] + C[c][nxt]
            new_cost = C[prev][nxt]
            saving = old_cost - new_cost
            savings.append((saving, ri, ci, c))

    # Randomized worst: sort by saving descending, pick with bias
    savings.sort(key=lambda x: -x[0])
    removed = []
    removed_set = set()
    to_remove = {}

    p = 3  # randomization exponent
    remaining = list(savings)
    while len(removed) < q and remaining:
        # Pick with probability proportional to rank^(-p)... simplified: use random index biased to front
        idx = int(len(remaining) * (random.random() ** p))
        idx = min(idx, len(remaining) - 1)
        _, ri, ci, c = remaining.pop(idx)
        if c not in removed_set:
            removed.append(c)
            removed_set.add(c)
            to_remove.setdefault(ri, []).append(ci)

    for ri in to_remove:
        for ci in sorted(to_remove[ri], reverse=True):
            sol.routes[ri].remove(ci)
    sol.unassigned.update(removed)
    sol.routes = [r for r in sol.routes if r.customers]
    return removed


def shaw_removal(sol):
    q = get_destroy_size()
    # Pick random seed customer
    all_custs = []
    for r in sol.routes:
        all_custs.extend(r.customers)
    if not all_custs:
        return []

    seed = random.choice(all_custs)
    # Compute relatedness (lower = more related)
    phi1, phi2, phi3 = 9, 3, 2
    # Normalize by max values
    max_dist = max(max(row) for row in C) or 1
    max_tw = max(tw_hi) - min(tw_lo[1:] if N > 0 else [0]) or 1
    max_s = max(s[1:]) - min(s[1:]) if N > 0 else 1
    max_s = max(max_s, 1)

    relatedness = []
    for c in all_custs:
        if c == seed:
            continue
        rel = (phi1 * C[seed][c] / max_dist +
               phi2 * abs(tw_lo[seed] - tw_lo[c]) / max_tw +
               phi3 * abs(s[seed] - s[c]) / max_s)
        relatedness.append((rel, c))

    relatedness.sort()
    to_remove_set = {seed}
    for _, c in relatedness:
        if len(to_remove_set) >= q:
            break
        to_remove_set.add(c)

    # Remove from routes
    for r in sol.routes:
        r.customers = [c for c in r.customers if c not in to_remove_set]
        r.recompute()
    sol.unassigned.update(to_remove_set)
    sol.routes = [r for r in sol.routes if r.customers]
    return list(to_remove_set)


def tw_removal(sol):
    q = get_destroy_size()
    all_custs = []
    for r in sol.routes:
        all_custs.extend(r.customers)
    if not all_custs:
        return []

    seed = random.choice(all_custs)
    # Sort by time window similarity
    by_tw = sorted(all_custs, key=lambda c: abs(tw_lo[c] - tw_lo[seed]))
    to_remove_set = set(by_tw[:q])

    for r in sol.routes:
        r.customers = [c for c in r.customers if c not in to_remove_set]
        r.recompute()
    sol.unassigned.update(to_remove_set)
    sol.routes = [r for r in sol.routes if r.customers]
    return list(to_remove_set)


# ─── Repair Operators ───

def greedy_insertion(sol):
    """Insert all unassigned customers greedily (cheapest insertion)."""
    unassigned = list(sol.unassigned)
    random.shuffle(unassigned)

    for c in unassigned:
        best_delta = float('inf')
        best_ri = -1
        best_pos = -1

        for ri, r in enumerate(sol.routes):
            for pos in range(len(r.customers) + 1):
                delta, feas = r.check_insert(c, pos)
                if feas and delta < best_delta:
                    best_delta = delta
                    best_ri = ri
                    best_pos = pos

        # Compare with new route
        new_route_cost = C[0][c] + C[c][0]
        new_route_delta = new_route_cost + Gamma  # cost of new route including van cost

        if best_ri >= 0 and best_delta < new_route_delta:
            sol.routes[best_ri].insert(c, best_pos)
        else:
            sol.routes.append(Route([c]))

        sol.unassigned.discard(c)


def regret2_insertion(sol):
    """Insert customers by regret-2 heuristic."""
    while sol.unassigned:
        best_customer = None
        best_regret = -float('inf')
        best_ri = -1
        best_pos = -1

        for c in sol.unassigned:
            insertions = []  # (total_delta, ri, pos)
            for ri, r in enumerate(sol.routes):
                for pos in range(len(r.customers) + 1):
                    delta, feas = r.check_insert(c, pos)
                    if feas:
                        insertions.append((delta, ri, pos))
            # New route option
            new_cost = C[0][c] + C[c][0] + Gamma
            insertions.append((new_cost, -1, 0))
            insertions.sort(key=lambda x: x[0])

            if len(insertions) >= 2:
                regret = insertions[1][0] - insertions[0][0]
            else:
                regret = float('inf')

            if regret > best_regret or (regret == best_regret and insertions[0][0] < best_ri):
                best_regret = regret
                best_customer = c
                best_ri = insertions[0][1]
                best_pos = insertions[0][2]

        if best_customer is None:
            break

        if best_ri == -1:
            sol.routes.append(Route([best_customer]))
        else:
            sol.routes[best_ri].insert(best_customer, best_pos)
        sol.unassigned.discard(best_customer)


# ─── Local Search ───

def local_search(sol):
    """Apply first-improvement local search operators."""
    improved = True
    max_iters = 5  # limit to avoid spending too much time
    iters = 0
    while improved and iters < max_iters and time_remaining() > 1.5:
        improved = False
        iters += 1

        # Intra-route relocate
        for r in sol.routes:
            if intra_relocate(r):
                improved = True

        # Inter-route relocate
        for i in range(len(sol.routes)):
            for j in range(len(sol.routes)):
                if i == j:
                    continue
                if inter_relocate(sol.routes[i], sol.routes[j]):
                    improved = True

        # Inter-route swap
        for i in range(len(sol.routes)):
            for j in range(i + 1, len(sol.routes)):
                if inter_swap(sol.routes[i], sol.routes[j]):
                    improved = True

        # Remove empty routes
        sol.routes = [r for r in sol.routes if r.customers]


def intra_relocate(route):
    """Try moving a customer to a better position within the same route."""
    m = len(route.customers)
    if m <= 1:
        return False
    best_improvement = 0
    best_from, best_to = -1, -1

    for i in range(m):
        c = route.customers[i]
        # Temporarily remove
        old_customers = route.customers[:]
        route.customers.pop(i)
        route.recompute()

        for j in range(len(route.customers) + 1):
            delta, feas = route.check_insert(c, j)
            if feas:
                # Total change: new cost - original cost
                new_cost = route.cost + delta
                orig_cost_with_c = Route(old_customers).cost
                improvement = orig_cost_with_c - (route.cost + delta)
                if improvement > best_improvement + 1e-9:
                    best_improvement = improvement
                    best_from = i
                    best_to = j

        route.customers = old_customers
        route.recompute()

    if best_from >= 0:
        c = route.customers.pop(best_from)
        # Adjust insertion index
        if best_to > best_from:
            best_to -= 1
        route.customers.insert(best_to, c)
        route.recompute()
        return True
    return False


def inter_relocate(r1, r2):
    """Try moving a customer from r1 to r2."""
    best_improvement = 0
    best_ci = -1
    best_pos = -1

    for ci in range(len(r1.customers)):
        c = r1.customers[ci]
        # Cost of removing from r1
        prev1 = 0 if ci == 0 else r1.customers[ci - 1]
        nxt1 = 0 if ci == len(r1.customers) - 1 else r1.customers[ci + 1]
        remove_saving = C[prev1][c] + C[c][nxt1] - C[prev1][nxt1]

        # Best insertion in r2
        for pos in range(len(r2.customers) + 1):
            delta, feas = r2.check_insert(c, pos)
            if feas:
                improvement = remove_saving - delta
                if improvement > best_improvement + 1e-9:
                    best_improvement = improvement
                    best_ci = ci
                    best_pos = pos

    if best_ci >= 0:
        c = r1.remove(best_ci)
        r2.insert(c, best_pos)
        return True
    return False


def inter_swap(r1, r2):
    """Try swapping one customer between two routes."""
    best_improvement = 0
    best_i, best_j = -1, -1

    for i in range(len(r1.customers)):
        c1 = r1.customers[i]
        for j in range(len(r2.customers)):
            c2 = r2.customers[j]

            # Check capacity
            new_load_1 = r1.load - s[c1] + s[c2]
            new_load_2 = r2.load - s[c2] + s[c1]
            if new_load_1 > Q or new_load_2 > Q:
                continue

            # Try the swap
            r1.customers[i] = c2
            r2.customers[j] = c1
            r1.recompute()
            r2.recompute()

            if r1.is_feasible() and r2.is_feasible():
                old_cost = Route(r1.customers[:i] + [c1] + r1.customers[i+1:]).cost + \
                           Route(r2.customers[:j] + [c2] + r2.customers[j+1:]).cost
                new_cost = r1.cost + r2.cost
                improvement = old_cost - new_cost
                if improvement > best_improvement + 1e-9:
                    best_improvement = improvement
                    best_i, best_j = i, j

            # Undo
            r1.customers[i] = c1
            r2.customers[j] = c2
            r1.recompute()
            r2.recompute()

    if best_i >= 0:
        c1 = r1.customers[best_i]
        c2 = r2.customers[best_j]
        r1.customers[best_i] = c2
        r2.customers[best_j] = c1
        r1.recompute()
        r2.recompute()
        return True
    return False


# ─── Van Minimization ───

def try_reduce_vans(sol):
    """Try to empty the smallest route by redistributing its customers."""
    if len(sol.routes) <= 1:
        return False

    sol.routes.sort(key=lambda r: len(r.customers))

    for ri in range(len(sol.routes)):
        route = sol.routes[ri]
        customers_to_move = route.customers[:]
        other_routes = [r.copy() for r in sol.routes if r is not route]

        success = True
        for c in sorted(customers_to_move, key=lambda c: tw_hi[c]):
            inserted = False
            best_delta, best_rj, best_pos = float('inf'), -1, -1
            for rj, r in enumerate(other_routes):
                for pos in range(len(r.customers) + 1):
                    delta, feas = r.check_insert(c, pos)
                    if feas and delta < best_delta:
                        best_delta, best_rj, best_pos = delta, rj, pos
                        inserted = True
            if inserted:
                other_routes[best_rj].insert(c, best_pos)
            else:
                success = False
                break

        if success:
            # Check if objective improves
            old_obj = sol.objective()
            new_obj = sum(r.cost for r in other_routes) + Gamma * len(other_routes)
            if new_obj < old_obj - 1e-9:
                sol.routes = other_routes
                return True

    return False


# ─── ALNS Main Loop ───

def solve():
    global start_time, TIME_LIMIT
    start_time = time.monotonic()

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    if len(sys.argv) > 3:
        TIME_LIMIT = float(sys.argv[3])

    parse_input(input_file)

    # Construction
    current = construct_initial()
    best = current.copy()
    best_obj = best.objective()

    # ALNS setup
    destroy_ops = [random_removal, worst_removal, shaw_removal, tw_removal]
    repair_ops = [greedy_insertion, regret2_insertion]

    n_d = len(destroy_ops)
    n_r = len(repair_ops)
    d_weights = [1.0] * n_d
    r_weights = [1.0] * n_r
    d_scores = [0.0] * n_d
    r_scores = [0.0] * n_r
    d_uses = [0] * n_d
    r_uses = [0] * n_r

    SIGMA_1 = 33  # new best
    SIGMA_2 = 9   # improving
    SIGMA_3 = 2   # accepted
    DECAY = 0.1
    SEGMENT_SIZE = 100

    # SA parameters
    init_obj = current.objective()
    temperature = 0.05 * max(init_obj, 1)
    cooling = 0.9997

    iteration = 0
    segment_iter = 0

    def select_op(weights):
        total = sum(weights)
        r = random.random() * total
        cum = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(weights) - 1

    while time_remaining() > 1.0:
        di = select_op(d_weights)
        ri = select_op(r_weights)

        candidate = current.copy()
        destroy_ops[di](candidate)
        repair_ops[ri](candidate)

        cand_obj = candidate.objective()
        curr_obj = current.objective()
        delta = cand_obj - curr_obj

        score = 0
        if cand_obj < best_obj - 1e-9:
            score = SIGMA_1
            best = candidate.copy()
            best_obj = cand_obj
        elif delta < -1e-9:
            score = SIGMA_2
        elif delta < 1e-9 or random.random() < math.exp(-delta / max(temperature, 1e-10)):
            score = SIGMA_3
        else:
            score = 0

        if score > 0:
            current = candidate

        d_scores[di] += score
        r_scores[ri] += score
        d_uses[di] += 1
        r_uses[ri] += 1

        # Periodic local search
        if iteration % 50 == 0 and time_remaining() > 2.0:
            local_search(current)
            curr_obj = current.objective()
            if curr_obj < best_obj - 1e-9:
                best = current.copy()
                best_obj = curr_obj

        # Periodic van reduction
        if iteration % 200 == 0 and time_remaining() > 2.0:
            if try_reduce_vans(current):
                curr_obj = current.objective()
                if curr_obj < best_obj - 1e-9:
                    best = current.copy()
                    best_obj = curr_obj

        # Weight update
        segment_iter += 1
        if segment_iter >= SEGMENT_SIZE:
            for i in range(n_d):
                if d_uses[i] > 0:
                    d_weights[i] = d_weights[i] * (1 - DECAY) + DECAY * d_scores[i] / d_uses[i]
                d_weights[i] = max(d_weights[i], 0.01)
                d_scores[i] = 0
                d_uses[i] = 0
            for i in range(n_r):
                if r_uses[i] > 0:
                    r_weights[i] = r_weights[i] * (1 - DECAY) + DECAY * r_scores[i] / r_uses[i]
                r_weights[i] = max(r_weights[i], 0.01)
                r_scores[i] = 0
                r_uses[i] = 0
            segment_iter = 0

        temperature *= cooling
        iteration += 1

    # Final local search on best
    if time_remaining() > 0.8:
        current = best.copy()
        local_search(current)
        if current.objective() < best_obj:
            best = current

    # Final van reduction
    if time_remaining() > 0.5:
        try_reduce_vans(best)

    # Write output
    write_output(output_file, best)


def write_output(filepath, sol):
    obj = sol.objective()
    routes = sol.routes
    lines = []
    if obj == int(obj):
        lines.append(f"{obj:.1f} {len(routes)}")
    else:
        lines.append(f"{obj} {len(routes)}")

    for r in routes:
        parts = [str(len(r.customers))]
        for k, c in enumerate(r.customers):
            arr = r.arrival[k]
            if arr == int(arr):
                parts.append(f"{c} {int(arr)}")
            else:
                parts.append(f"{c} {arr}")
        lines.append(' '.join(parts))

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == "__main__":
    solve()
