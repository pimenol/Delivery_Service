#!/usr/bin/env python3
"""CVRPTW Heuristic Solver (SP_CC_T / SP_CC_R) — ALNS with FTS + string removal."""
import sys
import math
import time
import numpy as np

N = K = Q = 0
Gamma = 0.0
dem = None       # demand[i]
tw_low = None    # tw_low[i]
tw_high = None   # tw_high[i]
ttime = None     # travel time matrix
cost_mat = None  # cost matrix
neighbors = None # neighbors[i] = sorted indices by distance

start_time = 0.0
TIME_LIMIT = 300.0
rng = np.random.default_rng(42)


def elapsed():
    return time.monotonic() - start_time

def time_remaining():
    return TIME_LIMIT - elapsed()

def parse_input(filepath):
    global N, K, Q, Gamma, dem, tw_low, tw_high, ttime, cost_mat, neighbors
    with open(filepath, 'r') as f:
        tokens = f.read().split()
    idx = 0

    def nxt():
        nonlocal idx
        v = tokens[idx]; idx += 1; return v

    N = int(nxt()); K = int(nxt()); Q = int(nxt()); Gamma = float(nxt())

    dem = np.zeros(N + 1, dtype=np.float64)
    tw_low = np.zeros(N + 1, dtype=np.float64)
    tw_high = np.zeros(N + 1, dtype=np.float64)
    for i in range(1, N + 1):
        dem[i] = int(nxt())
        tw_low[i] = float(nxt())
        tw_high[i] = float(nxt())

    ttime = np.zeros((N + 1, N + 1), dtype=np.float64)
    for i in range(N + 1):
        for j in range(N + 1):
            ttime[i][j] = float(nxt())

    cost_mat = np.zeros((N + 1, N + 1), dtype=np.float64)
    for i in range(N + 1):
        for j in range(N + 1):
            cost_mat[i][j] = float(nxt())

    tw_high[0] = max(tw_high[i] + ttime[i][0] for i in range(1, N + 1))
    neighbors = np.argsort(cost_mat, axis=1)


class Route:
    __slots__ = ['customers', 'load', 'cost', 'earliest', 'fts', 'waiting']

    def __init__(self, customers=None):
        self.customers = customers if customers is not None else []
        self.load = 0
        self.cost = 0.0
        self.earliest = []
        self.fts = []
        self.waiting = []
        if self.customers:
            self.recompute()

    def recompute(self):
        custs = self.customers
        m = len(custs)
        self.load = sum(dem[c] for c in custs)

        self.earliest = [0.0] * m
        self.waiting = [0.0] * m
        self.cost = 0.0

        prev = 0
        dep = 0.0
        for i in range(m):
            c = custs[i]
            phys_arr = dep + ttime[prev][c]
            self.earliest[i] = max(phys_arr, tw_low[c])
            self.waiting[i] = max(0.0, tw_low[c] - phys_arr)
            self.cost += cost_mat[prev][c]
            dep = self.earliest[i]
            prev = c
        self.cost += cost_mat[prev][0]

        self.fts = [0.0] * m
        if m > 0:
            self.fts[m - 1] = tw_high[custs[m - 1]] - self.earliest[m - 1]
            for i in range(m - 2, -1, -1):
                self.fts[i] = min(
                    tw_high[custs[i]] - self.earliest[i],
                    self.waiting[i + 1] + self.fts[i + 1]
                )

    def is_feasible(self):
        if self.load > Q:
            return False
        for k, c in enumerate(self.customers):
            if self.earliest[k] > tw_high[c] + 1e-9:
                return False
        return True

    def can_insert(self, pos, u):
        m = len(self.customers)

        if self.load + dem[u] > Q:
            return False, float('inf')

        if pos == 0:
            prev_node = 0
            prev_dep = 0.0
        else:
            prev_node = self.customers[pos - 1]
            prev_dep = self.earliest[pos - 1]

        arr_u = prev_dep + ttime[prev_node][u]
        start_u = max(arr_u, tw_low[u])
        if start_u > tw_high[u] + 1e-9:
            return False, float('inf')

        if pos < m:
            next_node = self.customers[pos]
            new_arr_next = start_u + ttime[u][next_node]
            push = max(0.0, new_arr_next - self.earliest[pos])
            if push > self.fts[pos] + 1e-9:
                return False, float('inf')
            delta = cost_mat[prev_node][u] + cost_mat[u][next_node] - cost_mat[prev_node][next_node]
        else:
            delta = cost_mat[prev_node][u] + cost_mat[u][0] - cost_mat[prev_node][0]

        return True, delta

    def insert(self, pos, u):
        self.customers.insert(pos, u)
        self.recompute()

    def remove_at(self, pos):
        c = self.customers.pop(pos)
        self.recompute()
        return c

    def copy(self):
        r = Route.__new__(Route)
        r.customers = self.customers[:]
        r.load = self.load
        r.cost = self.cost
        r.earliest = self.earliest[:]
        r.fts = self.fts[:]
        r.waiting = self.waiting[:]
        return r

class Solution:
    __slots__ = ['routes', 'total_cost', 'num_vans']

    def __init__(self):
        self.routes = []
        self.total_cost = 0.0
        self.num_vans = 0

    def recompute_objective(self):
        self.num_vans = len(self.routes)
        self.total_cost = sum(r.cost for r in self.routes) + Gamma * self.num_vans

    def copy(self):
        sol = Solution()
        sol.routes = [r.copy() for r in self.routes]
        sol.total_cost = self.total_cost
        sol.num_vans = self.num_vans
        return sol

    def find_customer(self, c):
        """Returns (route_idx, position) for customer c."""
        for ri, r in enumerate(self.routes):
            for ci, cust in enumerate(r.customers):
                if cust == c:
                    return ri, ci
        return -1, -1

    def all_customers(self):
        result = []
        for r in self.routes:
            result.extend(r.customers)
        return result

def construct_regret2():
    sol = Solution()
    unassigned = set(range(1, N + 1))

    sol.routes = [Route()]

    while unassigned:
        best_u = None
        best_regret = -float('inf')
        best_ri = -1
        best_pos = -1
        best_delta = float('inf')

        for u in unassigned:
            insertions = []  # (delta, ri, pos)
            for ri, r in enumerate(sol.routes):
                for pos in range(len(r.customers) + 1):
                    feas, delta = r.can_insert(pos, u)
                    if feas:
                        insertions.append((delta, ri, pos))

            # Option: new route
            new_route_cost = cost_mat[0][u] + cost_mat[u][0] + Gamma
            insertions.append((new_route_cost, -1, 0))
            insertions.sort(key=lambda x: x[0])

            if len(insertions) >= 2:
                regret = insertions[1][0] - insertions[0][0]
            else:
                regret = float('inf')

            if regret > best_regret or (abs(regret - best_regret) < 1e-9 and insertions[0][0] < best_delta):
                best_regret = regret
                best_u = u
                best_ri = insertions[0][1]
                best_pos = insertions[0][2]
                best_delta = insertions[0][0]

        if best_u is None:
            break

        if best_ri == -1:
            sol.routes.append(Route([best_u]))
        else:
            sol.routes[best_ri].insert(best_pos, best_u)
        unassigned.discard(best_u)

    sol.routes = [r for r in sol.routes if r.customers]
    sol.recompute_objective()
    return sol

def get_destroy_size():
    lo = max(1, int(0.1 * N))
    hi = max(lo + 1, min(int(0.4 * N), N))
    return rng.integers(lo, hi + 1)


def random_removal(sol, removed_set):
    q = get_destroy_size()
    all_custs = sol.all_customers()
    if q > len(all_custs):
        q = len(all_custs)
    chosen = rng.choice(all_custs, size=q, replace=False)
    removed_set.update(chosen)
    _remove_customers(sol, removed_set)


def worst_removal(sol, removed_set, p=3):
    q = get_destroy_size()
    for _ in range(q):
        savings = []
        for r in sol.routes:
            custs = r.customers
            m = len(custs)
            for ci in range(m):
                c = custs[ci]
                if c in removed_set:
                    continue
                prev = 0 if ci == 0 else custs[ci - 1]
                nxt = 0 if ci == m - 1 else custs[ci + 1]
                saving = cost_mat[prev][c] + cost_mat[c][nxt] - cost_mat[prev][nxt]
                savings.append((saving, c))
        if not savings:
            break
        savings.sort(key=lambda x: -x[0])
        idx = int(rng.random() ** p * len(savings))
        idx = min(idx, len(savings) - 1)
        removed_set.add(savings[idx][1])

    _remove_customers(sol, removed_set)


def shaw_removal(sol, removed_set, p=6):
    q = get_destroy_size()
    all_custs = [c for c in sol.all_customers() if c not in removed_set]
    if not all_custs:
        return

    seed = int(rng.choice(all_custs))
    to_remove = {seed}

    max_cost = float(np.max(cost_mat)) or 1.0
    max_tw = float(np.max(tw_high[1:])) - float(np.min(tw_low[1:])) or 1.0
    max_dem = float(np.max(dem[1:])) - float(np.min(dem[1:])) or 1.0

    relatedness = []
    for c in all_custs:
        if c == seed:
            continue
        rel = (9 * cost_mat[seed][c] / max_cost +
               3 * abs(tw_low[seed] - tw_low[c]) / max_tw +
               2 * abs(dem[seed] - dem[c]) / max_dem)
        relatedness.append((rel, c))
    relatedness.sort()

    for _, c in relatedness:
        if len(to_remove) >= q:
            break
        if rng.random() ** p < 0.5:
            to_remove.add(c)

    if len(to_remove) < q:
        for _, c in relatedness:
            if c not in to_remove:
                to_remove.add(c)
            if len(to_remove) >= q:
                break

    removed_set.update(to_remove)
    _remove_customers(sol, removed_set)


def string_removal(sol, removed_set):
    q = get_destroy_size()
    all_custs = [c for c in sol.all_customers() if c not in removed_set]
    if not all_custs:
        return

    seed = int(rng.choice(all_custs))

    avg_route_len = np.mean([len(r.customers) for r in sol.routes]) if sol.routes else 1
    l_max = min(10, max(1, int(avg_route_len)))
    k_max = max(1, int(4 * q / (1 + l_max)) - 1)
    k = rng.integers(1, k_max + 1)

    ruined_routes = set()
    removed = []

    for nb_idx in range(1, N + 1):
        if len(ruined_routes) >= k or len(removed) >= q:
            break
        c = int(neighbors[seed][nb_idx])
        if c == 0 or c in removed_set:
            continue

        ri, ci = sol.find_customer(c)
        if ri < 0 or ri in ruined_routes:
            continue

        r = sol.routes[ri]
        m = len(r.customers)
        l = rng.integers(1, min(l_max, m) + 1)
        start = max(0, ci - l + 1)
        start = rng.integers(start, min(ci, m - l) + 1)
        end = start + l
        string = r.customers[start:end]
        removed.extend(string)
        ruined_routes.add(ri)

    removed_set.update(removed[:q])
    _remove_customers(sol, removed_set)


def _remove_customers(sol, removed_set):
    for r in sol.routes:
        r.customers = [c for c in r.customers if c not in removed_set]
        r.recompute()
    sol.routes = [r for r in sol.routes if r.customers]


def greedy_insertion(sol, removed):
    removed_list = list(removed)
    rng.shuffle(removed_list)

    for u in removed_list:
        best_delta = float('inf')
        best_ri = -1
        best_pos = -1

        for ri, r in enumerate(sol.routes):
            for pos in range(len(r.customers) + 1):
                feas, delta = r.can_insert(pos, u)
                if feas and delta < best_delta:
                    best_delta = delta
                    best_ri = ri
                    best_pos = pos

        new_route_cost = cost_mat[0][u] + cost_mat[u][0] + Gamma
        if best_ri >= 0 and best_delta < new_route_cost:
            sol.routes[best_ri].insert(best_pos, u)
        else:
            sol.routes.append(Route([u]))

    removed.clear()
    sol.recompute_objective()


def regret2_insertion(sol, removed):
    unassigned = set(removed)

    while unassigned:
        best_u = None
        best_regret = -float('inf')
        best_ri = -1
        best_pos = -1
        best_delta = float('inf')

        for u in unassigned:
            insertions = []
            for ri, r in enumerate(sol.routes):
                for pos in range(len(r.customers) + 1):
                    feas, delta = r.can_insert(pos, u)
                    if feas:
                        insertions.append((delta, ri, pos))

            new_route_cost = cost_mat[0][u] + cost_mat[u][0] + Gamma
            insertions.append((new_route_cost, -1, 0))
            insertions.sort(key=lambda x: x[0])

            regret = insertions[1][0] - insertions[0][0] if len(insertions) >= 2 else float('inf')

            if regret > best_regret or (abs(regret - best_regret) < 1e-9 and insertions[0][0] < best_delta):
                best_regret = regret
                best_u = u
                best_ri = insertions[0][1]
                best_pos = insertions[0][2]
                best_delta = insertions[0][0]

        if best_u is None:
            break

        if best_ri == -1:
            sol.routes.append(Route([best_u]))
        else:
            sol.routes[best_ri].insert(best_pos, best_u)
        unassigned.discard(best_u)

    removed.clear()
    sol.recompute_objective()


def noisy_greedy_insertion(sol, removed):
    d_max = float(np.max(cost_mat))
    eta = 0.025
    removed_list = list(removed)
    rng.shuffle(removed_list)

    for u in removed_list:
        best_delta = float('inf')
        best_ri = -1
        best_pos = -1

        for ri, r in enumerate(sol.routes):
            for pos in range(len(r.customers) + 1):
                feas, delta = r.can_insert(pos, u)
                if feas:
                    noisy_delta = delta + rng.uniform(-1, 1) * eta * d_max
                    if noisy_delta < best_delta:
                        best_delta = noisy_delta
                        best_ri = ri
                        best_pos = pos

        new_route_cost = cost_mat[0][u] + cost_mat[u][0] + Gamma
        noisy_new = new_route_cost + rng.uniform(-1, 1) * eta * d_max
        if best_ri >= 0 and best_delta < noisy_new:
            sol.routes[best_ri].insert(best_pos, u)
        else:
            sol.routes.append(Route([u]))

    removed.clear()
    sol.recompute_objective()

def local_search(sol):
    improved = True
    max_passes = 3
    passes = 0
    while improved and passes < max_passes and time_remaining() > 1.5:
        improved = False
        passes += 1

        for r in sol.routes:
            if _intra_relocate(r):
                improved = True

        n_routes = len(sol.routes)
        for i in range(n_routes):
            for j in range(n_routes):
                if i == j or not sol.routes[i].customers:
                    continue
                if _inter_relocate(sol.routes[i], sol.routes[j]):
                    improved = True

        for i in range(n_routes):
            for j in range(i + 1, n_routes):
                if not sol.routes[i].customers or not sol.routes[j].customers:
                    continue
                if _inter_swap(sol.routes[i], sol.routes[j]):
                    improved = True

        sol.routes = [r for r in sol.routes if r.customers]
    sol.recompute_objective()


def _intra_relocate(route):
    m = len(route.customers)
    if m <= 1:
        return False
    original_cost = route.cost

    for i in range(m):
        c = route.customers[i]
        # Remove c temporarily
        route.customers.pop(i)
        route.recompute()

        best_pos = -1
        best_delta = float('inf')
        for j in range(len(route.customers) + 1):
            feas, delta = route.can_insert(j, c)
            if feas and delta < best_delta:
                best_delta = delta
                best_pos = j

        if best_pos >= 0 and (route.cost + best_delta) < original_cost - 1e-9:
            route.insert(best_pos, c)
            return True
        else:
            # Restore
            route.customers.insert(i, c)
            route.recompute()

    return False


def _inter_relocate(r1, r2):
    for ci in range(len(r1.customers)):
        c = r1.customers[ci]
        # Removal saving from r1
        prev = 0 if ci == 0 else r1.customers[ci - 1]
        nxt = 0 if ci == len(r1.customers) - 1 else r1.customers[ci + 1]
        remove_saving = cost_mat[prev][c] + cost_mat[c][nxt] - cost_mat[prev][nxt]

        # Best insertion in r2
        for pos in range(len(r2.customers) + 1):
            feas, delta = r2.can_insert(pos, c)
            if feas and remove_saving - delta > 1e-9:
                r1.customers.pop(ci)
                r1.recompute()
                r2.insert(pos, c)
                return True
    return False


def _inter_swap(r1, r2):
    for i in range(len(r1.customers)):
        c1 = r1.customers[i]
        for j in range(len(r2.customers)):
            c2 = r2.customers[j]

            new_load_1 = r1.load - dem[c1] + dem[c2]
            new_load_2 = r2.load - dem[c2] + dem[c1]
            if new_load_1 > Q or new_load_2 > Q:
                continue

            old_cost = r1.cost + r2.cost
            r1.customers[i] = c2
            r2.customers[j] = c1
            r1.recompute()
            r2.recompute()

            if r1.is_feasible() and r2.is_feasible() and r1.cost + r2.cost < old_cost - 1e-9:
                return True

            # Undo
            r1.customers[i] = c1
            r2.customers[j] = c2
            r1.recompute()
            r2.recompute()
    return False

def try_reduce_vans(sol):
    if len(sol.routes) <= 1:
        return False

    sol.routes.sort(key=lambda r: len(r.customers))

    for ri in range(min(3, len(sol.routes))):
        route = sol.routes[ri]
        customers_to_move = route.customers[:]
        other_routes = [r.copy() for r in sol.routes if r is not route]

        success = True
        for c in sorted(customers_to_move, key=lambda c: tw_high[c]):
            inserted = False
            best_delta, best_rj, best_pos = float('inf'), -1, -1
            for rj, r in enumerate(other_routes):
                for pos in range(len(r.customers) + 1):
                    feas, delta = r.can_insert(pos, c)
                    if feas and delta < best_delta:
                        best_delta = delta
                        best_rj = rj
                        best_pos = pos
                        inserted = True
            if inserted:
                other_routes[best_rj].insert(best_pos, c)
            else:
                success = False
                break

        if success:
            old_obj = sol.total_cost
            new_obj = sum(r.cost for r in other_routes) + Gamma * len(other_routes)
            if new_obj < old_obj - 1e-9:
                sol.routes = other_routes
                sol.recompute_objective()
                return True

    return False

def solve():
    global start_time, TIME_LIMIT
    start_time = time.monotonic()

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    if len(sys.argv) > 3:
        TIME_LIMIT = float(sys.argv[3])

    parse_input(input_file)

    # Construction
    current = construct_regret2()
    best = current.copy()
    best_obj = best.total_cost

    # ALNS operator lists
    destroy_ops = [random_removal, worst_removal, shaw_removal, string_removal]
    repair_ops = [greedy_insertion, regret2_insertion, noisy_greedy_insertion]

    n_d = len(destroy_ops)
    n_r = len(repair_ops)
    d_weights = [1.0] * n_d
    r_weights = [1.0] * n_r
    d_scores = [0.0] * n_d
    r_scores = [0.0] * n_r
    d_uses = [0] * n_d
    r_uses = [0] * n_r

    SIGMA_1 = 33   # new global best
    SIGMA_2 = 9    # improving, accepted
    SIGMA_3 = 13   # worse, accepted
    DECAY = 0.1
    SEGMENT_SIZE = 100

    init_obj = max(current.total_cost, 1e-6)
    temperature = -0.05 * init_obj / math.log(0.5)
    cooling = 0.99975

    iteration = 0
    segment_iter = 0

    def select_op(weights):
        total = sum(weights)
        r = rng.random() * total
        cum = 0
        for i, w in enumerate(weights):
            cum += w
            if r <= cum:
                return i
        return len(weights) - 1

    # ALNS main loop
    while time_remaining() > 1.5:
        di = select_op(d_weights)
        ri_op = select_op(r_weights)

        candidate = current.copy()
        removed = set()
        destroy_ops[di](candidate, removed)
        repair_ops[ri_op](candidate, removed)

        cand_obj = candidate.total_cost
        curr_obj = current.total_cost
        delta = cand_obj - curr_obj

        score = 0
        if cand_obj < best_obj - 1e-9:
            score = SIGMA_1
            best = candidate.copy()
            best_obj = cand_obj
        elif delta < -1e-9:
            score = SIGMA_2
        elif delta < 1e-9 or rng.random() < math.exp(-delta / max(temperature, 1e-10)):
            score = SIGMA_3
        else:
            score = 0

        if score > 0:
            current = candidate

        d_scores[di] += score
        r_scores[ri_op] += score
        d_uses[di] += 1
        r_uses[ri_op] += 1

        # Periodic local search
        if iteration % 50 == 0 and time_remaining() > 2.0:
            local_search(current)
            if current.total_cost < best_obj - 1e-9:
                best = current.copy()
                best_obj = best.total_cost

        # Periodic van reduction
        if iteration % 500 == 0 and time_remaining() > 2.0:
            if try_reduce_vans(current):
                if current.total_cost < best_obj - 1e-9:
                    best = current.copy()
                    best_obj = best.total_cost

        # Adaptive weight update
        segment_iter += 1
        if segment_iter >= SEGMENT_SIZE:
            for i in range(n_d):
                if d_uses[i] > 0:
                    d_weights[i] = d_weights[i] * (1 - DECAY) + DECAY * d_scores[i] / d_uses[i]
                d_weights[i] = max(d_weights[i], 0.01)
                d_scores[i] = 0; d_uses[i] = 0
            for i in range(n_r):
                if r_uses[i] > 0:
                    r_weights[i] = r_weights[i] * (1 - DECAY) + DECAY * r_scores[i] / r_uses[i]
                r_weights[i] = max(r_weights[i], 0.01)
                r_scores[i] = 0; r_uses[i] = 0
            segment_iter = 0

        # Adaptive cooling rate after warmup
        if iteration == 200:
            rate = 200 / max(elapsed(), 0.01)
            est_remaining = int(rate * time_remaining())
            if est_remaining > 0:
                cooling = 0.01 ** (1.0 / est_remaining)

        temperature *= cooling
        iteration += 1

    # Final polish
    if time_remaining() > 1.0:
        current = best.copy()
        local_search(current)
        if current.total_cost < best_obj:
            best = current
            best_obj = best.total_cost

    if time_remaining() > 0.5:
        try_reduce_vans(best)
        best_obj = best.total_cost

    write_output(output_file, best)


def write_output(filepath, sol):
    obj = sol.total_cost
    routes = sol.routes
    lines = []
    if obj == int(obj):
        lines.append(f"{obj:.1f} {len(routes)}")
    else:
        lines.append(f"{obj} {len(routes)}")

    for r in routes:
        parts = [str(len(r.customers))]
        for k, c in enumerate(r.customers):
            arr = r.earliest[k]
            if arr == int(arr):
                parts.append(f"{c} {int(arr)}")
            else:
                parts.append(f"{c} {arr}")
        lines.append(' '.join(parts))

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines) + '\n')


if __name__ == "__main__":
    solve()
