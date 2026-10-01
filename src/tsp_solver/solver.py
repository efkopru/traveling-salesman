"""
Traveling Salesman Problem (TSP) Solver
========================================
A comprehensive implementation of multiple TSP algorithms with visualization support.

Author: Esad Kopru
License: MIT
"""

import math
import os
import time
import random
import numpy as np
from itertools import chain, permutations
from typing import List, Tuple, Dict, Optional, Union

from .tsplib import load_tsplib

# Moves must improve the tour by more than this to count, which stops
# local search from cycling on floating-point noise.
EPSILON = 1e-10


def _tsplib_nint(x: np.ndarray) -> np.ndarray:
    """TSPLIB nint(): round half up, i.e. (int)(x + 0.5)."""
    return np.floor(x + 0.5)


def _att_distance(squared: np.ndarray) -> np.ndarray:
    """TSPLIB ATT pseudo-Euclidean distance."""
    r = np.sqrt(squared / 10.0)
    t = _tsplib_nint(r)
    return np.where(t < r, t + 1, t)


# Distance from squared Euclidean distance, per convention.
DISTANCE_FUNCTIONS = {
    'euclidean': np.sqrt,
    'EUC_2D': lambda sq: _tsplib_nint(np.sqrt(sq)),
    'CEIL_2D': lambda sq: np.ceil(np.sqrt(sq)),
    'ATT': _att_distance,
}


class TSPSolver:
    """
    A comprehensive TSP solver implementing multiple algorithms.

    Algorithms included:
    - Brute Force (exact solution for small instances)
    - Held-Karp dynamic programming (exact, up to ~20 cities)
    - Nearest Neighbor (greedy heuristic)
    - Nearest Insertion (constructive heuristic)
    - 2-Opt (local search improvement)
    - 3-Opt (local search improvement)
    - Or-Opt and 2-Opt + Or-Opt (local search improvement)
    - Iterated Local Search (2-opt with double-bridge kicks)
    - Simulated Annealing (metaheuristic)
    - Genetic Algorithm (evolutionary approach)
    """

    def __init__(self, cities: np.ndarray, city_names: Optional[List[str]] = None,
                 seed: Optional[int] = None, distance: str = 'euclidean'):
        """
        Initialize TSP solver with city coordinates.

        Args:
            cities: Array of shape (n, 2) with city coordinates
            city_names: Optional list of city names
            seed: Optional seed for the randomized algorithms (simulated
                annealing, genetic algorithm, iterated local search), for
                reproducible results
            distance: 'euclidean' (default), or a TSPLIB convention:
                'EUC_2D' (rounded to nearest integer), 'CEIL_2D' (rounded
                up) or 'ATT' (pseudo-Euclidean). Use the TSPLIB convention
                to compare against published optimal tour lengths.
        """
        if distance not in DISTANCE_FUNCTIONS:
            raise ValueError(f"Unknown distance '{distance}'. "
                             f"Choose from {list(DISTANCE_FUNCTIONS)}")
        self.distance = distance
        self.cities = np.asarray(cities, dtype=float)
        if self.cities.ndim != 2 or self.cities.shape[1] != 2 or len(self.cities) == 0:
            raise ValueError(f"cities must be a non-empty array of shape (n, 2), "
                             f"got shape {self.cities.shape}")
        self.n_cities = len(self.cities)
        if city_names is not None and len(city_names) != self.n_cities:
            raise ValueError(f"{len(city_names)} city names for "
                             f"{self.n_cities} cities")
        self.city_names = city_names or [f"City_{i}" for i in range(self.n_cities)]
        self.seed = seed
        self.rng = random.Random(seed)
        self.distance_matrix = self._calculate_distance_matrix()
        # Plain nested lists are much faster than numpy for scalar lookups
        # inside the Python loops below.
        self._dist = self.distance_matrix.tolist()

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate the distance matrix between all cities."""
        diff = self.cities[:, np.newaxis, :] - self.cities[np.newaxis, :, :]
        return DISTANCE_FUNCTIONS[self.distance]((diff ** 2).sum(axis=-1))

    @classmethod
    def from_tsplib(cls, source: Union[str, os.PathLike, Dict],
                    seed: Optional[int] = None) -> 'TSPSolver':
        """
        Create a solver from a TSPLIB instance, using its distance
        convention (EDGE_WEIGHT_TYPE).

        Args:
            source: Path to a .tsp file, or an instance already read with
                load_tsplib() (to keep its metadata, e.g. 'optimum')
            seed: Seed for the randomized algorithms
        """
        instance = source if isinstance(source, dict) else load_tsplib(source)
        return cls(instance['coordinates'], instance['city_names'], seed=seed,
                   distance=instance['edge_weight_type'])

    def calculate_tour_distance(self, tour: List[int]) -> float:
        """Calculate total distance of a tour."""
        D = self._dist
        n = len(tour)
        return sum(D[tour[i]][tour[(i + 1) % n]] for i in range(n))

    # ==================== EXACT ALGORITHM ====================

    def brute_force(self, start_city: int = 0) -> Tuple[List[int], float]:
        """
        Find optimal solution using brute force (only for small instances).

        Time Complexity: O(n!)
        Use only for n <= 10 cities
        """
        if self.n_cities > 10:
            raise ValueError("Brute force only feasible for <= 10 cities")

        other_cities = [i for i in range(self.n_cities) if i != start_city]
        min_distance = float('inf')
        best_tour = [start_city]

        for perm in permutations(other_cities):
            # A tour and its reverse have the same length; check one of them.
            if len(perm) > 1 and perm[0] > perm[-1]:
                continue
            tour = [start_city] + list(perm)
            distance = self.calculate_tour_distance(tour)
            if distance < min_distance:
                min_distance = distance
                best_tour = tour

        return best_tour, self.calculate_tour_distance(best_tour)

    def held_karp(self, max_cities: int = 20) -> Tuple[List[int], float]:
        """
        Exact solution by Held-Karp dynamic programming.

        dp[S][j] is the shortest path that starts at city 0, visits exactly
        the cities in subset S, and ends at j. Subsets are processed in
        order of size, vectorized with NumPy.

        Time Complexity: O(n² · 2ⁿ); memory O(n · 2ⁿ) (about 90 MB at
        n = 20).

        Args:
            max_cities: Refuse larger instances (memory grows as 2ⁿ)
        """
        n = self.n_cities
        if n > max_cities:
            raise ValueError(f"Held-Karp limited to <= {max_cities} cities "
                             f"(got {n}); memory grows as n * 2^n")
        if n <= 3:
            tour = list(range(n))
            return tour, self.calculate_tour_distance(tour)

        D = self.distance_matrix
        m = n - 1                      # cities 1..n-1 map to bits 0..m-1
        size = 1 << m
        to_city = D[1:, 1:]            # to_city[k, j] = D[k+1][j+1]

        dp = np.full((size, m), np.inf)
        parent = np.full((size, m), -1, dtype=np.int8)
        for j in range(m):
            dp[1 << j, j] = D[0, j + 1]

        masks = np.arange(size)
        popcount = np.zeros(size, dtype=np.int8)
        for bit in range(m):
            popcount += ((masks >> bit) & 1).astype(np.int8)

        for subset_size in range(2, m + 1):
            layer = masks[popcount == subset_size]
            for j in range(m):
                with_j = layer[(layer >> j) & 1 == 1]
                previous = with_j ^ (1 << j)
                candidates = dp[previous] + to_city[:, j]
                best_k = np.argmin(candidates, axis=1)
                dp[with_j, j] = candidates[np.arange(len(with_j)), best_k]
                parent[with_j, j] = best_k

        full = size - 1
        j = int(np.argmin(dp[full] + D[1:, 0]))
        mask = full
        reversed_path = []
        while True:
            reversed_path.append(j + 1)
            k = int(parent[mask, j])
            mask ^= 1 << j
            if k < 0:
                break
            j = k

        tour = [0] + reversed_path[::-1]
        return tour, self.calculate_tour_distance(tour)

    # ==================== GREEDY HEURISTICS ====================

    def nearest_neighbor(self, start_city: int = 0) -> Tuple[List[int], float]:
        """
        Greedy nearest neighbor algorithm.

        Time Complexity: O(n²)
        """
        D = self._dist
        unvisited = set(range(self.n_cities))
        tour = [start_city]
        unvisited.remove(start_city)
        current = start_city

        while unvisited:
            nearest = min(unvisited, key=lambda x: D[current][x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest

        return tour, self.calculate_tour_distance(tour)

    def nearest_insertion(self) -> Tuple[List[int], float]:
        """
        Nearest insertion algorithm - builds tour incrementally.

        Time Complexity: O(n²)
        """
        n = self.n_cities
        if n < 3:
            tour = list(range(n))
            return tour, self.calculate_tour_distance(tour)

        D = self._dist

        # Start with two nearest cities
        min_dist = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                if D[i][j] < min_dist:
                    min_dist = D[i][j]
                    start_pair = (i, j)

        tour = list(start_pair)
        unvisited = set(range(n)) - set(tour)
        # Distance from each unvisited city to its closest city in the tour,
        # updated incrementally so each step is O(n) instead of O(n²).
        dist_to_tour = {c: min(D[c][start_pair[0]], D[c][start_pair[1]])
                        for c in unvisited}

        while unvisited:
            # Find nearest unvisited city to tour
            nearest_city = min(unvisited, key=dist_to_tour.__getitem__)

            # Find best insertion position
            best_increase = float('inf')
            best_pos = 0
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                increase = (D[tour[i]][nearest_city] +
                            D[nearest_city][tour[j]] -
                            D[tour[i]][tour[j]])
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i + 1

            tour.insert(best_pos, nearest_city)
            unvisited.remove(nearest_city)
            del dist_to_tour[nearest_city]
            for c in unvisited:
                if D[c][nearest_city] < dist_to_tour[c]:
                    dist_to_tour[c] = D[c][nearest_city]

        return tour, self.calculate_tour_distance(tour)

    # ==================== LOCAL SEARCH ====================

    def _neighbor_lists(self, k: Optional[int] = None):
        """
        For each city, the other cities in order of increasing distance
        (ties by index), cached per k.

        k given: lists of the k nearest cities, found with a partial sort.
        k None: all cities, as a `_SortedNeighbors` whose rows are iterated
        lazily, so scans that stop early never pay for a full n x n sort.
        """
        if k is not None and k < 1:
            raise ValueError(f"neighbors must be a positive integer or None (got {k})")
        cache = self.__dict__.setdefault('_neighbor_cache', {})
        if k is not None:
            k = min(k, self.n_cities - 1)
        if k not in cache:
            if k is None:
                cache[k] = _SortedNeighbors(self, prefix=16)
            else:
                cache[k] = self._nearest(k)
        return cache[k]

    def _nearest(self, k: int) -> List[List[int]]:
        """The k nearest other cities of every city, in the same order as a
        stable argsort of the distance row (self excluded)."""
        D = self.distance_matrix
        n = self.n_cities
        if k <= 0:
            return [[] for _ in range(n)]
        width = min(k + 1, n)                 # + 1 for the city itself
        kth = np.partition(D, width - 1, axis=1)[:, width - 1]
        lists = []
        for i in range(n):
            row = D[i]
            candidates = np.nonzero(row <= kth[i])[0]
            candidates = candidates[np.lexsort((candidates, row[candidates]))]
            lists.append([int(c) for c in candidates if c != i][:k])
        return lists

    def _validate_tour(self, tour: List[int]) -> List[int]:
        """Return `tour` as a new list of distinct city indices, or raise."""
        tour = [int(c) for c in tour]
        if len(set(tour)) != len(tour):
            raise ValueError("tour contains repeated cities")
        if any(c < 0 or c >= self.n_cities for c in tour):
            raise ValueError(f"tour contains a city index outside 0..{self.n_cities - 1}")
        return tour

    @staticmethod
    def _rotate_to(tour: List[int], start: int) -> None:
        """Rotate `tour` in place so that it begins with `start`."""
        idx = tour.index(start)
        if idx:
            tour[:] = tour[idx:] + tour[:idx]

    def _two_opt_dlb(self, tour: List[int], neighbors,
                     queue: Optional[List[int]] = None,
                     max_passes: Optional[int] = None,
                     until_optimal: bool = False) -> int:
        """
        In-place 2-opt with neighbor lists and don't-look bits.

        Only cities in `queue` (all cities if None) are examined at first;
        endpoints of every applied move are re-queued. For each city a and
        each tour direction, candidate partners c are scanned in order of
        distance and the scan stops once D[a][c] >= D[a][next(a)], because
        an improving move must add an edge shorter than one it removes.

        Don't-look bits alone can miss a move that appears at a city whose
        own edges did not change. With `until_optimal`, every city is
        re-examined after the queue empties, until a full sweep finds no
        move; with full neighbor lists the result is then 2-optimal.

        Passes: the initially queued cities form pass 1; a city re-queued
        while processing a pass-p city belongs to pass p + 1. With
        `max_passes`, cities beyond that pass are not examined (0 = no
        change). The tour may be a subset of the cities. It keeps its
        first city.

        Returns the number of improving moves applied.
        """
        n = len(tour)
        if n < 4 or max_passes == 0:
            return 0
        start = tour[0]
        D = self._dist
        size = self.n_cities
        pos = [0] * size
        in_tour = [False] * size
        for idx, city in enumerate(tour):
            pos[city] = idx
            in_tour[city] = True

        def reverse(i, j):
            """Reverse tour positions i..j (cyclic, inclusive), choosing the
            shorter side so each move costs at most n/2 swaps."""
            length = (j - i) % n + 1
            if 2 * length > n:
                i, j = (j + 1) % n, (i - 1) % n
                length = n - length
            for _ in range(length // 2):
                ci, cj = tour[i], tour[j]
                tour[i], tour[j] = cj, ci
                pos[cj], pos[ci] = i, j
                i = (i + 1) % n
                j = (j - 1) % n

        full_sweep = queue is None
        if queue is None:
            queue = list(tour)
        in_queue = [False] * size
        for city in queue:
            in_queue[city] = True
        queue = list(queue)
        pass_of = [1] * size
        moves = 0
        sweep_moves = 0

        while True:
            if not queue:
                if not until_optimal or (full_sweep and sweep_moves == 0):
                    break
                queue = list(tour)
                for city in queue:
                    in_queue[city] = True
                full_sweep, sweep_moves = True, 0
            a = queue.pop()
            in_queue[a] = False
            current_pass = pass_of[a]
            if max_passes is not None and current_pass > max_passes:
                continue
            improved = False
            for direction in (1, -1):
                pa = pos[a]
                b = tour[(pa + direction) % n]
                d_ab = D[a][b]
                for c in neighbors[a]:
                    d_ac = D[a][c]
                    if d_ac >= d_ab:
                        break
                    if not in_tour[c]:
                        continue
                    pc = pos[c]
                    d = tour[(pc + direction) % n]
                    if c == b or d == a:
                        continue
                    delta = d_ac + D[b][d] - d_ab - D[c][d]
                    if delta < -EPSILON:
                        if direction == 1:
                            reverse(pos[b], pc)      # a b ... c d -> a c ... b d
                        else:
                            reverse(pa, pos[d])      # b a ... d c -> b d ... a c
                        moves += 1
                        sweep_moves += 1
                        improved = True
                        for x in (a, b, c, d):
                            if not in_queue[x]:
                                in_queue[x] = True
                                pass_of[x] = current_pass + 1
                                queue.append(x)
                        break
                if improved:
                    break

        self._rotate_to(tour, start)
        return moves

    def two_opt(self, initial_tour: Optional[List[int]] = None,
                max_iterations: Optional[int] = None,
                neighbors: Optional[int] = None) -> Tuple[List[int], float]:
        """
        2-opt local search improvement.

        Uses distance-sorted neighbor lists and don't-look bits, and
        evaluates each move in O(1).

        Args:
            initial_tour: Starting tour (default: Nearest Neighbor)
            max_iterations: Maximum number of improvement passes. Pass 1
                examines every city; each later pass examines the cities
                whose edges changed in the previous pass. None (default)
                runs until no improving move remains; 0 returns the tour
                unchanged.
            neighbors: Only consider the k nearest cities as new edge
                partners. None (default) considers all cities; with
                max_iterations=None the result is then a true 2-opt local
                optimum. k = 8-10 is faster on large instances at a small
                cost in quality.

        Time Complexity: O(n²) per pass in the worst case, typically far
        less thanks to the early stop on sorted neighbor lists. The
        returned tour starts with the same city as the input tour.
        """
        if initial_tour is None:
            tour, _ = self.nearest_neighbor()
        else:
            tour = self._validate_tour(initial_tour)

        self._two_opt_dlb(tour, self._neighbor_lists(neighbors),
                          max_passes=max_iterations,
                          until_optimal=max_iterations is None)
        return tour, self.calculate_tour_distance(tour)

    def _or_opt_pass(self, tour: List[int], neighbors,
                     max_segment: int = 3) -> int:
        """
        In-place Or-opt: move segments of 1..max_segment consecutive cities
        (optionally reversed) next to one of their endpoints' neighbors.
        Repeats full passes until no move improves. Returns moves applied.
        """
        n = len(tour)
        if n == 0:
            return 0
        D = self._dist
        start = tour[0]
        total = 0
        improved = True
        while improved:
            improved = False
            for seg_len in range(1, max_segment + 1):
                if n < seg_len + 3:
                    break
                pos = {city: idx for idx, city in enumerate(tour)}
                for i in range(n):
                    s, e = tour[i], tour[(i + seg_len - 1) % n]
                    p, nx = tour[(i - 1) % n], tour[(i + seg_len) % n]
                    gain_remove = D[p][s] + D[e][nx] - D[p][nx]
                    if gain_remove <= EPSILON:
                        continue

                    best = None
                    for x in ((s,) if s == e else (s, e)):
                        for c in neighbors[x]:
                            pc = pos.get(c)
                            if pc is None or (pc - i) % n < seg_len:
                                continue  # c is not in the tour, or inside the segment
                            for u, v in ((c, tour[(pc + 1) % n]),
                                         (tour[(pc - 1) % n], c)):
                                if (pos[u] - i) % n < seg_len or \
                                        (pos[v] - i) % n < seg_len:
                                    continue
                                base = D[u][v]
                                forward = D[u][s] + D[e][v] - base
                                backward = D[u][e] + D[s][v] - base
                                cost, rev = min((forward, False), (backward, True))
                                gain = gain_remove - cost
                                if gain > EPSILON and (best is None or gain > best[0]):
                                    best = (gain, u, rev)
                    if best is None:
                        continue

                    _, u, rev = best
                    segment = [tour[(i + t) % n] for t in range(seg_len)]
                    if rev:
                        segment.reverse()
                    seg_set = set(segment)
                    rest = [city for city in tour if city not in seg_set]
                    at = rest.index(u) + 1
                    tour[:] = rest[:at] + segment + rest[at:]
                    pos = {city: idx for idx, city in enumerate(tour)}
                    total += 1
                    improved = True
        self._rotate_to(tour, start)
        return total

    def or_opt(self, initial_tour: Optional[List[int]] = None,
               max_segment: int = 3,
               neighbors: Optional[int] = 10) -> Tuple[List[int], float]:
        """
        Or-opt local search: relocate segments of up to `max_segment`
        cities (in either orientation) to a cheaper position.

        Args:
            initial_tour: Starting tour (default: Nearest Neighbor)
            max_segment: Longest segment moved
            neighbors: Insertion points considered per segment endpoint
                (k nearest cities; None = all)
        """
        if initial_tour is None:
            tour, _ = self.nearest_neighbor()
        else:
            tour = self._validate_tour(initial_tour)
        self._or_opt_pass(tour, self._neighbor_lists(neighbors), max_segment)
        return tour, self.calculate_tour_distance(tour)

    def _local_search(self, tour: List[int], neighbors: Optional[int] = None,
                      or_opt_neighbors: Optional[int] = 10) -> None:
        """In-place 2-opt and Or-opt, alternated until neither improves."""
        full = self._neighbor_lists(neighbors)
        near = self._neighbor_lists(or_opt_neighbors)
        self._two_opt_dlb(tour, full, until_optimal=True)
        while self._or_opt_pass(tour, near):
            self._two_opt_dlb(tour, full, until_optimal=True)

    def local_search(self, initial_tour: Optional[List[int]] = None,
                     neighbors: Optional[int] = None) -> Tuple[List[int], float]:
        """
        2-opt followed by Or-opt, repeated until neither improves.

        Args:
            initial_tour: Starting tour (default: Nearest Neighbor)
            neighbors: Neighbor-list size for 2-opt (None = all cities)
        """
        if initial_tour is None:
            tour, _ = self.nearest_neighbor()
        else:
            tour = self._validate_tour(initial_tour)
        self._local_search(tour, neighbors)
        return tour, self.calculate_tour_distance(tour)

    def _double_bridge(self, tour: List[int]) -> Tuple[List[int], List[int]]:
        """Random double-bridge kick (a 4-opt move 2-opt cannot undo).
        Returns the new tour and the cities whose edges changed."""
        n = len(tour)
        a, b, c = sorted(self.rng.sample(range(1, n), 3))
        new = tour[:a] + tour[b:c] + tour[a:b] + tour[c:]
        # 1 <= a < b < c <= n - 1, so the closing edge (tour[-1], tour[0])
        # is never cut.
        touched = [tour[a - 1], tour[a], tour[b - 1], tour[b],
                   tour[c - 1], tour[c]]
        return new, touched

    def iterated_local_search(self, initial_tour: Optional[List[int]] = None,
                              iterations: int = 1000,
                              neighbors: Optional[int] = None) -> Tuple[List[int], float]:
        """
        Iterated local search: repeatedly apply a random double-bridge kick
        to the best tour, re-optimize locally with 2-opt (only around the
        kicked edges), and keep the result if it is shorter. The final best
        tour is polished with Or-opt.

        Args:
            initial_tour: Starting tour (default: Nearest Neighbor)
            iterations: Number of kicks
            neighbors: Neighbor-list size for 2-opt (None = all cities)
        """
        if initial_tour is None:
            best, _ = self.nearest_neighbor()
        else:
            best = self._validate_tour(initial_tour)
        n = len(best)
        self._local_search(best, neighbors)
        if n < 8:
            return best, self.calculate_tour_distance(best)

        full = self._neighbor_lists(neighbors)
        best_distance = self.calculate_tour_distance(best)
        for _ in range(iterations):
            candidate, touched = self._double_bridge(best)
            self._two_opt_dlb(candidate, full, queue=touched)
            distance = self.calculate_tour_distance(candidate)
            if distance < best_distance - EPSILON:
                best, best_distance = candidate, distance

        self._local_search(best, neighbors)
        return best, self.calculate_tour_distance(best)

    def three_opt(self, initial_tour: Optional[List[int]] = None,
                  max_iterations: int = 100) -> Tuple[List[int], float]:
        """
        3-opt local search - more powerful but slower than 2-opt.

        For every triple of edges, tries the segment reversals and the
        segment exchange that reconnect the tour, each evaluated in O(1).

        Time Complexity: O(n³ * iterations)
        """
        if initial_tour is None:
            tour, _ = self.nearest_neighbor()
        else:
            tour = self._validate_tour(initial_tour)

        n = len(tour)
        D = self._dist

        def apply_best_move(i, j, k):
            """Apply the first improving reconnection of the three edges
            (tour[i-1], tour[i]), (tour[j-1], tour[j]), (tour[k-1], tour[k])."""
            A, B = tour[i - 1], tour[i]
            C, Dn = tour[j - 1], tour[j]
            E, F = tour[k - 1], tour[k % n]
            d0 = D[A][B] + D[C][Dn] + D[E][F]
            d1 = D[A][C] + D[B][Dn] + D[E][F]
            d2 = D[A][B] + D[C][E] + D[Dn][F]
            d3 = D[A][Dn] + D[E][B] + D[C][F]
            d4 = D[F][B] + D[C][Dn] + D[E][A]

            if d1 < d0 - EPSILON:
                tour[i:j] = reversed(tour[i:j])
            elif d2 < d0 - EPSILON:
                tour[j:k] = reversed(tour[j:k])
            elif d4 < d0 - EPSILON:
                tour[i:k] = reversed(tour[i:k])
            elif d3 < d0 - EPSILON:
                tour[i:k] = tour[j:k] + tour[i:j]
            else:
                return False
            return True

        improved = n >= 6
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            for i in range(n):
                for j in range(i + 2, n):
                    for k in range(j + 2, n + (i > 0)):
                        if apply_best_move(i, j, k):
                            improved = True

            iterations += 1

        return tour, self.calculate_tour_distance(tour)

    # ==================== METAHEURISTICS ====================

    def _random_two_opt_move(self, tour: List[int]):
        """Pick a random 2-opt move reversing tour[i:j]; return (i, j, delta)
        or None for a degenerate pick."""
        n = len(tour)
        D = self._dist
        i, j = sorted(self.rng.sample(range(n + 1), 2))
        if j - i < 2 or (i == 0 and j == n):
            return None
        a, b = tour[i - 1], tour[i]
        c, d = tour[j - 1], tour[j % n]
        return i, j, D[a][c] + D[b][d] - D[a][b] - D[c][d]

    @staticmethod
    def _annealing_temperatures(propose, samples: int = 500) -> Tuple[float, float]:
        """
        Choose start/end temperatures from a sample of the moves the search
        will actually propose (`propose()` returns (i, j, delta) or None):

        - start: the median worsening move is accepted with probability 0.3
        - end: a 5th-percentile worsening move is accepted with
          probability 0.01

        Calibrating on the proposal distribution keeps the schedule matched
        to the move type: nearest-neighbor moves are far smaller than moves
        between random cities. Temperatures scale with the coordinates.
        """
        worsening = []
        for _ in range(samples):
            move = propose()
            if move is not None and move[2] > EPSILON:
                worsening.append(move[2])
        if not worsening:
            return 1.0, 1e-3
        worsening.sort()

        def quantile(q):
            return worsening[min(len(worsening) - 1, int(q * len(worsening)))]

        return (-quantile(0.50) / math.log(0.3),
                -quantile(0.05) / math.log(0.01))

    def simulated_annealing(self, initial_temp: Optional[float] = None,
                            cooling_rate: Optional[float] = None,
                            min_temp: Optional[float] = None,
                            max_iterations: int = 100000,
                            polish: bool = True,
                            neighbors: Optional[int] = 5) -> Tuple[List[int], float]:
        """
        Simulated annealing metaheuristic.

        Uses random 2-opt moves evaluated in O(1). By default a move joins
        a random city to one of its `neighbors` nearest cities, which wastes
        far fewer iterations than moves between random cities.

        Args:
            initial_temp: Starting temperature. If None, derived from a
                sample of the proposed moves (see _annealing_temperatures).
            cooling_rate: Geometric cooling factor per iteration. If None, it
                is chosen so the temperature reaches min_temp exactly at
                max_iterations, so the full iteration budget is used.
            min_temp: Temperature at which the search stops. If None,
                derived from the instance.
            max_iterations: Maximum number of moves attempted
            polish: Finish with 2-opt + Or-opt on the best tour found
            neighbors: Draw moves from each city's k nearest cities (None =
                reverse a uniformly random segment)
        """
        n = self.n_cities
        # Start with nearest neighbor solution
        current_tour, current_distance = self.nearest_neighbor()
        if n < 4:
            return current_tour, current_distance

        rng = self.rng
        if neighbors is None:
            def propose():
                return self._random_two_opt_move(current_tour)
        else:
            near = self._neighbor_lists(neighbors)
            pos = [0] * n
            for idx, city in enumerate(current_tour):
                pos[city] = idx
            D = self._dist

            def propose():
                """2-opt move adding edge (a, c), c one of a's nearest cities:
                reverse current_tour[i:j] with i = pos[a] + 1, j = pos[c] + 1
                (positions ordered)."""
                a = rng.randrange(n)
                c = near[a][rng.randrange(len(near[a]))]
                i, j = sorted((pos[a], pos[c]))
                if j - i < 2 or (i == 0 and j == n - 1):
                    return None
                A, B = current_tour[i], current_tour[i + 1]
                C, Dn = current_tour[j], current_tour[(j + 1) % n]
                return i + 1, j + 1, D[A][C] + D[B][Dn] - D[A][B] - D[C][Dn]

        if initial_temp is None or min_temp is None:
            auto_start, auto_end = self._annealing_temperatures(propose)
            initial_temp = auto_start if initial_temp is None else initial_temp
            min_temp = auto_end if min_temp is None else min_temp
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be at least 1 (got {max_iterations})")
        if not 0 < min_temp < initial_temp:
            raise ValueError(f"need 0 < min_temp < initial_temp, got min_temp={min_temp:g}, "
                             f"initial_temp={initial_temp:g} (a value left as None is "
                             f"derived from the instance's move sizes)")
        if cooling_rate is None:
            cooling_rate = (min_temp / initial_temp) ** (1 / max_iterations)
        elif not 0 < cooling_rate < 1:
            raise ValueError(f"cooling_rate must be between 0 and 1 (got {cooling_rate:g})")

        best_tour = current_tour.copy()
        best_distance = current_distance
        temp = initial_temp
        iteration = 0

        while temp > min_temp and iteration < max_iterations:
            temp *= cooling_rate
            iteration += 1

            move = propose()
            if move is None:
                continue
            i, j, delta = move

            # Accept or reject move
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current_tour[i:j] = reversed(current_tour[i:j])
                if neighbors is not None:
                    for idx in range(i, j):
                        pos[current_tour[idx]] = idx
                current_distance += delta

                if current_distance < best_distance - EPSILON:
                    best_tour = current_tour.copy()
                    best_distance = current_distance

        if polish:
            self._local_search(best_tour)
        return best_tour, self.calculate_tour_distance(best_tour)

    def genetic_algorithm(self, population_size: int = 30,
                          generations: int = 100,
                          mutation_rate: float = 0.2,
                          elite_size: int = 4,
                          neighbors: Optional[int] = 10) -> Tuple[List[int], float]:
        """
        Memetic genetic algorithm for TSP.

        - Initial population: Nearest Neighbor tours from different start
          cities (plus random tours if the population is larger than n),
          each improved with 2-opt.
        - Tournament selection, order crossover (OX), inversion mutation
          (reverse a random segment), then 2-opt on every child.
        - The `elite_size` best tours survive unchanged.

        Args:
            population_size: Tours per generation
            generations: Number of generations
            mutation_rate: Probability that a child is mutated
            elite_size: Best tours copied to the next generation
            neighbors: Neighbor-list size for the 2-opt step (None = all)
        """
        n = self.n_cities
        if n < 4:
            tour = list(range(n))
            return tour, self.calculate_tour_distance(tour)

        rng = self.rng
        neighbor_lists = self._neighbor_lists(neighbors)
        elite_size = min(elite_size, population_size)

        def improve(tour):
            self._two_opt_dlb(tour, neighbor_lists)
            return tour

        def selection(population, distances):
            """Tournament selection (shorter tour wins)."""
            tournament_size = min(3, len(population))
            selected = []
            for _ in range(len(population)):
                contenders = rng.sample(range(len(population)), tournament_size)
                selected.append(population[min(contenders, key=distances.__getitem__)])
            return selected

        def crossover(parent1, parent2):
            """Order crossover (OX)."""
            size = len(parent1)
            start, end = sorted(rng.sample(range(size), 2))

            child = [-1] * size
            child[start:end] = parent1[start:end]
            in_child = set(parent1[start:end])

            pointer = end
            for city in parent2[end:] + parent2[:end]:
                if city not in in_child:
                    child[pointer % size] = city
                    pointer += 1

            return child

        def mutate(individual):
            """Inversion mutation: reverse a random segment."""
            if rng.random() < mutation_rate:
                i, j = sorted(rng.sample(range(len(individual)), 2))
                individual[i:j + 1] = reversed(individual[i:j + 1])
            return individual

        # Initialize population
        starts = rng.sample(range(n), min(n, population_size))
        population = [improve(self.nearest_neighbor(c)[0]) for c in starts]
        while len(population) < population_size:
            population.append(improve(rng.sample(range(n), n)))
        distances = [self.calculate_tour_distance(t) for t in population]

        for generation in range(generations):
            # Elite preservation
            order = sorted(range(len(population)), key=distances.__getitem__)
            elite = [population[i] for i in order[:elite_size]]
            elite_distances = [distances[i] for i in order[:elite_size]]

            # Selection, crossover, mutation, local search
            selected = selection(population, distances)
            children = []
            for i in range(population_size - elite_size):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % len(selected)]
                children.append(improve(mutate(crossover(parent1, parent2))))

            population = elite + children
            distances = elite_distances + [self.calculate_tour_distance(t)
                                           for t in children]

        best_idx = min(range(len(population)), key=distances.__getitem__)
        best_tour = population[best_idx]
        return best_tour, self.calculate_tour_distance(best_tour)

    # ==================== VISUALIZATION ====================

    def visualize_tour(self, tour: List[int], title: str = "TSP Tour",
                       save_path: Optional[str] = None, show: bool = True):
        """
        Visualize a TSP tour.

        Args:
            tour: Tour to draw
            title: Plot title
            save_path: If given, save the figure to this path
            show: Whether to display the figure (set False in scripts/tests)

        Returns:
            The matplotlib Figure
        """
        import matplotlib.pyplot as plt  # optional dependency

        fig, ax = plt.subplots(figsize=(10, 8))

        x = self.cities[:, 0]
        y = self.cities[:, 1]

        # Plot tour (closed)
        closed = list(tour) + [tour[0]]
        ax.plot(x[closed], y[closed], color='#2a78d6', linewidth=2,
                solid_joinstyle='round', zorder=2)

        # Plot cities
        ax.scatter(x, y, s=60, color='#52514e', edgecolor='white',
                   linewidth=1.5, zorder=3)

        # Add city labels
        for i, name in enumerate(self.city_names):
            ax.annotate(name, (x[i], y[i]), xytext=(6, 6),
                        textcoords='offset points', fontsize=9, color='#52514e')

        # Add distance to title
        distance = self.calculate_tour_distance(tour)
        ax.set_title(f"{title}\nTotal Distance: {distance:.2f}",
                     fontsize=14, fontweight='bold')
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(True, color='#e1e0d9', linewidth=0.8)
        ax.set_axisbelow(True)

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    ALGORITHMS = {
        'brute_force': 'brute_force',
        'held_karp': 'held_karp',
        'nearest_neighbor': 'nearest_neighbor',
        'nearest_insertion': 'nearest_insertion',
        '2-opt': 'two_opt',
        '3-opt': 'three_opt',
        '2-opt+or-opt': 'local_search',
        'iterated_local_search': 'iterated_local_search',
        'simulated_annealing': 'simulated_annealing',
        'genetic_algorithm': 'genetic_algorithm',
    }

    # Exact algorithms are skipped by compare_algorithms above these sizes.
    EXACT_LIMITS = {'brute_force': 10, 'held_karp': 20}

    def exceeds_size_limit(self, algorithm: str) -> bool:
        """True if `algorithm` is an exact method and this instance has more
        cities than it supports (EXACT_LIMITS)."""
        return self.n_cities > self.EXACT_LIMITS.get(algorithm, self.n_cities)

    def compare_algorithms(self, algorithms: Optional[List[str]] = None) -> Dict:
        """
        Compare performance of different algorithms.

        Args:
            algorithms: List of algorithm names to compare (keys of
                TSPSolver.ALGORITHMS). Exact algorithms are skipped above
                their size limit (EXACT_LIMITS: brute force 10, Held-Karp 20).

        Returns:
            Dictionary with results for each algorithm
        """
        if algorithms is None:
            algorithms = ['nearest_neighbor', '2-opt', 'simulated_annealing']

        unknown = [a for a in algorithms if a not in self.ALGORITHMS]
        if unknown:
            raise ValueError(f"Unknown algorithm(s): {unknown}. "
                             f"Choose from {list(self.ALGORITHMS)}")

        results = {}

        for algo in algorithms:
            if self.exceeds_size_limit(algo):
                continue

            if self.seed is not None:
                # Each algorithm sees the same random stream, so its result
                # does not depend on which algorithms ran before it.
                self.rng.seed(self.seed)
            start_time = time.perf_counter()
            tour, distance = getattr(self, self.ALGORITHMS[algo])()
            execution_time = time.perf_counter() - start_time

            results[algo] = {
                'tour': tour,
                'distance': distance,
                'time': execution_time
            }

        return results


def generate_random_cities(n: int, seed: int = 42) -> np.ndarray:
    """Generate random city coordinates."""
    return np.random.RandomState(seed).rand(n, 2) * 100


class _SortedNeighbors:
    """
    Every other city ordered by distance from each city (ties by index),
    for the early-stopping neighbor scans.

    The `prefix` nearest cities are precomputed for all rows; the rest of a
    row is sorted only when a scan gets that far, which is rare because
    scans stop at the first partner farther than the current tour edge.
    """

    def __init__(self, solver: 'TSPSolver', prefix: int = 16):
        n = solver.n_cities
        self._prefix = solver._nearest(min(prefix, n - 1))
        if prefix >= n - 1:
            self._tails = None                    # rows are already complete
        else:
            self._tails = [_LazyTail(solver.distance_matrix, city, self._prefix[city])
                           for city in range(n)]

    def __len__(self) -> int:
        return len(self._prefix)

    def __iter__(self):
        return (self[city] for city in range(len(self)))

    def __getitem__(self, city: int):
        if not 0 <= city < len(self._prefix):
            raise IndexError(city)
        if self._tails is None:
            return self._prefix[city]
        # chain() only asks the tail for its items once the prefix is used up.
        return chain(self._prefix[city], self._tails[city])


class _LazyTail:
    """The part of one city's sorted neighbor row after its prefix,
    sorted on first iteration."""

    __slots__ = ('_D', '_city', '_prefix', '_items')

    def __init__(self, D: np.ndarray, city: int, prefix: List[int]):
        self._D, self._city, self._prefix, self._items = D, city, prefix, None

    def __iter__(self):
        if self._items is None:
            skip = set(self._prefix)
            skip.add(self._city)
            order = np.argsort(self._D[self._city], kind='stable')
            self._items = [int(c) for c in order if c not in skip]
        return iter(self._items)
