"""
Traveling Salesman Problem (TSP) Solver
========================================
A comprehensive implementation of multiple TSP algorithms with visualization support.

Author: Esad Kopru
License: MIT
"""

import math
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import permutations
from typing import List, Tuple, Dict, Optional

# Moves must improve the tour by more than this to count, which stops
# local search from cycling on floating-point noise.
EPSILON = 1e-10


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
                 seed: Optional[int] = None):
        """
        Initialize TSP solver with city coordinates.

        Args:
            cities: Array of shape (n, 2) with city coordinates
            city_names: Optional list of city names
            seed: Optional seed for the randomized algorithms (simulated
                annealing, genetic algorithm), for reproducible results
        """
        self.cities = np.asarray(cities, dtype=float)
        self.n_cities = len(self.cities)
        self.city_names = city_names or [f"City_{i}" for i in range(self.n_cities)]
        self.rng = random.Random(seed)
        self.distance_matrix = self._calculate_distance_matrix()
        # Plain nested lists are much faster than numpy for scalar lookups
        # inside the Python loops below.
        self._dist = self.distance_matrix.tolist()

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate Euclidean distance matrix between all cities."""
        diff = self.cities[:, np.newaxis, :] - self.cities[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=-1))

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

    def _neighbor_lists(self, k: Optional[int] = None) -> List[List[int]]:
        """For each city, the other cities sorted by distance (k nearest, or
        all when k is None). Cached per k."""
        cache = self.__dict__.setdefault('_neighbor_cache', {})
        k = None if k is None or k >= self.n_cities - 1 else k
        if k not in cache:
            order = np.argsort(self.distance_matrix, axis=1, kind='stable')
            # Column 0 is the city itself (distance 0) unless there are ties
            # at zero distance, so drop the city explicitly.
            lists = [[int(c) for c in row if c != i]
                     for i, row in enumerate(order)]
            cache[k] = lists if k is None else [row[:k] for row in lists]
        return cache[k]

    def _two_opt_dlb(self, tour: List[int], neighbors: List[List[int]],
                     queue: Optional[List[int]] = None,
                     max_moves: Optional[int] = None) -> int:
        """
        In-place 2-opt with neighbor lists and don't-look bits.

        Only cities in `queue` (all cities if None) are examined at first;
        endpoints of every applied move are re-queued. For each city a and
        each tour direction, candidate partners c are scanned in order of
        distance and the scan stops once D[a][c] >= D[a][next(a)], because
        an improving move must add an edge shorter than one it removes.
        With full neighbor lists the result is therefore 2-optimal.

        Returns the number of improving moves applied.
        """
        n = len(tour)
        if n < 4:
            return 0
        D = self._dist
        pos = [0] * n
        for idx, city in enumerate(tour):
            pos[city] = idx

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

        if queue is None:
            queue = list(tour)
        in_queue = [False] * n
        for city in queue:
            in_queue[city] = True
        queue = list(queue)
        moves = 0

        while queue:
            a = queue.pop()
            in_queue[a] = False
            improved = False
            for direction in (1, -1):
                pa = pos[a]
                b = tour[(pa + direction) % n]
                d_ab = D[a][b]
                for c in neighbors[a]:
                    d_ac = D[a][c]
                    if d_ac >= d_ab:
                        break
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
                        improved = True
                        for x in (a, b, c, d):
                            if not in_queue[x]:
                                in_queue[x] = True
                                queue.append(x)
                        break
                if improved:
                    break
            if max_moves is not None and moves >= max_moves:
                break
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
            max_iterations: Maximum number of improving moves (None = until
                no improving move remains)
            neighbors: Only consider the k nearest cities as new edge
                partners. None (default) considers all cities, which
                guarantees a true 2-opt local optimum; k = 8-10 is much
                faster on large instances at a small cost in quality.

        Time Complexity: O(n²) per pass in the worst case, typically far
        less thanks to the early stop on sorted neighbor lists.
        """
        if initial_tour is None:
            tour, _ = self.nearest_neighbor()
        else:
            tour = list(initial_tour)

        self._two_opt_dlb(tour, self._neighbor_lists(neighbors),
                          max_moves=max_iterations)
        return tour, self.calculate_tour_distance(tour)

    def _or_opt_pass(self, tour: List[int], neighbors: List[List[int]],
                     max_segment: int = 3) -> int:
        """
        In-place Or-opt: move segments of 1..max_segment consecutive cities
        (optionally reversed) next to one of their endpoints' neighbors.
        Repeats full passes until no move improves. Returns moves applied.
        """
        n = len(tour)
        D = self._dist
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
                    for x in (s, e):
                        for c in neighbors[x]:
                            if (pos[c] - i) % n < seg_len:
                                continue  # c is inside the segment
                            pc = pos[c]
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
            tour = list(initial_tour)
        self._or_opt_pass(tour, self._neighbor_lists(neighbors), max_segment)
        return tour, self.calculate_tour_distance(tour)

    def _local_search(self, tour: List[int], neighbors: Optional[int] = None,
                      or_opt_neighbors: Optional[int] = 10) -> None:
        """In-place 2-opt and Or-opt, alternated until neither improves."""
        full = self._neighbor_lists(neighbors)
        near = self._neighbor_lists(or_opt_neighbors)
        self._two_opt_dlb(tour, full)
        while self._or_opt_pass(tour, near):
            self._two_opt_dlb(tour, full)

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
            tour = list(initial_tour)
        self._local_search(tour, neighbors)
        return tour, self.calculate_tour_distance(tour)

    def _double_bridge(self, tour: List[int]) -> Tuple[List[int], List[int]]:
        """Random double-bridge kick (a 4-opt move 2-opt cannot undo).
        Returns the new tour and the cities whose edges changed."""
        n = len(tour)
        a, b, c = sorted(self.rng.sample(range(1, n), 3))
        new = tour[:a] + tour[b:c] + tour[a:b] + tour[c:]
        touched = [tour[a - 1], tour[a], tour[b - 1], tour[b],
                   tour[c - 1], tour[c % n], tour[0], tour[-1]]
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
            best = list(initial_tour)
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
            tour = list(initial_tour)

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

    def _annealing_temperatures(self, tour: List[int],
                                samples: int = 500) -> Tuple[float, float]:
        """
        Choose start/end temperatures for this instance from a sample of
        random 2-opt moves on `tour`:

        - start: a 10th-percentile worsening move is accepted with
          probability 0.2
        - end: a 1st-percentile worsening move is accepted with
          probability 0.01

        Percentiles rather than the mean are used because random moves on
        a decent tour are mostly very bad; the moves that matter late in
        the search are the smallest ones.
        """
        worsening = []
        for _ in range(samples):
            move = self._random_two_opt_move(tour)
            if move is not None and move[2] > EPSILON:
                worsening.append(move[2])
        if not worsening:
            return 1.0, 1e-3
        worsening.sort()

        def quantile(q):
            return worsening[min(len(worsening) - 1, int(q * len(worsening)))]

        return (-quantile(0.10) / math.log(0.2),
                -quantile(0.01) / math.log(0.01))

    def simulated_annealing(self, initial_temp: Optional[float] = None,
                            cooling_rate: Optional[float] = None,
                            min_temp: Optional[float] = None,
                            max_iterations: int = 100000,
                            polish: bool = True) -> Tuple[List[int], float]:
        """
        Simulated annealing metaheuristic.

        Uses random 2-opt moves evaluated in O(1).

        Args:
            initial_temp: Starting temperature. If None, derived from the
                instance (see _annealing_temperatures).
            cooling_rate: Geometric cooling factor per iteration. If None, it
                is chosen so the temperature reaches min_temp exactly at
                max_iterations, so the full iteration budget is used.
            min_temp: Temperature at which the search stops. If None,
                derived from the instance.
            max_iterations: Maximum number of moves attempted
            polish: Finish with 2-opt + Or-opt on the best tour found
        """
        n = self.n_cities
        # Start with nearest neighbor solution
        current_tour, current_distance = self.nearest_neighbor()
        if n < 4:
            return current_tour, current_distance

        if initial_temp is None or min_temp is None:
            auto_start, auto_end = self._annealing_temperatures(current_tour)
            initial_temp = auto_start if initial_temp is None else initial_temp
            min_temp = auto_end if min_temp is None else min_temp
        if cooling_rate is None:
            cooling_rate = (min_temp / initial_temp) ** (1 / max_iterations)

        rng = self.rng
        best_tour = current_tour.copy()
        best_distance = current_distance

        temp = initial_temp
        iteration = 0

        while temp > min_temp and iteration < max_iterations:
            temp *= cooling_rate
            iteration += 1

            move = self._random_two_opt_move(current_tour)
            if move is None:
                continue
            i, j, delta = move

            # Accept or reject move
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current_tour[i:j] = reversed(current_tour[i:j])
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
            if self.n_cities > self.EXACT_LIMITS.get(algo, self.n_cities):
                continue

            start_time = time.perf_counter()
            tour, distance = getattr(self, self.ALGORITHMS[algo])()
            execution_time = time.perf_counter() - start_time

            results[algo] = {
                'tour': tour,
                'distance': distance,
                'time': execution_time
            }

        return results


# ==================== EXAMPLE USAGE ====================

def generate_random_cities(n: int, seed: int = 42) -> np.ndarray:
    """Generate random city coordinates."""
    return np.random.RandomState(seed).rand(n, 2) * 100


def run_example(show: bool = True):
    """Run example demonstrating TSP solver capabilities."""

    # Generate problem instance
    n_cities = 20
    cities = generate_random_cities(n_cities)
    city_names = [f"C{i}" for i in range(n_cities)]

    # Create solver
    solver = TSPSolver(cities, city_names, seed=42)

    print("=" * 60)
    print(f"TRAVELING SALESMAN PROBLEM - {n_cities} Cities")
    print("=" * 60)

    # Compare algorithms
    algorithms = ['nearest_neighbor', 'nearest_insertion', '2-opt', '3-opt',
                  'simulated_annealing', 'genetic_algorithm']

    results = solver.compare_algorithms(algorithms)

    # Print results
    print("\nAlgorithm Comparison:")
    print("-" * 60)
    print(f"{'Algorithm':<20} {'Distance':<15} {'Time (s)':<15}")
    print("-" * 60)

    for algo, result in results.items():
        print(f"{algo:<20} {result['distance']:<15.2f} {result['time']:<15.4f}")

    # Find best solution
    best_algo = min(results.keys(), key=lambda x: results[x]['distance'])
    best_tour = results[best_algo]['tour']
    best_distance = results[best_algo]['distance']

    print("-" * 60)
    print(f"\nBest Solution: {best_algo}")
    print(f"Tour: {' -> '.join([city_names[i] for i in best_tour[:5]])} -> ...")
    print(f"Total Distance: {best_distance:.2f}")

    # Visualize best tour
    solver.visualize_tour(best_tour, f"Best Tour ({best_algo})", show=show)

    return solver, results


# ==================== ADVANCED FEATURES ====================

class TSPBenchmark:
    """Benchmark suite for TSP algorithms."""

    @staticmethod
    def generate_benchmark_instances():
        """Generate standard benchmark instances."""
        instances = {
            'random_10': generate_random_cities(10),
            'random_20': generate_random_cities(20),
            'random_50': generate_random_cities(50),
            'grid_16': np.array([(i, j) for i in range(4) for j in range(4)]),
            'circle_20': np.array([(10 * np.cos(2 * np.pi * i / 20),
                                    10 * np.sin(2 * np.pi * i / 20))
                                   for i in range(20)])
        }
        return instances

    @staticmethod
    def run_benchmark(instances: Dict[str, np.ndarray],
                      algorithms: List[str],
                      seed: Optional[int] = None) -> pd.DataFrame:
        """Run benchmark on multiple instances and algorithms."""
        results = []

        for instance_name, cities in instances.items():
            solver = TSPSolver(cities, seed=seed)
            algo_results = solver.compare_algorithms(algorithms)

            for algo, result in algo_results.items():
                results.append({
                    'Instance': instance_name,
                    'Cities': len(cities),
                    'Algorithm': algo,
                    'Distance': result['distance'],
                    'Time': result['time']
                })

        return pd.DataFrame(results)


if __name__ == "__main__":
    # Run example
    solver, results = run_example()

    # Additional analysis
    print("\n" + "=" * 60)
    print("ADDITIONAL ANALYSIS")
    print("=" * 60)

    # Test on smaller instance for exact solution
    small_cities = generate_random_cities(8)
    small_solver = TSPSolver(small_cities, seed=42)

    print("\nSmall Instance (8 cities) - Exact vs Heuristic:")
    exact_tour, exact_dist = small_solver.brute_force()
    heuristic_tour, heuristic_dist = small_solver.simulated_annealing()

    print(f"Exact Solution: {exact_dist:.2f}")
    print(f"Heuristic Solution: {heuristic_dist:.2f}")
    print(f"Gap: {(heuristic_dist - exact_dist) / exact_dist * 100:.2f}%")
