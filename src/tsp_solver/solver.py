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
from typing import Any, List, Tuple, Dict, Optional, Union

from .tsplib import load_tsplib

# Smallest improvement that counts as one, for distances of the order of
# 100. The algorithms use TSPSolver.tolerance, which scales with the
# instance; this constant remains for code that imports it.
EPSILON = 1e-10

# Longest piece (in cities) that a double-bridge kick moves; see
# _TourSearch.double_bridge and TSPSolver.iterated_local_search.
_KICK_PIECE = 50

# From about this many cities on, scanning a distance row with NumPy beats
# a Python loop over it (used by the constructive heuristics).
_NUMPY_ROW_THRESHOLD = 150


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
    - Iterated Local Search (double-bridge kicks repaired by 2-opt and Or-opt)
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
        if not np.isfinite(self.cities).all():
            bad = np.nonzero(~np.isfinite(self.cities).all(axis=1))[0]
            raise ValueError(f"cities must have finite coordinates; "
                             f"city {int(bad[0])} is {self.cities[bad[0]].tolist()}")
        self.n_cities = len(self.cities)
        if city_names is None:
            self.city_names = [f"City_{i}" for i in range(self.n_cities)]
        else:
            self.city_names = [str(name) for name in city_names]
            if len(self.city_names) != self.n_cities:
                raise ValueError(f"{len(self.city_names)} city names for "
                                 f"{self.n_cities} cities")
        self.seed = seed
        self.rng = random.Random(seed)
        self.distance_matrix = self._calculate_distance_matrix()
        if not np.isfinite(self.distance_matrix).all():
            raise ValueError("coordinates are too large: a distance between "
                             "two cities overflows")
        # Plain nested lists are much faster than numpy for scalar lookups
        # inside the Python loops below.
        self._dist = self.distance_matrix.tolist()
        # Smallest improvement that counts as one, relative to the largest
        # distance. A move's gain is a sum of a few distances, so its
        # rounding error grows with them: a fixed threshold lets local
        # search trade equal-length tours back and forth forever when the
        # coordinates are large (grid points in the millions), and ignores
        # every real improvement when they are tiny.
        self.tolerance = 1e-12 * float(self.distance_matrix.max())
        self._neighbor_cache: Dict[Optional[int], Any] = {}

    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate the distance matrix between all cities."""
        # One coordinate at a time: two n x n arrays of scratch space
        # instead of the five that broadcasting both at once needs.
        x, y = self.cities[:, 0], self.cities[:, 1]
        with np.errstate(over='ignore'):    # overflow is reported by __init__
            diff = x[:, np.newaxis] - x[np.newaxis, :]
            squared = diff * diff
            np.subtract(y[:, np.newaxis], y[np.newaxis, :], out=diff)
            np.multiply(diff, diff, out=diff)
            squared += diff
            return DISTANCE_FUNCTIONS[self.distance](squared)

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
        """
        Calculate total distance of a tour (returning to its first city).

        The edges are added exactly and rounded once (math.fsum), so the
        result does not depend on where the tour starts, on its direction
        or on the Python version (sum() only compensates for rounding from
        3.12 on). The genetic algorithm needs that: it ranks tours by this
        number, and copies of one tour must not be ordered by rounding noise.
        """
        D = self._dist
        tour = list(tour)
        return math.fsum(D[a][b] for a, b in zip(tour, tour[1:] + tour[:1]))

    def _check_city(self, city: int, name: str = 'start_city') -> int:
        """Return `city` as an int index, or raise if it is not a city."""
        if isinstance(city, bool) or not isinstance(city, (int, np.integer)) \
                or not 0 <= city < self.n_cities:
            raise ValueError(f"{name} must be a city index in "
                             f"0..{self.n_cities - 1} (got {city!r})")
        return int(city)

    # ==================== EXACT ALGORITHM ====================

    def brute_force(self, start_city: int = 0) -> Tuple[List[int], float]:
        """
        Find optimal solution using brute force (only for small instances).

        Time Complexity: O(n!)
        Use only for n <= 10 cities
        """
        if self.n_cities > 10:
            raise ValueError("Brute force only feasible for <= 10 cities")
        start_city = self._check_city(start_city)

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
        start_city = self._check_city(start_city)
        n = self.n_cities
        tour = [start_city]
        current = start_city
        if n < _NUMPY_ROW_THRESHOLD:
            D = self._dist
            unvisited = set(range(n))
            unvisited.remove(start_city)
            while unvisited:
                row = D[current]
                current = min(unvisited, key=row.__getitem__)
                tour.append(current)
                unvisited.remove(current)
        else:
            # Same choices (ties go to the lower index), one vectorized
            # row scan per step.
            D = self.distance_matrix
            visited = np.zeros(n)               # +inf once a city is used
            visited[start_city] = np.inf
            for _ in range(n - 1):
                current = int(np.argmin(D[current] + visited))
                visited[current] = np.inf
                tour.append(current)

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

        # Start with the two nearest cities (the first such pair i < j)
        off_diagonal = self.distance_matrix.copy()
        np.fill_diagonal(off_diagonal, np.inf)
        start_pair = divmod(int(np.argmin(off_diagonal)), n)

        if n >= _NUMPY_ROW_THRESHOLD:
            # The loop below, one vectorized scan per step (same choices:
            # ties go to the lower index or the earlier tour position).
            D = self.distance_matrix
            order = np.empty(n, dtype=np.intp)      # the tour so far
            order[:2] = start_pair
            size = 2
            in_tour = np.zeros(n)                   # +inf once inserted
            in_tour[list(start_pair)] = np.inf
            dist_to_tour = np.minimum(D[start_pair[0]], D[start_pair[1]])
            while size < n:
                city = int(np.argmin(dist_to_tour + in_tour))
                row = D[city]
                before = order[:size]
                after = np.roll(before, -1)
                increase = row[before] + row[after] - D[before, after]
                at = int(np.argmin(increase)) + 1
                order[at + 1:size + 1] = order[at:size]
                order[at] = city
                size += 1
                in_tour[city] = np.inf
                np.minimum(dist_to_tour, row, out=dist_to_tour)
            tour = order.tolist()
            return tour, self.calculate_tour_distance(tour)

        D = self._dist
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
        cache = self._neighbor_cache
        if k is not None:
            k = min(k, self.n_cities - 1)
        if k not in cache:
            if k is None:
                cache[k] = _SortedNeighbors(self, prefix=16)
            else:
                # A shorter list is the start of any longer one already made.
                longer = [rows for size, rows in cache.items()
                          if size is not None and size > k]
                if None in cache and len(cache[None].prefix[0]) >= k:
                    longer.append(cache[None].prefix)
                if longer:
                    cache[k] = [row[:k] for row in longer[0]]
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

    def _start_tour(self, initial_tour: Optional[List[int]]) -> List[int]:
        """A validated copy of `initial_tour`, or the Nearest Neighbor tour."""
        if initial_tour is None:
            return self.nearest_neighbor()[0]
        return self._validate_tour(initial_tour)

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
        tour = self._start_tour(initial_tour)
        if tour:
            start = tour[0]
            _TourSearch(self, tour).optimize(
                self._neighbor_lists(neighbors), max_passes=max_iterations,
                until_optimal=max_iterations is None)
            self._rotate_to(tour, start)
        return tour, self.calculate_tour_distance(tour)

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
        tour = self._start_tour(initial_tour)
        if tour:
            start = tour[0]
            _TourSearch(self, tour).optimize(
                None, self._neighbor_lists(neighbors), until_optimal=True,
                max_segment=max_segment)
            self._rotate_to(tour, start)
        return tour, self.calculate_tour_distance(tour)

    def _local_search(self, tour: List[int], neighbors: Optional[int] = None,
                      or_opt_neighbors: Optional[int] = 10) -> None:
        """In-place 2-opt and Or-opt until neither finds an improving move.
        The tour keeps its first city."""
        if tour:
            start = tour[0]
            search = _TourSearch(self, tour)
            two_opt_lists = self._neighbor_lists(neighbors)
            # 2-opt alone first, exactly as two_opt() does it, so that the
            # result is never longer than plain 2-opt from the same start.
            search.optimize(two_opt_lists, until_optimal=True)
            search.optimize(two_opt_lists, self._neighbor_lists(or_opt_neighbors),
                            until_optimal=True)
            self._rotate_to(tour, start)

    def local_search(self, initial_tour: Optional[List[int]] = None,
                     neighbors: Optional[int] = None) -> Tuple[List[int], float]:
        """
        2-opt, then 2-opt and Or-opt together until neither finds an
        improving move. The result is never longer than two_opt() from the
        same start.

        In the combined phase both moves are tried from the same
        don't-look-bit queue: 2-opt first at each city, then Or-opt for
        the segments of 1-3 cities that start there.

        Args:
            initial_tour: Starting tour (default: Nearest Neighbor)
            neighbors: Neighbor-list size for 2-opt (None = all cities)
        """
        tour = self._start_tour(initial_tour)
        self._local_search(tour, neighbors)
        return tour, self.calculate_tour_distance(tour)

    def iterated_local_search(self, initial_tour: Optional[List[int]] = None,
                              iterations: Optional[int] = None,
                              neighbors: Optional[int] = 10,
                              time_limit: Optional[float] = None
                              ) -> Tuple[List[int], float]:
        """
        Iterated local search: kick the tour with a random double-bridge
        move, repair it with 2-opt and Or-opt, and keep the result unless
        it is longer than before. Starts and ends with a full local search.

        The kick reconnects three short consecutive pieces of the tour in
        the opposite order (see _TourSearch.double_bridge), and the repair
        looks only at the cities whose edges changed, so beyond a few
        hundred cities the cost of an iteration no longer grows with the
        instance.

        Args:
            initial_tour: Starting tour (default: Nearest Neighbor)
            iterations: Number of kicks. None (default) means
                max(1000, 2 * cities), or no limit if time_limit is given.
            neighbors: Neighbor-list size for the 2-opt repairs after each
                kick (None = all cities)
            time_limit: Stop kicking after about this many seconds. The
                result then depends on the machine's speed as well as on
                the seed.
        """
        if iterations is not None and iterations < 0:
            raise ValueError(f"iterations must be non-negative (got {iterations})")
        if time_limit is not None and time_limit < 0:
            raise ValueError(f"time_limit must be non-negative (got {time_limit})")
        tour = self._start_tour(initial_tour)
        n = len(tour)
        self._local_search(tour)
        if n < 8:
            return tour, self.calculate_tour_distance(tour)
        if iterations is None and time_limit is None:
            iterations = max(1000, 2 * n)

        start = tour[0]
        rng = self.rng
        tolerance = self.tolerance
        search = _TourSearch(self, tour)
        pos = search.pos
        repair_two_opt = self._neighbor_lists(neighbors)
        repair_or_opt = self._neighbor_lists(8)
        max_piece = min(_KICK_PIECE, (n - 2) // 3)
        best_tour, best_pos = tour[:], pos[:]
        deadline = None if time_limit is None else time.perf_counter() + time_limit
        kicks = 0
        while iterations is None or kicks < iterations:
            if deadline is not None and kicks % 16 == 0 \
                    and time.perf_counter() >= deadline:
                break
            kicks += 1
            change, touched = search.double_bridge(rng, max_piece)
            change += search.optimize(repair_two_opt, repair_or_opt,
                                      queue=touched, prune=True)
            if change < tolerance:
                # Shorter, or as long: moving between equal tours lets the
                # search cross plateaus (common with integer distances).
                best_tour[:] = tour
                best_pos[:] = pos
            else:
                tour[:] = best_tour
                pos[:] = best_pos

        self._rotate_to(tour, start)
        self._local_search(tour)
        return tour, self.calculate_tour_distance(tour)

    def three_opt(self, initial_tour: Optional[List[int]] = None,
                  max_iterations: int = 100) -> Tuple[List[int], float]:
        """
        3-opt local search - more powerful but much slower than 2-opt.

        For every three tour edges, evaluates in O(1) each of the seven
        ways to remove them and reconnect the tour (three are 2-opt moves,
        the other four replace all three edges) and applies the best one
        if it shortens the tour. Passes over all triples repeat until one
        finds nothing, so the result is a 3-opt local optimum unless
        `max_iterations` passes run out first.

        Time Complexity: O(n³) per pass, practical up to about 150 cities.
        The returned tour starts with the same city as the input tour.
        """
        tour = self._start_tour(initial_tour)
        n = len(tour)
        if n < 4:
            return tour, self.calculate_tour_distance(tour)
        D = self._dist
        tolerance = self.tolerance
        start = tour[0]
        improved = True
        passes = 0

        while improved and passes < max_iterations:
            improved = False
            passes += 1
            # Edges (tour[i-1], tour[i]), (tour[j-1], tour[j]) and
            # (tour[k-1], tour[k]) cut the tour into  ..A [B..C] [Dn..E] F..
            # where B..C is tour[i:j] and Dn..E is tour[j:k].
            for i in range(n - 2):
                A = tour[i - 1]             # no move changes tour[i - 1]
                DA = D[A]
                for j in range(i + 1, n - 1):
                    B, C, Dn = tour[i], tour[j - 1], tour[j]
                    DB, DC, DD = D[B], D[C], D[Dn]
                    ab, ac, ad, cd, bd = DA[B], DA[C], DA[Dn], DC[Dn], DB[Dn]
                    for k in range(j + 1, n):
                        E, F = tour[k - 1], tour[k]
                        DE = D[E]
                        ef, ae, bf, cf, df = DE[F], DA[E], DB[F], DC[F], DD[F]
                        best = ab + cd + ef - tolerance
                        move = 0
                        length = ac + bd + ef               # A [C..B] [Dn..E] F
                        if length < best:
                            best, move = length, 1
                        length = ab + DC[E] + df            # A [B..C] [E..Dn] F
                        if length < best:
                            best, move = length, 2
                        length = ae + cd + bf               # A [E..Dn] [C..B] F
                        if length < best:
                            best, move = length, 3
                        length = ad + DE[B] + cf            # A [Dn..E] [B..C] F
                        if length < best:
                            best, move = length, 4
                        length = ac + DB[E] + df            # A [C..B] [E..Dn] F
                        if length < best:
                            best, move = length, 5
                        length = ad + DE[C] + bf            # A [Dn..E] [C..B] F
                        if length < best:
                            best, move = length, 6
                        length = ae + bd + cf               # A [E..Dn] [B..C] F
                        if length < best:
                            best, move = length, 7
                        if not move:
                            continue

                        first, second = tour[i:j], tour[j:k]
                        if move == 1:
                            first.reverse()
                        elif move == 2:
                            second.reverse()
                        elif move == 3:
                            first, second = second[::-1], first[::-1]
                        elif move == 4:
                            first, second = second, first
                        elif move == 5:
                            first.reverse()
                            second.reverse()
                        elif move == 6:
                            first, second = second, first[::-1]
                        else:
                            first, second = second[::-1], first
                        tour[i:k] = first + second
                        improved = True
                        B, C, Dn = tour[i], tour[j - 1], tour[j]
                        DB, DC, DD = D[B], D[C], D[Dn]
                        ab, ac, ad, cd, bd = DA[B], DA[C], DA[Dn], DC[Dn], DB[Dn]

        self._rotate_to(tour, start)
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
    def _annealing_temperatures(propose, samples: int = 500,
                                tolerance: float = EPSILON) -> Tuple[float, float]:
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
            if move is not None and move[2] > tolerance:
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
                            max_iterations: Optional[int] = None,
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
            max_iterations: Maximum number of moves attempted. None
                (default) means max(100000, 500 * cities): annealing needs
                a budget that grows with the instance.
            polish: Finish with 2-opt + Or-opt on the best tour found
            neighbors: Draw moves from each city's k nearest cities (None =
                reverse a uniformly random segment)
        """
        n = self.n_cities
        if max_iterations is None:
            max_iterations = max(100000, 500 * n)
        # Start with nearest neighbor solution
        current_tour, current_distance = self.nearest_neighbor()
        if n < 4:
            return current_tour, current_distance

        rng = self.rng
        if neighbors is None:
            search = None

            def propose():
                return self._random_two_opt_move(current_tour)
        else:
            near = self._neighbor_lists(neighbors)
            search = _TourSearch(self, current_tour)
            pos = search.pos
            D = self._dist
            last = n - 1

            def propose():
                """2-opt move adding edge (a, c), where c is one of the
                nearest cities of a: with b and d the cities after a and
                c, reverse tour positions pos[b]..pos[c] (cyclic). Returns
                those two positions and the change in length."""
                a = rng.randrange(n)
                c = near[a][rng.randrange(len(near[a]))]
                pa, pc = pos[a], pos[c]
                b = current_tour[pa + 1 if pa != last else 0]
                d = current_tour[pc + 1 if pc != last else 0]
                if b == c or d == a:
                    return None
                return pos[b], pc, D[a][c] + D[b][d] - D[a][b] - D[c][d]

        if initial_temp is None or min_temp is None:
            auto_start, auto_end = self._annealing_temperatures(
                propose, tolerance=self.tolerance)
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
        tolerance = self.tolerance
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
                if search is None:
                    current_tour[i:j] = reversed(current_tour[i:j])
                else:
                    search.reverse(i, j)
                current_distance += delta

                if current_distance < best_distance - tolerance:
                    best_tour = current_tour.copy()
                    best_distance = current_distance

        if polish:
            self._local_search(best_tour)
        return best_tour, self.calculate_tour_distance(best_tour)

    def genetic_algorithm(self, population_size: int = 30,
                          generations: int = 40,
                          mutation_rate: float = 0.2,
                          elite_size: int = 4,
                          neighbors: Optional[int] = 10) -> Tuple[List[int], float]:
        """
        Memetic genetic algorithm for TSP.

        - Initial population: Nearest Neighbor tours from different start
          cities (plus random tours if the population is larger than n).
        - Each generation: tournament selection and order crossover (OX).
          With probability `mutation_rate` a child is mutated by a
          double-bridge kick (see _TourSearch.double_bridge), a change the
          local search cannot simply undo.
        - Every tour is improved by local search (2-opt and Or-opt).
        - A child as long as a tour already in the next generation is
          almost always a copy of it, so it is kicked and improved again:
          a population of copies has nothing left to recombine.
        - The `elite_size` best tours survive unchanged.

        Args:
            population_size: Tours per generation
            generations: Number of generations
            mutation_rate: Probability that a child is mutated
            elite_size: Best tours copied to the next generation
            neighbors: Neighbor-list size for the 2-opt step (None = all)
        """
        if population_size < 1:
            raise ValueError(f"population_size must be at least 1 (got {population_size})")
        if generations < 0:
            raise ValueError(f"generations must be non-negative (got {generations})")
        if not 0 <= mutation_rate <= 1:
            raise ValueError(f"mutation_rate must be between 0 and 1 (got {mutation_rate})")
        if elite_size < 0:
            raise ValueError(f"elite_size must be non-negative (got {elite_size})")
        n = self.n_cities
        if n < 4:
            tour = list(range(n))
            return tour, self.calculate_tour_distance(tour)

        rng = self.rng
        two_opt_lists = self._neighbor_lists(neighbors)
        or_opt_lists = self._neighbor_lists(8)
        elite_size = min(elite_size, population_size)
        max_piece = min(_KICK_PIECE, (n - 2) // 3)

        def improve(tour, kick=False):
            """Local search in place, after a double-bridge kick if asked."""
            start = tour[0]
            search = _TourSearch(self, tour)
            if kick and max_piece:
                search.double_bridge(rng, max_piece)
            search.optimize(two_opt_lists, or_opt_lists, prune=True)
            self._rotate_to(tour, start)
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

        # Initialize population
        starts = rng.sample(range(n), min(n, population_size))
        population = [improve(self.nearest_neighbor(c)[0]) for c in starts]
        while len(population) < population_size:
            population.append(improve(rng.sample(range(n), n)))
        distances = [self.calculate_tour_distance(t) for t in population]
        # Tour lengths closer than this count as equal (a small fraction of
        # an average edge: far above rounding error, far below any real
        # difference between two tours).
        resolution = 1e-7 * min(distances) / n or 1.0

        for generation in range(generations):
            # Elite preservation
            order = sorted(range(len(population)), key=distances.__getitem__)
            elite = [population[i] for i in order[:elite_size]]
            elite_distances = [distances[i] for i in order[:elite_size]]
            taken = {round(d / resolution) for d in elite_distances}

            # Selection, crossover, mutation, local search
            selected = selection(population, distances)
            children, child_distances = [], []
            for i in range(population_size - elite_size):
                parent1 = selected[i]
                parent2 = selected[(i + 1) % len(selected)]
                child = improve(crossover(parent1, parent2),
                                kick=rng.random() < mutation_rate)
                distance = self.calculate_tour_distance(child)
                if round(distance / resolution) in taken:
                    improve(child, kick=True)
                    distance = self.calculate_tour_distance(child)
                taken.add(round(distance / resolution))
                children.append(child)
                child_distances.append(distance)

            population = elite + children
            distances = elite_distances + child_distances

        best_idx = min(range(len(population)), key=distances.__getitem__)
        best_tour = population[best_idx]
        self._local_search(best_tour)
        return best_tour, self.calculate_tour_distance(best_tour)

    # ==================== VISUALIZATION ====================

    def visualize_tour(self, tour: List[int], title: str = "TSP Tour",
                       save_path: Optional[str] = None, show: bool = True,
                       labels: Optional[bool] = None):
        """
        Visualize a TSP tour.

        Args:
            tour: Tour to draw
            title: Plot title
            save_path: If given, save the figure to this path
            show: Whether to display the figure (set False in scripts/tests)
            labels: Write each city's name beside it. None (default) does
                so for up to 50 cities; more labels than that overlap.

        Returns:
            The matplotlib Figure
        """
        tour = list(tour)
        if not tour:
            raise ValueError("cannot draw an empty tour")
        import matplotlib.pyplot as plt  # optional dependency

        if labels is None:
            labels = self.n_cities <= 50
        # Lines and markers shrink as the instance grows, so that a
        # thousand cities can still be told apart.
        shrink = min(1.0, 50 / self.n_cities) ** 0.5

        fig, ax = plt.subplots(figsize=(10, 8))

        x = self.cities[:, 0]
        y = self.cities[:, 1]

        # Plot tour (closed)
        closed = tour + tour[:1]
        ax.plot(x[closed], y[closed], color='#2a78d6',
                linewidth=max(0.8, 2 * shrink), solid_joinstyle='round', zorder=2)

        # Plot cities
        ax.scatter(x, y, s=max(4.0, 60 * shrink ** 2), color='#52514e',
                   edgecolor='white', linewidth=1.5 * shrink, zorder=3)

        # Add city labels
        if labels:
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

    def compare_algorithms(self, algorithms: Optional[List[str]] = None,
                           options: Optional[Dict[str, Dict[str, Any]]] = None
                           ) -> Dict:
        """
        Compare performance of different algorithms.

        Args:
            algorithms: List of algorithm names to compare (keys of
                TSPSolver.ALGORITHMS). Exact algorithms are skipped above
                their size limit (EXACT_LIMITS: brute force 10, Held-Karp 20).
            options: Keyword arguments for individual algorithms, by name,
                e.g. {'iterated_local_search': {'time_limit': 5}}

        Returns:
            Dictionary with results for each algorithm
        """
        if algorithms is None:
            algorithms = ['nearest_neighbor', '2-opt', 'simulated_annealing']
        options = options or {}

        unknown = [a for a in list(algorithms) + list(options)
                   if a not in self.ALGORITHMS]
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
            tour, distance = getattr(self, self.ALGORITHMS[algo])(
                **options.get(algo, {}))
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
        self.prefix = solver._nearest(min(prefix, n - 1))
        if prefix >= n - 1:
            self._tails = None                    # rows are already complete
        else:
            self._tails = [_LazyTail(solver.distance_matrix, city, self.prefix[city])
                           for city in range(n)]

    def __len__(self) -> int:
        return len(self.prefix)

    def __iter__(self):
        return (self[city] for city in range(len(self)))

    def __getitem__(self, city: int):
        if not 0 <= city < len(self.prefix):
            raise IndexError(city)
        if self._tails is None:
            return self.prefix[city]
        # chain() only asks the tail for its items once the prefix is used up.
        return chain(self.prefix[city], self._tails[city])


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


class _TourSearch:
    """
    A tour stored as an array, with each city's position in it, and the
    local-search moves that work on that representation.

    Every change to the tour is a reversal of a stretch of positions
    (`reverse`), which keeps the position index exact. A 2-opt move is one
    reversal, an Or-opt move two or three, a double-bridge kick four.

    The tour may hold a subset of the solver's cities.
    """

    __slots__ = ('D', 'tolerance', 'tour', 'n', 'pos', 'queued')

    def __init__(self, solver: 'TSPSolver', tour: List[int]):
        self.D = solver._dist
        self.tolerance = solver.tolerance
        self.tour = tour                          # modified in place
        self.n = len(tour)
        self.pos = [-1] * solver.n_cities         # -1: not in the tour
        for idx, city in enumerate(tour):
            self.pos[city] = idx
        self.queued = [False] * solver.n_cities   # don't-look bits, inverted

    def reverse(self, i: int, j: int) -> None:
        """
        Reverse tour positions i..j (cyclic, inclusive), or every other
        position if that side is shorter. Read as a cycle the tour is the
        same either way, and no reversal writes more than n/2 positions.
        """
        tour, pos, n = self.tour, self.pos, self.n
        length = j - i + 1
        if length <= 0:
            length += n
        if 2 * length > n:
            if length == n:
                return
            i, j = j + 1, i - 1
            if i == n:
                i = 0
            if j < 0:
                j = n - 1
        if i <= j:
            segment = tour[i:j + 1]
            segment.reverse()
            tour[i:j + 1] = segment
            for city in segment:
                pos[city] = i
                i += 1
        else:                                     # wraps past the array end
            segment = tour[i:] + tour[:j + 1]
            segment.reverse()
            head = n - i
            tour[i:] = segment[:head]
            tour[:j + 1] = segment[head:]
            for city in segment[:head]:
                pos[city] = i
                i += 1
            i = 0
            for city in segment[head:]:
                pos[city] = i
                i += 1

    def exchange(self, a: int, b: int, c: int, d: int) -> None:
        """
        2-opt move named by its cities: replace tour edges (a, b) and
        (c, d) by (a, c) and (b, d). The two edges must run the same way
        along the tour (b after a and d after c, or both before).
        """
        pos = self.pos
        following = pos[a] + 1
        if self.tour[following if following < self.n else 0] == b:
            self.reverse(pos[b], pos[c])          # a [b .. c] d
        else:
            self.reverse(pos[c], pos[b])          # d [c .. b] a

    def optimize(self, two_opt_neighbors=None, or_opt_neighbors=None,
                 queue: Optional[List[int]] = None,
                 max_passes: Optional[int] = None,
                 until_optimal: bool = False, max_segment: int = 3,
                 prune: bool = False) -> float:
        """
        Apply improving 2-opt and Or-opt moves until the queue of cities
        to examine is empty. Returns the change in tour length (<= 0).

        Only cities in `queue` (all cities if None) are examined at first;
        the cities at the ends of every edge a move changes are queued
        again (don't-look bits). At each city, 2-opt is tried first and
        Or-opt only if 2-opt finds nothing there. Either kind of move is
        skipped if its neighbor lists are None.

        2-opt: for city a and each tour direction, candidate partners c
        are scanned in order of distance, and the scan stops once
        D[a][c] >= D[a][next(a)], because an improving move must add an
        edge shorter than one it removes. The first improving move found
        is applied.

        Or-opt: each segment of 1..max_segment cities that starts or ends
        at city a may move, in either orientation, into a tour edge next
        to one of the nearest neighbors c of its end cities. The best such
        move is applied. With `prune`, the scan of an end city's neighbors
        stops at the first one that is at least as far away as the gain
        from taking the segment out. An improving move rarely needs a
        longer new edge than that and the scan gets several times cheaper,
        but a few moves are missed: iterated local search prunes while it
        repairs the tour after a kick, not in its final local search.

        Don't-look bits alone can miss a move that appears at a city whose
        own edges did not change. With `until_optimal`, every city is
        examined again after the queue empties, until a full sweep finds
        no move: the tour is then a local optimum for both kinds of move,
        within the given neighbor lists.

        Passes (`max_passes`): the initially queued cities form pass 1; a
        city queued again while a pass-p city is examined belongs to pass
        p + 1, and cities beyond pass `max_passes` are not examined
        (0 = no change).
        """
        n = self.n
        if n < 4 or max_passes == 0:
            return 0.0
        tour, pos, D, queued = self.tour, self.pos, self.D, self.queued
        reverse, exchange = self.reverse, self.exchange
        tolerance = self.tolerance
        last = n - 1
        full_sweep = queue is None
        queue = tour[:] if queue is None else list(queue)
        for city in queue:
            queued[city] = True
        pass_of = None if max_passes is None else [1] * len(pos)
        current_pass = 0
        change = 0.0
        sweep_moved = False

        while True:
            if not queue:
                if not until_optimal or (full_sweep and not sweep_moved):
                    break
                queue = tour[:]
                for city in queue:
                    queued[city] = True
                full_sweep, sweep_moved = True, False
            a = queue.pop()
            queued[a] = False
            if pass_of is not None:
                current_pass = pass_of[a]
                if current_pass > max_passes:
                    continue
            pa = pos[a]
            moved = None                # cities at the ends of changed edges

            if two_opt_neighbors is not None:
                Da = D[a]
                for direction in (1, -1):
                    b = tour[pa + direction if pa != last or direction < 0 else 0]
                    d_ab = Da[b]
                    for c in two_opt_neighbors[a]:
                        d_ac = Da[c]
                        if d_ac >= d_ab:
                            break
                        pc = pos[c]
                        if pc < 0:
                            continue
                        d = tour[pc + direction if pc != last or direction < 0 else 0]
                        if c == b or d == a:
                            continue
                        delta = d_ac + D[b][d] - d_ab - D[c][d]
                        if delta < -tolerance:
                            if direction == 1:
                                reverse(pos[b], pc)      # a b ... c d -> a c ... b d
                            else:
                                reverse(pa, pos[d])      # b a ... d c -> b d ... a c
                            change += delta
                            moved = (a, b, c, d)
                            break
                    if moved is not None:
                        break

            if moved is None and or_opt_neighbors is not None:
                best_gain = tolerance
                best = None
                for length in range(1, max_segment + 1):
                    if n < length + 3:
                        break
                    # The segments of this length with city a at one end:
                    # tour positions first .. first + length - 1 (cyclic).
                    # A sweep over all cities reaches every segment from
                    # its first city, so it need not also look backwards.
                    for first in ((pa,) if length == 1 or full_sweep
                                  else (pa, pa - length + 1)):
                        if first < 0:
                            first += n
                        end = first + length - 1
                        if end > last:
                            end -= n
                        s, e = tour[first], tour[end]
                        p, nx = tour[first - 1], tour[end + 1 if end != last else 0]
                        removal_gain = D[p][s] + D[e][nx] - D[p][nx]
                        # Inserting the segment costs at least -D[s][e]
                        # (triangle inequality), which bounds the total gain.
                        if removal_gain + (0.0 if prune or length == 1 else D[s][e]) \
                                <= best_gain:
                            continue
                        for x, y in (((s, e),) if length == 1 else ((s, e), (e, s))):
                            Dx, Dy = D[x], D[y]
                            for c in or_opt_neighbors[x]:
                                d_xc = Dx[c]
                                if prune and d_xc >= removal_gain:
                                    break
                                pc = pos[c]
                                if pc < 0:
                                    continue
                                offset = pc - first
                                if offset < 0:
                                    offset += n
                                if offset < length:
                                    continue          # c is in the segment
                                Dc = D[c]
                                # c's two tour edges, (c, w), are the places
                                # to insert the segment.
                                for pw in (pc + 1 if pc != last else 0, pc - 1):
                                    w = tour[pw]
                                    offset = pw - first
                                    if offset < 0:
                                        offset += n
                                    if offset < length:
                                        continue      # w is in the segment
                                    saved = removal_gain + Dc[w]
                                    gain = saved - d_xc - Dy[w]    # c-x..y-w
                                    if gain > best_gain:
                                        best_gain = gain
                                        best = (s, e, p, nx, c, w, x)
                                    if length > 1:
                                        gain = saved - Dy[c] - Dx[w]    # c-y..x-w
                                        if gain > best_gain:
                                            best_gain = gain
                                            best = (s, e, p, nx, c, w, y)
                if best is not None:
                    s, e, p, nx, c, w, beside_c = best
                    # Call the insertion edge u -> v, in the direction in
                    # which the segment runs s -> e.
                    following = pos[c] + 1
                    if tour[following if following < n else 0] == w:
                        u, v = c, w
                    else:
                        u, v = w, c
                    exchange(p, s, u, v)
                    exchange(p, u, nx, e)             # now  p nx .. u e..s v
                    if s != e and (beside_c == s) == (u == c):
                        exchange(u, e, s, v)          # now  p nx .. u s..e v
                    change -= best_gain
                    moved = (p, nx, s, e, u, v)

            if moved is not None:
                sweep_moved = True
                for city in moved:
                    if not queued[city]:
                        queued[city] = True
                        queue.append(city)
                        if pass_of is not None:
                            pass_of[city] = current_pass + 1
        return change

    def double_bridge(self, rng: random.Random,
                      max_piece: int) -> Tuple[float, List[int]]:
        """
        Random double-bridge kick. The tour is cut into A B C D, where B,
        C and D are consecutive pieces of 1..max_piece cities at a random
        place, and reconnected as A D C B. All four connecting edges
        change, and the move cannot be reversed by a single 2-opt, Or-opt
        or 3-opt move, which is what makes it a useful perturbation.
        Keeping the pieces short keeps the damage, and the repair, local.

        Returns the change in tour length and the eight cities at the cut
        edges. Requires len(tour) > 3 * max_piece.
        """
        tour, n, D = self.tour, self.n, self.D
        b_start = int(rng.random() * n)
        c_start = b_start + 1 + int(rng.random() * max_piece)
        d_start = c_start + 1 + int(rng.random() * max_piece)
        a_start = d_start + 1 + int(rng.random() * max_piece)
        a1 = tour[b_start - 1]                                   # A's last city
        b0, b1 = tour[b_start], tour[(c_start - 1) % n]
        c0, c1 = tour[c_start % n], tour[(d_start - 1) % n]
        d0, d1 = tour[d_start % n], tour[(a_start - 1) % n]
        a0 = tour[a_start % n]                                   # A's first city
        change = (D[a1][d0] + D[d1][c0] + D[c1][b0] + D[b1][a0]
                  - D[a1][b0] - D[b1][c0] - D[c1][d0] - D[d1][a0])
        self.exchange(a1, b0, d1, a0)     # B C D reversed as one stretch,
        self.exchange(a1, d1, d0, c1)     # then each piece turned back:
        self.exchange(d1, c1, c0, b1)     # a1 [d0..d1] [c0..c1] [b0..b1] a0
        self.exchange(c1, b1, b0, a0)
        return change, [a1, b0, b1, c0, c1, d0, d1, a0]
