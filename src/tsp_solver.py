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
    - Nearest Neighbor (greedy heuristic)
    - Nearest Insertion (constructive heuristic)
    - 2-Opt (local search improvement)
    - 3-Opt (local search improvement)
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
            tour = [start_city] + list(perm)
            distance = self.calculate_tour_distance(tour)
            if distance < min_distance:
                min_distance = distance
                best_tour = tour

        return best_tour, self.calculate_tour_distance(best_tour)

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

    def two_opt(self, initial_tour: Optional[List[int]] = None,
                max_iterations: int = 1000) -> Tuple[List[int], float]:
        """
        2-opt local search improvement.

        Considers every pair of non-adjacent edges, including the edge that
        closes the tour, and evaluates each move in O(1).

        Time Complexity: O(n² * iterations)
        """
        if initial_tour is None:
            tour, _ = self.nearest_neighbor()
        else:
            tour = list(initial_tour)

        n = len(tour)
        D = self._dist
        improved = n >= 4
        iterations = 0

        while improved and iterations < max_iterations:
            improved = False
            for i in range(n - 1):
                # When i == 0, j == n - 1 would pick the edge adjacent to
                # (tour[0], tour[1]) through the wrap-around, so skip it.
                for j in range(i + 2, n if i > 0 else n - 1):
                    a, b = tour[i], tour[i + 1]
                    c, d = tour[j], tour[(j + 1) % n]
                    # Replace edges (a,b) and (c,d) with (a,c) and (b,d)
                    delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]
                    if delta < -EPSILON:
                        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                        improved = True

            iterations += 1

        return tour, self.calculate_tour_distance(tour)

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

    def simulated_annealing(self, initial_temp: float = 1000,
                            cooling_rate: Optional[float] = None,
                            min_temp: float = 1,
                            max_iterations: int = 100000) -> Tuple[List[int], float]:
        """
        Simulated annealing metaheuristic.

        Uses random 2-opt moves evaluated in O(1).

        Args:
            initial_temp: Starting temperature
            cooling_rate: Geometric cooling factor per iteration. If None, it
                is chosen so the temperature reaches min_temp exactly at
                max_iterations, so the full iteration budget is used.
            min_temp: Temperature at which the search stops
            max_iterations: Maximum number of moves attempted
        """
        n = self.n_cities
        # Start with nearest neighbor solution
        current_tour, current_distance = self.nearest_neighbor()
        if n < 4:
            return current_tour, current_distance

        if cooling_rate is None:
            cooling_rate = (min_temp / initial_temp) ** (1 / max_iterations)

        D = self._dist
        rng = self.rng
        best_tour = current_tour.copy()
        best_distance = current_distance

        temp = initial_temp
        iteration = 0

        while temp > min_temp and iteration < max_iterations:
            temp *= cooling_rate
            iteration += 1

            # Random 2-opt move: reverse current_tour[i:j]
            i, j = sorted(rng.sample(range(n + 1), 2))
            if j - i < 2 or (i == 0 and j == n):
                continue

            a, b = current_tour[i - 1], current_tour[i]
            c, d = current_tour[j - 1], current_tour[j % n]
            delta = D[a][c] + D[b][d] - D[a][b] - D[c][d]

            # Accept or reject move
            if delta < 0 or rng.random() < math.exp(-delta / temp):
                current_tour[i:j] = reversed(current_tour[i:j])
                current_distance += delta

                if current_distance < best_distance - EPSILON:
                    best_tour = current_tour.copy()
                    best_distance = current_distance

        return best_tour, self.calculate_tour_distance(best_tour)

    def genetic_algorithm(self, population_size: int = 100,
                          generations: int = 500,
                          mutation_rate: float = 0.02,
                          elite_size: int = 20) -> Tuple[List[int], float]:
        """
        Genetic algorithm for TSP.

        Uses order crossover (OX) and swap mutation.
        """
        n = self.n_cities
        if n < 4:
            tour = list(range(n))
            return tour, self.calculate_tour_distance(tour)

        rng = self.rng

        def create_individual():
            """Create random tour."""
            return rng.sample(range(n), n)

        def fitness(individual):
            """Fitness is inverse of distance."""
            return 1 / self.calculate_tour_distance(individual)

        def selection(population, scores):
            """Tournament selection."""
            tournament_size = min(5, len(population))
            candidates = list(zip(population, scores))
            selected = []
            for _ in range(len(population)):
                tournament = rng.sample(candidates, tournament_size)
                winner = max(tournament, key=lambda x: x[1])
                selected.append(winner[0])
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
            """Swap mutation."""
            if rng.random() < mutation_rate:
                i, j = rng.sample(range(len(individual)), 2)
                individual[i], individual[j] = individual[j], individual[i]
            return individual

        # Initialize population
        population = [create_individual() for _ in range(population_size)]

        for generation in range(generations):
            # Calculate fitness
            scores = [fitness(ind) for ind in population]

            # Elite preservation
            elite_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:elite_size]
            elite = [population[i] for i in elite_indices]

            # Selection and crossover
            selected = selection(population, scores)
            children = []

            for i in range(0, population_size - elite_size, 2):
                parent1 = selected[i]
                parent2 = selected[i + 1] if i + 1 < len(selected) else selected[0]
                child1 = crossover(parent1, parent2)
                child2 = crossover(parent2, parent1)
                children.extend([mutate(child1), mutate(child2)])

            # New population
            population = elite + children[:population_size - elite_size]

        # Return best solution
        scores = [fitness(ind) for ind in population]
        best_idx = scores.index(max(scores))
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
        fig = plt.figure(figsize=(10, 8))

        # Plot cities
        x = self.cities[:, 0]
        y = self.cities[:, 1]
        plt.scatter(x, y, c='red', s=200, zorder=5)

        # Add city labels
        for i, name in enumerate(self.city_names):
            plt.annotate(name, (x[i], y[i]), xytext=(5, 5),
                         textcoords='offset points', fontsize=9)

        # Plot tour
        tour_x = [x[tour[i]] for i in range(len(tour))]
        tour_y = [y[tour[i]] for i in range(len(tour))]
        tour_x.append(tour_x[0])  # Close the tour
        tour_y.append(tour_y[0])

        plt.plot(tour_x, tour_y, 'b-', linewidth=2, alpha=0.7)
        plt.plot(tour_x, tour_y, 'bo', markersize=8)

        # Add distance to title
        distance = self.calculate_tour_distance(tour)
        plt.title(f"{title}\nTotal Distance: {distance:.2f}", fontsize=14, fontweight='bold')
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        return fig

    ALGORITHMS = {
        'brute_force': 'brute_force',
        'nearest_neighbor': 'nearest_neighbor',
        'nearest_insertion': 'nearest_insertion',
        '2-opt': 'two_opt',
        '3-opt': 'three_opt',
        'simulated_annealing': 'simulated_annealing',
        'genetic_algorithm': 'genetic_algorithm',
    }

    def compare_algorithms(self, algorithms: Optional[List[str]] = None) -> Dict:
        """
        Compare performance of different algorithms.

        Args:
            algorithms: List of algorithm names to compare (keys of
                TSPSolver.ALGORITHMS). 'brute_force' is skipped when the
                instance has more than 10 cities.

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
            if algo == 'brute_force' and self.n_cities > 10:
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
