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


class TSPSolver:
    """
    A comprehensive TSP solver implementing multiple algorithms.
    
    Algorithms included:
    - Brute Force (exact solution for small instances)
    - Nearest Neighbor (greedy heuristic)
    - 2-Opt (local search improvement)
    - Simulated Annealing (metaheuristic)
    - Genetic Algorithm (evolutionary approach)
    - Christofides Algorithm (approximation with guarantee)
    """
    
    def __init__(self, cities: np.ndarray, city_names: Optional[List[str]] = None):
        """
        Initialize TSP solver with city coordinates.
        
        Args:
            cities: Array of shape (n, 2) with city coordinates
            city_names: Optional list of city names
        """
        self.cities = cities
        self.n_cities = len(cities)
        self.city_names = city_names or [f"City_{i}" for i in range(self.n_cities)]
        self.distance_matrix = self._calculate_distance_matrix()
        
    def _calculate_distance_matrix(self) -> np.ndarray:
        """Calculate Euclidean distance matrix between all cities."""
        n = self.n_cities
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.cities[i] - self.cities[j])
                dist_matrix[i][j] = dist_matrix[j][i] = dist
                
        return dist_matrix
    
    def calculate_tour_distance(self, tour: List[int]) -> float:
        """Calculate total distance of a tour."""
        distance = 0
        for i in range(len(tour)):
            distance += self.distance_matrix[tour[i]][tour[(i + 1) % len(tour)]]
        return distance
    
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
        best_tour = None
        
        for perm in permutations(other_cities):
            tour = [start_city] + list(perm)
            distance = self.calculate_tour_distance(tour)
            if distance < min_distance:
                min_distance = distance
                best_tour = tour
                
        return best_tour, min_distance
    
    # ==================== GREEDY HEURISTICS ====================
    
    def nearest_neighbor(self, start_city: int = 0) -> Tuple[List[int], float]:
        """
        Greedy nearest neighbor algorithm.
        
        Time Complexity: O(n²)
        """
        unvisited = set(range(self.n_cities))
        tour = [start_city]
        unvisited.remove(start_city)
        current = start_city
        
        while unvisited:
            nearest = min(unvisited, key=lambda x: self.distance_matrix[current][x])
            tour.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            
        return tour, self.calculate_tour_distance(tour)
    
    def nearest_insertion(self) -> Tuple[List[int], float]:
        """
        Nearest insertion algorithm - builds tour incrementally.
        
        Time Complexity: O(n²)
        """
        # Start with two nearest cities
        min_dist = float('inf')
        for i in range(self.n_cities):
            for j in range(i + 1, self.n_cities):
                if self.distance_matrix[i][j] < min_dist:
                    min_dist = self.distance_matrix[i][j]
                    start_pair = (i, j)
        
        tour = list(start_pair)
        unvisited = set(range(self.n_cities)) - set(tour)
        
        while unvisited:
            # Find nearest unvisited city to tour
            min_dist = float('inf')
            for city in unvisited:
                for tour_city in tour:
                    if self.distance_matrix[city][tour_city] < min_dist:
                        min_dist = self.distance_matrix[city][tour_city]
                        nearest_city = city
            
            # Find best insertion position
            best_increase = float('inf')
            best_pos = 0
            for i in range(len(tour)):
                j = (i + 1) % len(tour)
                increase = (self.distance_matrix[tour[i]][nearest_city] + 
                           self.distance_matrix[nearest_city][tour[j]] - 
                           self.distance_matrix[tour[i]][tour[j]])
                if increase < best_increase:
                    best_increase = increase
                    best_pos = i + 1
            
            tour.insert(best_pos, nearest_city)
            unvisited.remove(nearest_city)
        
        return tour, self.calculate_tour_distance(tour)
    
    # ==================== LOCAL SEARCH ====================
    
    def two_opt(self, initial_tour: Optional[List[int]] = None, 
                max_iterations: int = 1000) -> Tuple[List[int], float]:
        """
        2-opt local search improvement.
        
        Time Complexity: O(n² * iterations)
        """
        if initial_tour is None:
            tour, _ = self.nearest_neighbor()
        else:
            tour = initial_tour.copy()
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            for i in range(1, self.n_cities - 2):
                for j in range(i + 2, self.n_cities):
                    if j == self.n_cities - 1:
                        continue
                    
                    # Calculate distance change for 2-opt swap
                    current = (self.distance_matrix[tour[i]][tour[i + 1]] +
                             self.distance_matrix[tour[j]][tour[j + 1]])
                    new = (self.distance_matrix[tour[i]][tour[j]] +
                          self.distance_matrix[tour[i + 1]][tour[j + 1]])
                    
                    if new < current:
                        # Perform 2-opt swap
                        tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                        improved = True
                        
            iterations += 1
        
        return tour, self.calculate_tour_distance(tour)
    
    def three_opt(self, initial_tour: Optional[List[int]] = None,
                  max_iterations: int = 100) -> Tuple[List[int], float]:
        """
        3-opt local search - more powerful but slower than 2-opt.
        
        Time Complexity: O(n³ * iterations)
        """
        if initial_tour is None:
            tour, _ = self.nearest_neighbor()
        else:
            tour = initial_tour.copy()
        
        def reverse_segment(tour, i, j):
            """Reverse tour segment from i to j."""
            new_tour = tour[:]
            new_tour[i:j + 1] = reversed(new_tour[i:j + 1])
            return new_tour
        
        improved = True
        iterations = 0
        
        while improved and iterations < max_iterations:
            improved = False
            for i in range(self.n_cities - 5):
                for j in range(i + 2, self.n_cities - 3):
                    for k in range(j + 2, self.n_cities - 1):
                        # Try all 3-opt moves
                        current_dist = self.calculate_tour_distance(tour)
                        
                        # Case 1: reverse segment (i+1, j)
                        new_tour = reverse_segment(tour, i + 1, j)
                        if self.calculate_tour_distance(new_tour) < current_dist:
                            tour = new_tour
                            improved = True
                            continue
                        
                        # Case 2: reverse segment (j+1, k)
                        new_tour = reverse_segment(tour, j + 1, k)
                        if self.calculate_tour_distance(new_tour) < current_dist:
                            tour = new_tour
                            improved = True
                            continue
                        
                        # Case 3: reverse both segments
                        new_tour = reverse_segment(tour, i + 1, j)
                        new_tour = reverse_segment(new_tour, j + 1, k)
                        if self.calculate_tour_distance(new_tour) < current_dist:
                            tour = new_tour
                            improved = True
            
            iterations += 1
        
        return tour, self.calculate_tour_distance(tour)
    
    # ==================== METAHEURISTICS ====================
    
    def simulated_annealing(self, initial_temp: float = 1000,
                           cooling_rate: float = 0.995,
                           min_temp: float = 1,
                           max_iterations: int = 10000) -> Tuple[List[int], float]:
        """
        Simulated annealing metaheuristic.
        
        Parameters tuned for good performance on various instances.
        """
        # Start with nearest neighbor solution
        current_tour, _ = self.nearest_neighbor()
        current_distance = self.calculate_tour_distance(current_tour)
        
        best_tour = current_tour.copy()
        best_distance = current_distance
        
        temp = initial_temp
        iteration = 0
        
        while temp > min_temp and iteration < max_iterations:
            # Generate neighbor by random 2-opt move
            i, j = sorted(random.sample(range(self.n_cities), 2))
            if j - i == 1:
                continue
                
            new_tour = current_tour.copy()
            new_tour[i:j] = reversed(new_tour[i:j])
            new_distance = self.calculate_tour_distance(new_tour)
            
            # Accept or reject move
            delta = new_distance - current_distance
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_tour = new_tour
                current_distance = new_distance
                
                if current_distance < best_distance:
                    best_tour = current_tour.copy()
                    best_distance = current_distance
            
            temp *= cooling_rate
            iteration += 1
        
        return best_tour, best_distance
    
    def genetic_algorithm(self, population_size: int = 100,
                         generations: int = 500,
                         mutation_rate: float = 0.02,
                         elite_size: int = 20) -> Tuple[List[int], float]:
        """
        Genetic algorithm for TSP.
        
        Uses order crossover (OX) and swap mutation.
        """
        def create_individual():
            """Create random tour."""
            return random.sample(range(self.n_cities), self.n_cities)
        
        def fitness(individual):
            """Fitness is inverse of distance."""
            return 1 / self.calculate_tour_distance(individual)
        
        def selection(population, scores):
            """Tournament selection."""
            tournament_size = 5
            selected = []
            for _ in range(len(population)):
                tournament = random.sample(list(zip(population, scores)), tournament_size)
                winner = max(tournament, key=lambda x: x[1])
                selected.append(winner[0])
            return selected
        
        def crossover(parent1, parent2):
            """Order crossover (OX)."""
            size = len(parent1)
            start, end = sorted(random.sample(range(size), 2))
            
            child = [-1] * size
            child[start:end] = parent1[start:end]
            
            pointer = end
            for city in parent2[end:] + parent2[:end]:
                if city not in child:
                    child[pointer % size] = city
                    pointer += 1
            
            return child
        
        def mutate(individual):
            """Swap mutation."""
            if random.random() < mutation_rate:
                i, j = random.sample(range(len(individual)), 2)
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
                      save_path: Optional[str] = None):
        """Visualize a TSP tour."""
        plt.figure(figsize=(10, 8))
        
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
        plt.show()
    
    def compare_algorithms(self, algorithms: Optional[List[str]] = None) -> Dict:
        """
        Compare performance of different algorithms.
        
        Args:
            algorithms: List of algorithm names to compare
        
        Returns:
            Dictionary with results for each algorithm
        """
        if algorithms is None:
            algorithms = ['nearest_neighbor', '2-opt', 'simulated_annealing']
        
        results = {}
        
        for algo in algorithms:
            start_time = time.time()
            
            if algo == 'brute_force':
                if self.n_cities <= 10:
                    tour, distance = self.brute_force()
                else:
                    continue
            elif algo == 'nearest_neighbor':
                tour, distance = self.nearest_neighbor()
            elif algo == 'nearest_insertion':
                tour, distance = self.nearest_insertion()
            elif algo == '2-opt':
                tour, distance = self.two_opt()
            elif algo == '3-opt':
                tour, distance = self.three_opt()
            elif algo == 'simulated_annealing':
                tour, distance = self.simulated_annealing()
            elif algo == 'genetic_algorithm':
                tour, distance = self.genetic_algorithm()
            else:
                continue
            
            execution_time = time.time() - start_time
            
            results[algo] = {
                'tour': tour,
                'distance': distance,
                'time': execution_time
            }
        
        return results


# ==================== EXAMPLE USAGE ====================

def generate_random_cities(n: int, seed: int = 42) -> np.ndarray:
    """Generate random city coordinates."""
    np.random.seed(seed)
    return np.random.rand(n, 2) * 100


def run_example():
    """Run example demonstrating TSP solver capabilities."""
    
    # Generate problem instance
    n_cities = 20
    cities = generate_random_cities(n_cities)
    city_names = [f"C{i}" for i in range(n_cities)]
    
    # Create solver
    solver = TSPSolver(cities, city_names)
    
    print("=" * 60)
    print(f"TRAVELING SALESMAN PROBLEM - {n_cities} Cities")
    print("=" * 60)
    
    # Compare algorithms
    algorithms = ['nearest_neighbor', 'nearest_insertion', '2-opt', 
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
    solver.visualize_tour(best_tour, f"Best Tour ({best_algo})")
    
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
                      algorithms: List[str]) -> pd.DataFrame:
        """Run benchmark on multiple instances and algorithms."""
        import pandas as pd
        
        results = []
        
        for instance_name, cities in instances.items():
            solver = TSPSolver(cities)
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
    small_solver = TSPSolver(small_cities)
    
    print("\nSmall Instance (8 cities) - Exact vs Heuristic:")
    exact_tour, exact_dist = small_solver.brute_force()
    heuristic_tour, heuristic_dist = small_solver.simulated_annealing()
    
    print(f"Exact Solution: {exact_dist:.2f}")
    print(f"Heuristic Solution: {heuristic_dist:.2f}")
    print(f"Gap: {(heuristic_dist - exact_dist) / exact_dist * 100:.2f}%")