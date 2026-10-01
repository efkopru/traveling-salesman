"""The example comparison printed by `python -m tsp_solver demo`."""

from typing import Optional

from .solver import TSPSolver, generate_random_cities


def run_example(show: bool = True, save_path: Optional[str] = None):
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
                  '2-opt+or-opt', 'iterated_local_search',
                  'simulated_annealing', 'genetic_algorithm', 'held_karp']

    results = solver.compare_algorithms(algorithms)

    # Print results
    print("\nAlgorithm Comparison:")
    print("-" * 60)
    print(f"{'Algorithm':<22} {'Distance':<15} {'Time (s)'}")
    print("-" * 60)

    for algo, result in results.items():
        print(f"{algo:<22} {result['distance']:<15.2f} {result['time']:.4f}")

    # Find best solution
    best_algo = min(results.keys(), key=lambda x: results[x]['distance'])
    best_tour = results[best_algo]['tour']
    best_distance = results[best_algo]['distance']

    print("-" * 60)
    print(f"\nBest Solution: {best_algo}")
    print(f"Tour: {' -> '.join([city_names[i] for i in best_tour[:5]])} -> ...")
    print(f"Total Distance: {best_distance:.2f}")

    # Visualize best tour
    if show or save_path:
        solver.visualize_tour(best_tour, f"Best Tour ({best_algo})",
                              save_path=save_path, show=show)

    return solver, results


def run_demo(show: bool = False, save_path: Optional[str] = None):
    """Run the 20-city comparison plus an exact-vs-heuristic check."""
    solver, results = run_example(show=show, save_path=save_path)

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
    gap = (heuristic_dist - exact_dist) / exact_dist * 100
    print(f"Gap: {max(gap, 0.0):.2f}%")  # clamp floating-point noise (-0.00%)
