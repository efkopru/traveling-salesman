# Traveling Salesman Problem Solver

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Algorithms](https://img.shields.io/badge/Algorithms-7-orange.svg)](#algorithms)
[![Tests](https://github.com/efkopru/traveling-salesman/actions/workflows/tests.yml/badge.svg)](https://github.com/efkopru/traveling-salesman/actions/workflows/tests.yml)

A comprehensive Python implementation of multiple algorithms for solving the Traveling Salesman Problem (TSP), featuring optimized implementations and performance comparisons.

## Algorithms

| Algorithm | Type | Time Complexity | Strengths |
|-----------|------|----------------|-----------|
| **Brute Force** | Exact | O(n!) | Guarantees optimal solution; only feasible for n ≤ 10 |
| **Nearest Neighbor** | Greedy | O(n²) | Very fast, simple |
| **Nearest Insertion** | Constructive | O(n²) | Fast, similar quality to Nearest Neighbor |
| **2-Opt** | Local Search | O(n²) per pass | Large improvement for very little time |
| **3-Opt** | Local Search | O(n³) per pass | Escapes some 2-opt local optima |
| **Simulated Annealing** | Metaheuristic | O(iterations) | Escapes local optima |
| **Genetic Algorithm** | Evolutionary | O(n·gen·pop) | Population-based; included for comparison |

All local search and annealing moves are evaluated in O(1) from the distance matrix, without recomputing the whole tour.

## Performance Results

Random uniform instances in a 100×100 square, `generate_random_cities(n)` with the default seed and `TSPSolver(..., seed=42)`. "vs Best" is relative to the best result in each column, not to the proven optimum. Times are from a single run and vary by machine.

| Algorithm | 20 cities | 50 cities | 100 cities | Time at 100 cities (s) |
|-----------|-----------|-----------|------------|------------------------|
| Nearest Neighbor | 465.04 (+20.3%) | 694.76 (+22.3%) | 1006.40 (+31.2%) | 0.0003 |
| Nearest Insertion | 462.66 (+19.7%) | 699.63 (+23.2%) | 942.29 (+22.8%) | 0.0012 |
| **2-Opt** | 386.63 (+0.05%) | **568.11** | 783.38 (+2.1%) | 0.0022 |
| **3-Opt** | 386.63 (+0.05%) | **568.11** | **767.35** | 0.13 |
| Simulated Annealing | **386.43** | 579.99 (+2.1%) | 777.86 (+1.4%) | 0.25 |
| Genetic Algorithm | 451.21 (+16.8%) | 965.08 (+69.9%) | 1636.94 (+113%) | 0.75 |

*Note: 2-Opt starting from Nearest Neighbor gives near-best quality in milliseconds. The Genetic Algorithm starts from random tours and uses no local search, so its quality drops quickly as the instance grows.*

### Dependencies

```
pip install -r requirements.txt       # numpy, matplotlib, pandas
pip install -r requirements-dev.txt   # adds pytest
```

## Quick Start

```python
# Run from the src/ directory, or add src/ to your PYTHONPATH
from tsp_solver import TSPSolver, generate_random_cities

# Generate random cities
cities = generate_random_cities(20)

# Create solver (seed makes the randomized algorithms reproducible)
solver = TSPSolver(cities, seed=42)

# Fast high-quality solution
tour, distance = solver.two_opt()
print(f"2-Opt distance: {distance:.2f}")

# Metaheuristic
tour, distance = solver.simulated_annealing()
print(f"Simulated annealing distance: {distance:.2f}")

# Visualize the tour
solver.visualize_tour(tour, "Optimized Tour")
```

To run the full example comparison:

```
python src/tsp_solver.py
```

## Example Output

*Best tour for the 20-city instance (Simulated Annealing, distance 386.43)*

![TSP Solution Visualization](images/best_tour_20.png)

## Usage Examples

### Basic Usage

```python
import numpy as np
from tsp_solver import TSPSolver

# Define city coordinates
cities = np.array([
    [60, 200], [180, 200], [80, 180], [140, 180],
    [20, 160], [100, 160], [200, 160], [140, 140]
])

# Solve with different algorithms
solver = TSPSolver(cities)

# Fast approximation
tour_nn, dist_nn = solver.nearest_neighbor()

# Improve an existing tour with 2-Opt or 3-Opt
tour_2opt, dist_2opt = solver.two_opt(tour_nn)
tour_3opt, dist_3opt = solver.three_opt(tour_2opt)

# Optimal solution for small instances
if len(cities) <= 10:
    tour_exact, dist_exact = solver.brute_force()
```

### Algorithm Comparison

```python
# Compare all algorithms
results = solver.compare_algorithms([
    'nearest_neighbor',
    'nearest_insertion',
    '2-opt',
    '3-opt',
    'simulated_annealing',
    'genetic_algorithm'
])

# Print comparison table
for algo, data in results.items():
    print(f"{algo}: Distance={data['distance']:.2f}, Time={data['time']:.4f}s")
```

Unknown algorithm names raise `ValueError`. `'brute_force'` is skipped for instances with more than 10 cities.

### Saving a plot without displaying it

```python
solver.visualize_tour(tour, "Tour", save_path="tour.png", show=False)
```

## Running Tests

```
pip install -r requirements-dev.txt
pytest
```

## Algorithm Selection Guide

### When to Use Each Algorithm

**Nearest Neighbor / Nearest Insertion**
- Need instant results
- Starting tour for local search

**2-Opt**
- Default choice: near-best quality in milliseconds on these instances
- Real-time applications

**3-Opt**
- Extra quality on larger instances (100+ cities) when ~0.1 s is acceptable
- Run it on a 2-Opt result to polish it

**Simulated Annealing**
- Can escape local optima that 2-Opt gets stuck in
- Results depend on the temperature settings and iteration budget

**Genetic Algorithm**
- Included for comparison; not competitive without local search in this implementation

**Brute Force**
- Guaranteed optimum for ≤ 10 cities


## Real-World Applications

This TSP solver can be applied to various optimization problems:

- **Logistics and Delivery**: Optimize delivery routes to minimize fuel costs
- **Manufacturing**: Minimize tool path length in CNC machining
- **Circuit Board Design**: Optimize component placement and routing
- **DNA Sequencing**: Find optimal sequence assembly order
- **Tourism Planning**: Create efficient sightseeing routes


## Performance Tips

1. **For instant results with good quality**: Use Nearest Neighbor followed by 2-Opt
2. **For the best quality on larger instances**: Apply 3-Opt to the 2-Opt result
3. **For reproducible results**: Pass `seed=` to `TSPSolver`
4. **For guaranteed optimal (≤10 cities)**: Use Brute Force

## References

1. Applegate, D. L., Bixby, R. E., Chvatal, V., & Cook, W. J. (2006). *The Traveling Salesman Problem: A Computational Study*
2. Helsgaun, K. (2000). "An effective implementation of the Lin-Kernighan traveling salesman heuristic"
3. Johnson, D. S., & McGeoch, L. A. (1997). "The traveling salesman problem: A case study in local optimization"

## License

MIT License - See [LICENSE](LICENSE) file for details

## Author

GitHub: [@efkopru](https://github.com/efkopru)
