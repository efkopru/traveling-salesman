# Traveling Salesman Problem Solver

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Algorithms](https://img.shields.io/badge/Algorithms-7-orange.svg)](#algorithms)

A comprehensive Python implementation of multiple algorithms for solving the Traveling Salesman Problem (TSP), featuring optimized implementations and performance comparisons.

## Features

- **7 Different Algorithms**: From exact solutions to advanced metaheuristics
- **Performance Optimized**: Fast execution even for larger instances
- **Visualization Support**: Tour visualization with matplotlib
- **Extensible Architecture**: Easy to add new algorithms
- **Comprehensive Benchmarking**: Built-in algorithm comparison tools

## Algorithms

| Algorithm | Type | Time Complexity | Strengths | Typical Performance |
|-----------|------|----------------|-----------|---------------------|
| **Brute Force** | Exact | O(n!) | Guarantees optimal solution | Only feasible for n ≤ 10 |
| **Nearest Neighbor** | Greedy | O(n²) | Very fast, simple | 15-20% above optimal |
| **Nearest Insertion** | Constructive | O(n²) | Fast, often better than NN | 15-20% above optimal |
| **2-Opt** | Local Search | O(n²) | Excellent improvement ratio | 5-10% above optimal |
| **3-Opt** | Local Search | O(n³) | Better than 2-opt | 2-5% above optimal |
| **Simulated Annealing** | Metaheuristic | O(n²·iterations) | Escapes local optima | 2-8% above optimal |
| **Genetic Algorithm** | Evolutionary | O(n²·gen·pop) | Good for complex landscapes | 5-15% above optimal |

## Performance Results

Results on a 20-city random instance:

| Algorithm | Distance | Time (s) | Performance vs Best |
|-----------|----------|----------|---------------------|
| **Genetic Algorithm** | 386.43 | 0.6333 | Best (baseline) |
| Simulated Annealing | 423.29 | 0.0090 | +9.53% |
| 2-Opt | 428.10 | 0.0000 | +10.78% |
| Nearest Insertion | 462.66 | 0.0000 | +19.72% |
| Nearest Neighbor | 465.04 | 0.0017 | +20.34% |

*Note: Genetic Algorithm achieves the best solution quality at the cost of higher computation time. For real-time applications, 2-Opt provides excellent quality with minimal computation time.*

### Dependencies

```
numpy>=1.20.0
matplotlib>=3.3.0
pandas>=1.3.0
scipy>=1.7.0
```

## Quick Start

```python
from tsp_solver import TSPSolver, generate_random_cities

# Generate random cities
cities = generate_random_cities(20)

# Create solver
solver = TSPSolver(cities)

# Find best solution quality (slower)
tour, distance = solver.genetic_algorithm()
print(f"Best distance: {distance:.2f}")

# Fast high-quality solution
print(f"Fast distance: {distance:.2f}")
tour, distance = solver.two_opt()

# Visualize the tour
solver.visualize_tour(tour, "Optimized Tour")
```

## Example Output

*Example visualization and results of an optimized tour for 20 cities using the Genetic Algorithm*

![TSP Solution Visualization](images/best%20tour%20figure%201.png)
![TSP Solution Results](images/20%20city%20example.png)

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

# Better quality with 2-Opt
tour_2opt, dist_2opt = solver.two_opt()

# Best quality with Genetic Algorithm
tour_ga, dist_ga = solver.genetic_algorithm()

# Best quality for small instances
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
    'simulated_annealing',
    'genetic_algorithm'
])

# Print comparison table
for algo, data in results.items():
    print(f"{algo}: Distance={data['distance']:.2f}, Time={data['time']:.4f}s")
```


## Algorithm Selection Guide

### When to Use Each Algorithm

**Nearest Neighbor**
- Need instant results (< 0.002s)
- Rough approximation is acceptable
- Starting point for other algorithms

**2-Opt**
- Need very fast results (< 0.001s)
- Good solution quality required
- Real-time applications

**Simulated Annealing**
- Balance between speed and quality
- Medium-sized instances (20-100 cities)
- ~0.01s computation time acceptable

**Genetic Algorithm**
- Best solution quality needed
- Can afford longer computation (0.5-2s)
- Complex solution landscapes
- Instances with 15-50 cities


## Real-World Applications

This TSP solver can be applied to various optimization problems:

- **Logistics and Delivery**: Optimize delivery routes to minimize fuel costs
- **Manufacturing**: Minimize tool path length in CNC machining
- **Circuit Board Design**: Optimize component placement and routing
- **DNA Sequencing**: Find optimal sequence assembly order
- **Tourism Planning**: Create efficient sightseeing routes

## Contributing

Contributions are welcome! Areas for improvement:

- [ ] Lin-Kernighan heuristic implementation
- [ ] Christofides algorithm for guaranteed approximation
- [ ] Parallel processing support
- [ ] GPU acceleration for large instances
- [ ] Web interface for interactive solving
- [ ] Support for asymmetric TSP

## Performance Tips

1. **For best quality (15-30 cities)**: Use Genetic Algorithm with sufficient time budget
2. **For balanced quality/speed (20-100 cities)**: Use Simulated Annealing
3. **For instant results with good quality**: Use 2-Opt
4. **For real-time applications**: Use Nearest Neighbor with optional 2-Opt improvement
5. **For guaranteed optimal (≤10 cities)**: Use Brute Force

## Algorithm Performance Analysis

Based on our 20-city benchmark:

- **Genetic Algorithm**: Achieves best results through population-based search and evolution
- **Simulated Annealing**: Good balance of quality and speed through probabilistic acceptance
- **2-Opt**: Lightning fast with respectable quality through local optimization
- **Nearest Insertion**: Better construction heuristic than Nearest Neighbor
- **Nearest Neighbor**: Fastest but simplest approach

The choice of algorithm depends on your specific requirements:
- **Quality-first**: Genetic Algorithm
- **Speed-first**: 2-Opt or Nearest Insertion  
- **Balanced**: Simulated Annealing

## Future Improvements

Planned enhancements include:

- Variable neighborhood search implementation
- Ant colony optimization
- Machine learning-guided heuristics
- Hybrid algorithms combining multiple approaches
- Support for time windows and capacity constraints

## References

1. Applegate, D. L., Bixby, R. E., Chvatal, V., & Cook, W. J. (2006). *The Traveling Salesman Problem: A Computational Study*
2. Helsgaun, K. (2000). "An effective implementation of the Lin-Kernighan traveling salesman heuristic"
3. Johnson, D. S., & McGeoch, L. A. (1997). "The traveling salesman problem: A case study in local optimization"

## License

MIT License - See [LICENSE](LICENSE) file for details

## Author

GitHub: [@efkopru](https://github.com/efkopru)

---

