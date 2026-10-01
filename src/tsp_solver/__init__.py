"""
Traveling Salesman Problem (TSP) Solver
========================================
Exact, constructive, local-search and metaheuristic TSP algorithms with
TSPLIB support, benchmarking and visualization.

Author: Esad Kopru
License: MIT
"""

from .benchmark import TSPBenchmark
from .demo import run_example
from .solver import (DISTANCE_FUNCTIONS, EPSILON, TSPSolver,
                     generate_random_cities)
from .tsplib import TSPLIB_OPTIMA, load_tsplib

__version__ = "0.2.0"

__all__ = [
    "DISTANCE_FUNCTIONS",
    "EPSILON",
    "TSPLIB_OPTIMA",
    "TSPBenchmark",
    "TSPSolver",
    "generate_random_cities",
    "load_tsplib",
    "run_example",
]
