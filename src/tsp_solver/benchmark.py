"""Benchmark helpers (require pandas)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from .solver import TSPSolver, generate_random_cities
from .tsplib import load_tsplib

if TYPE_CHECKING:
    import pandas as pd


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
        import pandas as pd  # optional dependency

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

    @staticmethod
    def run_tsplib_benchmark(paths: List[str], algorithms: List[str],
                             seed: Optional[int] = None) -> pd.DataFrame:
        """
        Run algorithms on TSPLIB files and report the gap to the published
        optimum (TSPLIB_OPTIMA) where known.
        """
        import pandas as pd  # optional dependency

        results = []
        for path in paths:
            instance = load_tsplib(path)
            solver = TSPSolver.from_tsplib(instance, seed=seed)
            optimum = instance['optimum']
            for algo, result in solver.compare_algorithms(algorithms).items():
                gap = (None if optimum is None
                       else 100 * (result['distance'] - optimum) / optimum)
                results.append({
                    'Instance': instance['name'],
                    'Cities': instance['dimension'],
                    'Algorithm': algo,
                    'Distance': result['distance'],
                    'Optimum': optimum,
                    'Gap (%)': gap,
                    'Time': result['time'],
                })
        return pd.DataFrame(results)
