import os
import pathlib
import random

import numpy as np
import pytest

from tsp_solver import (TSPLIB_OPTIMA, TSPBenchmark, TSPSolver,
                        generate_random_cities, load_tsplib)

HEURISTICS = ["nearest_neighbor", "nearest_insertion", "two_opt", "three_opt",
              "or_opt", "local_search", "iterated_local_search",
              "simulated_annealing", "genetic_algorithm"]


def run(solver, method):
    if method == "genetic_algorithm":
        return solver.genetic_algorithm(generations=50)
    if method == "simulated_annealing":
        return solver.simulated_annealing(max_iterations=5000)
    if method == "iterated_local_search":
        return solver.iterated_local_search(iterations=50)
    return getattr(solver, method)()


def assert_valid(solver, tour, distance):
    assert sorted(tour) == list(range(solver.n_cities))
    assert distance == pytest.approx(solver.calculate_tour_distance(tour))


def test_distance_matrix_matches_pairwise_norm():
    cities = generate_random_cities(7)
    solver = TSPSolver(cities)
    for i in range(7):
        for j in range(7):
            expected = np.linalg.norm(cities[i] - cities[j])
            assert solver.distance_matrix[i, j] == pytest.approx(expected)


def test_generate_random_cities_is_deterministic():
    assert np.array_equal(generate_random_cities(10, seed=3),
                          generate_random_cities(10, seed=3))


@pytest.mark.parametrize("method", HEURISTICS)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 20])
def test_heuristics_return_valid_tours(method, n):
    solver = TSPSolver(generate_random_cities(n), seed=0)
    tour, distance = run(solver, method)
    assert_valid(solver, tour, distance)


@pytest.mark.parametrize("method", HEURISTICS)
@pytest.mark.parametrize("seed", range(5))
def test_heuristics_never_beat_brute_force(method, seed):
    solver = TSPSolver(generate_random_cities(8, seed=seed), seed=seed)
    _, optimal = solver.brute_force()
    _, distance = run(solver, method)
    assert distance >= optimal - 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_local_search_does_not_worsen_start(seed):
    solver = TSPSolver(generate_random_cities(30, seed=seed))
    start, start_distance = solver.nearest_neighbor()
    _, two = solver.two_opt(start)
    _, three = solver.three_opt(start)
    assert two <= start_distance + 1e-9
    assert three <= start_distance + 1e-9


@pytest.mark.parametrize("seed", range(5))
def test_or_opt_and_combined_search_do_not_worsen(seed):
    solver = TSPSolver(generate_random_cities(40, seed=seed), seed=seed)
    start, start_distance = solver.nearest_neighbor()
    _, or_distance = solver.or_opt(start)
    _, two = solver.two_opt(start)
    _, combined = solver.local_search(start)
    _, ils = solver.iterated_local_search(start, iterations=100)
    assert or_distance <= start_distance + 1e-9
    assert combined <= two + 1e-9
    assert ils <= combined + 1e-9


@pytest.mark.parametrize("k", [3, 8])
def test_two_opt_with_neighbor_lists(k):
    solver = TSPSolver(generate_random_cities(60, seed=1))
    start, start_distance = solver.nearest_neighbor()
    tour, distance = solver.two_opt(start, neighbors=k)
    assert_valid(solver, tour, distance)
    assert distance <= start_distance + 1e-9


def test_neighbor_lists_are_sorted_and_exclude_self():
    solver = TSPSolver(generate_random_cities(12, seed=4))
    D = solver.distance_matrix
    for city, row in enumerate(solver._neighbor_lists()):
        row = list(row)
        assert city not in row and len(row) == 11
        assert all(D[city, a] <= D[city, b] for a, b in zip(row, row[1:]))
    assert all(len(row) == 3 for row in solver._neighbor_lists(3))


def test_iterated_local_search_is_reproducible():
    cities = generate_random_cities(30)
    first = TSPSolver(cities, seed=9).iterated_local_search(iterations=100)
    second = TSPSolver(cities, seed=9).iterated_local_search(iterations=100)
    assert first == second


def test_two_opt_max_iterations_counts_passes():
    solver = TSPSolver(generate_random_cities(50, seed=2))
    start, start_distance = solver.nearest_neighbor()
    zero_tour, zero = solver.two_opt(start, max_iterations=0)
    _, one_pass = solver.two_opt(start, max_iterations=1)
    _, full = solver.two_opt(start)
    assert zero_tour == start and zero == pytest.approx(start_distance)
    assert full <= one_pass < start_distance
    assert full == pytest.approx(solver.two_opt(start, max_iterations=1000)[1])


@pytest.mark.parametrize("method", ["two_opt", "or_opt", "local_search",
                                    "iterated_local_search"])
def test_local_search_keeps_first_city(method):
    solver = TSPSolver(generate_random_cities(40, seed=12), seed=0)
    start, _ = solver.nearest_neighbor(start_city=7)
    kwargs = {"iterations": 50} if method == "iterated_local_search" else {}
    tour, _ = getattr(solver, method)(start, **kwargs)
    assert tour[0] == 7


@pytest.mark.parametrize("k", [0, -3])
def test_invalid_neighbor_count_is_rejected(k):
    solver = TSPSolver(generate_random_cities(10, seed=1), seed=1)
    with pytest.raises(ValueError):
        solver.two_opt(neighbors=k)
    with pytest.raises(ValueError):
        solver.simulated_annealing(neighbors=k)


@pytest.mark.parametrize("cities", [
    generate_random_cities(40, seed=3),
    # duplicated and grid points create many distance ties
    np.vstack([generate_random_cities(15, seed=4)] * 2),
    np.array([(i, j) for i in range(6) for j in range(6)]),
])
def test_neighbor_order_matches_stable_sort(cities):
    solver = TSPSolver(cities)
    expected = [[int(c) for c in np.argsort(row, kind="stable") if c != i]
                for i, row in enumerate(solver.distance_matrix)]
    assert [list(row) for row in solver._neighbor_lists()] == expected  # lazy tail
    for k in (1, 5, 20):
        assert solver._neighbor_lists(k) == [row[:k] for row in expected]
    assert solver._neighbor_lists(5) is solver._neighbor_lists(5)  # cached


@pytest.mark.parametrize("cities", [[], [1, 2, 3], [[1, 2, 3]], np.zeros((0, 2))])
def test_invalid_cities_are_rejected(cities):
    with pytest.raises(ValueError):
        TSPSolver(cities)


def test_city_names_must_match_cities():
    with pytest.raises(ValueError):
        TSPSolver(generate_random_cities(3), city_names=["a", "b"])


def test_two_opt_does_not_mutate_input():
    solver = TSPSolver(generate_random_cities(15))
    start, _ = solver.nearest_neighbor()
    original = list(start)
    solver.two_opt(start)
    assert start == original


def test_two_opt_is_locally_optimal():
    solver = TSPSolver(generate_random_cities(40, seed=7))
    tour, _ = solver.two_opt()
    D = solver.distance_matrix
    n = len(tour)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            a, b = tour[i], tour[i + 1]
            c, d = tour[j], tour[(j + 1) % n]
            assert D[a, c] + D[b, d] >= D[a, b] + D[c, d] - 1e-9


def test_two_opt_finds_optimum_on_circle():
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    cities = np.column_stack([np.cos(angles), np.sin(angles)])
    solver = TSPSolver(cities)
    scrambled = [0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]
    _, distance = solver.two_opt(scrambled)
    assert distance == pytest.approx(12 * 2 * np.sin(np.pi / 12))


def test_brute_force_rejects_large_instances():
    with pytest.raises(ValueError):
        TSPSolver(generate_random_cities(11)).brute_force()


@pytest.mark.parametrize("method", ["simulated_annealing", "genetic_algorithm"])
def test_seed_makes_randomized_algorithms_reproducible(method):
    cities = generate_random_cities(15)
    first = run(TSPSolver(cities, seed=123), method)
    second = run(TSPSolver(cities, seed=123), method)
    assert first == second


def test_compare_algorithms():
    solver = TSPSolver(generate_random_cities(8), seed=0)
    results = solver.compare_algorithms(["brute_force", "nearest_neighbor", "2-opt"])
    assert set(results) == {"brute_force", "nearest_neighbor", "2-opt"}
    for result in results.values():
        assert_valid(solver, result["tour"], result["distance"])
        assert result["time"] >= 0


def test_compare_algorithms_skips_brute_force_on_large_instances():
    solver = TSPSolver(generate_random_cities(12))
    assert "brute_force" not in solver.compare_algorithms(["brute_force"])


def test_compare_algorithms_rejects_unknown_names():
    with pytest.raises(ValueError):
        TSPSolver(generate_random_cities(5)).compare_algorithms(["no_such_algo"])


def test_visualize_tour_without_showing(tmp_path):
    solver = TSPSolver(generate_random_cities(6))
    tour, _ = solver.nearest_neighbor()
    path = tmp_path / "tour.png"
    fig = solver.visualize_tour(tour, save_path=str(path), show=False)
    assert fig is not None
    assert path.exists()


def test_run_benchmark():
    instances = {"grid_16": TSPBenchmark.generate_benchmark_instances()["grid_16"]}
    df = TSPBenchmark.run_benchmark(instances, ["nearest_neighbor", "2-opt"], seed=0)
    assert list(df["Algorithm"]) == ["nearest_neighbor", "2-opt"]
    distance = dict(zip(df["Algorithm"], df["Distance"]))
    # A 4x4 unit grid has an optimal tour of length 16.
    assert 16 - 1e-9 <= distance["2-opt"] <= distance["nearest_neighbor"]


def test_annealing_temperatures_scale_with_coordinates():
    cities = generate_random_cities(30, seed=5)
    small = TSPSolver(cities, seed=1)
    large = TSPSolver(cities * 1000, seed=1)
    tour = small.nearest_neighbor()[0]
    t_small = small._annealing_temperatures(lambda: small._random_two_opt_move(tour))
    t_large = large._annealing_temperatures(lambda: large._random_two_opt_move(tour))
    assert t_small[0] > t_small[1] > 0
    assert t_large[0] == pytest.approx(1000 * t_small[0])
    assert t_large[1] == pytest.approx(1000 * t_small[1])


def test_annealing_temperatures_follow_the_proposal_distribution():
    # Larger proposed moves must give hotter temperatures.
    moves = iter([(0, 2, d) for d in range(1, 501)] * 2)
    low = TSPSolver._annealing_temperatures(lambda: next(moves))
    big = iter([(0, 2, 10.0 * d) for d in range(1, 501)])
    high = TSPSolver._annealing_temperatures(lambda: next(big))
    assert high[0] == pytest.approx(10 * low[0])
    assert high[1] == pytest.approx(10 * low[1])


def test_simulated_annealing_without_polish_is_valid():
    solver = TSPSolver(generate_random_cities(25, seed=3), seed=3)
    tour, distance = solver.simulated_annealing(max_iterations=5000, polish=False)
    assert_valid(solver, tour, distance)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 9])
@pytest.mark.parametrize("seed", range(3))
def test_held_karp_matches_brute_force(n, seed):
    solver = TSPSolver(generate_random_cities(n, seed=seed))
    tour, distance = solver.held_karp()
    assert_valid(solver, tour, distance)
    assert distance == pytest.approx(solver.brute_force()[1])


@pytest.mark.parametrize("seed", range(3))
def test_heuristics_never_beat_held_karp(seed):
    solver = TSPSolver(generate_random_cities(14, seed=seed), seed=seed)
    _, optimal = solver.held_karp()
    for method in HEURISTICS:
        _, distance = run(solver, method)
        assert distance >= optimal - 1e-9


def test_iterated_local_search_finds_optimum_on_small_instance():
    solver = TSPSolver(generate_random_cities(14, seed=11), seed=0)
    _, optimal = solver.held_karp()
    _, distance = solver.iterated_local_search(iterations=300)
    assert distance == pytest.approx(optimal)


def test_held_karp_rejects_large_instances():
    with pytest.raises(ValueError):
        TSPSolver(generate_random_cities(21)).held_karp()
    with pytest.raises(ValueError):
        TSPSolver(generate_random_cities(13)).held_karp(max_cities=12)


def test_compare_algorithms_skips_held_karp_above_limit():
    solver = TSPSolver(generate_random_cities(21))
    assert "held_karp" not in solver.compare_algorithms(["held_karp"])


TSPLIB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tsplib")


@pytest.mark.parametrize("name, dimension", [("eil51", 51), ("berlin52", 52),
                                             ("st70", 70), ("kroA100", 100)])
def test_load_bundled_tsplib_instances(name, dimension):
    instance = load_tsplib(os.path.join(TSPLIB_DIR, f"{name}.tsp"))
    assert instance["name"] == name
    assert instance["dimension"] == dimension
    assert instance["coordinates"].shape == (dimension, 2)
    assert instance["edge_weight_type"] == "EUC_2D"
    assert instance["optimum"] == TSPLIB_OPTIMA[name]


def test_from_tsplib_uses_rounded_distances_and_reaches_optimum():
    solver = TSPSolver.from_tsplib(os.path.join(TSPLIB_DIR, "berlin52.tsp"), seed=42)
    D = solver.distance_matrix
    assert np.array_equal(D, np.round(D))
    tour, distance = solver.iterated_local_search()
    assert_valid(solver, tour, distance)
    assert distance == TSPLIB_OPTIMA["berlin52"]


def write_tsp(tmp_path, edge_type, coords, dimension=None, tsp_type="TSP"):
    lines = ["NAME : tiny", f"TYPE : {tsp_type}",
             f"DIMENSION : {dimension or len(coords)}",
             f"EDGE_WEIGHT_TYPE : {edge_type}", "NODE_COORD_SECTION"]
    lines += [f"{i + 1} {x} {y}" for i, (x, y) in enumerate(coords)]
    lines.append("EOF")
    path = tmp_path / "tiny.tsp"
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def test_tsplib_distance_conventions(tmp_path):
    coords = [(0, 0), (1.5, 0), (0, 2.6)]
    euc = TSPSolver.from_tsplib(write_tsp(tmp_path, "EUC_2D", coords))
    assert euc.distance_matrix[0, 1] == 2      # nint(1.5) = 2
    assert euc.distance_matrix[0, 2] == 3      # nint(2.6) = 3
    ceil = TSPSolver.from_tsplib(write_tsp(tmp_path, "CEIL_2D", coords))
    assert ceil.distance_matrix[0, 1] == 2 and ceil.distance_matrix[0, 2] == 3
    att = TSPSolver.from_tsplib(write_tsp(tmp_path, "ATT", [(0, 0), (10, 0)]))
    # r = sqrt(100 / 10) = 3.162..., nint(r) = 3 < r, so distance 4
    assert att.distance_matrix[0, 1] == 4
    assert TSPSolver(coords).distance_matrix[0, 1] == pytest.approx(1.5)


def test_tsplib_rejects_unsupported_files(tmp_path):
    with pytest.raises(ValueError):
        load_tsplib(write_tsp(tmp_path, "GEO", [(0, 0), (1, 1)]))
    with pytest.raises(ValueError):
        load_tsplib(write_tsp(tmp_path, "EUC_2D", [(0, 0), (1, 1)], tsp_type="ATSP"))
    with pytest.raises(ValueError):
        load_tsplib(write_tsp(tmp_path, "EUC_2D", [(0, 0), (1, 1)], dimension=3))
    with pytest.raises(ValueError):
        TSPSolver([(0, 0), (1, 1)], distance="manhattan")


def test_from_tsplib_accepts_loaded_instance():
    instance = load_tsplib(os.path.join(TSPLIB_DIR, "eil51.tsp"))
    from_dict = TSPSolver.from_tsplib(instance)
    from_path = TSPSolver.from_tsplib(os.path.join(TSPLIB_DIR, "eil51.tsp"))
    assert np.array_equal(from_dict.distance_matrix, from_path.distance_matrix)
    assert from_dict.city_names == instance["city_names"]


def test_run_tsplib_benchmark_reports_gap():
    df = TSPBenchmark.run_tsplib_benchmark(
        [os.path.join(TSPLIB_DIR, "eil51.tsp")], ["nearest_neighbor", "2-opt"], seed=0)
    assert list(df["Instance"]) == ["eil51", "eil51"]
    assert (df["Optimum"] == 426).all()
    assert (df["Gap (%)"] >= 0).all()
    nn, two = df["Distance"]
    assert two <= nn


def test_compare_algorithms_results_do_not_depend_on_order():
    cities = generate_random_cities(30, seed=6)
    alone = TSPSolver(cities, seed=7).compare_algorithms(["simulated_annealing"])
    after = TSPSolver(cities, seed=7).compare_algorithms(
        ["iterated_local_search", "genetic_algorithm", "simulated_annealing"])
    assert alone["simulated_annealing"]["tour"] == after["simulated_annealing"]["tour"]


@pytest.mark.parametrize("neighbors", [None, 3, 10])
def test_simulated_annealing_move_strategies(neighbors):
    solver = TSPSolver(generate_random_cities(40, seed=8), seed=8)
    _, start_distance = solver.nearest_neighbor()
    tour, distance = solver.simulated_annealing(max_iterations=20000,
                                                neighbors=neighbors)
    assert_valid(solver, tour, distance)
    assert distance <= start_distance + 1e-9



def improving_two_opt_move_exists(solver, tour):
    D = solver.distance_matrix
    n = len(tour)
    for i in range(n):
        for j in range(i + 2, n):
            if i == 0 and j == n - 1:
                continue
            a, b, c, d = tour[i], tour[i + 1], tour[j], tour[(j + 1) % n]
            if D[a, c] + D[b, d] < D[a, b] + D[c, d] - 1e-9:
                return True
    return False


@pytest.mark.parametrize("seed", range(12))
def test_two_opt_reaches_local_optimum_from_random_starts(seed):
    # Don't-look bits alone left an improving move in ~5% of such runs.
    solver = TSPSolver(generate_random_cities(80, seed=seed))
    start = list(range(80))
    random.Random(seed).shuffle(start)
    tour, distance = solver.two_opt(start)
    assert not improving_two_opt_move_exists(solver, tour)
    assert solver.two_opt(tour)[1] == pytest.approx(distance)


@pytest.mark.parametrize("kwargs", [
    {"max_iterations": 0},
    {"min_temp": 0},
    {"initial_temp": 1.0, "min_temp": 2.0},
    {"cooling_rate": 1.5},
])
def test_simulated_annealing_rejects_invalid_parameters(kwargs):
    solver = TSPSolver(generate_random_cities(20, seed=1), seed=1)
    with pytest.raises(ValueError):
        solver.simulated_annealing(**kwargs)


def test_simulated_annealing_rejects_min_temp_above_derived_start():
    # min_temp=1 was the old default; on a 0-1 scale instance it is hotter
    # than the derived start temperature and used to skip annealing silently.
    solver = TSPSolver(generate_random_cities(50, seed=1) / 100, seed=1)
    with pytest.raises(ValueError, match="min_temp"):
        solver.simulated_annealing(min_temp=1)


def test_from_tsplib_accepts_pathlib_path():
    path = pathlib.Path(TSPLIB_DIR) / "eil51.tsp"
    assert TSPSolver.from_tsplib(path).n_cities == 51


@pytest.mark.parametrize("method", ["two_opt", "or_opt", "local_search",
                                    "three_opt", "iterated_local_search"])
def test_local_search_on_partial_tour(method):
    solver = TSPSolver(generate_random_cities(30, seed=9), seed=0)
    subset = [3, 17, 8, 25, 1, 12, 29, 6, 20, 14]
    start_distance = solver.calculate_tour_distance(subset)
    tour, distance = getattr(solver, method)(subset)
    assert sorted(tour) == sorted(subset)
    assert distance <= start_distance + 1e-9


@pytest.mark.parametrize("tour", [[0, 1, 1, 2], [0, 1, 2, 99], [-1, 0, 1, 2]])
def test_invalid_initial_tours_are_rejected(tour):
    solver = TSPSolver(generate_random_cities(10, seed=1))
    with pytest.raises(ValueError):
        solver.two_opt(tour)


def test_double_bridge_reports_exactly_the_changed_endpoints():
    solver = TSPSolver(generate_random_cities(30, seed=2), seed=5)
    tour = list(range(30))

    def edges(t):
        return {frozenset((t[i], t[(i + 1) % len(t)])) for i in range(len(t))}

    for _ in range(50):
        new, touched = solver._double_bridge(tour)
        changed = edges(tour) ^ edges(new)
        assert set(touched) == {city for edge in changed for city in edge}


def test_exceeds_size_limit():
    small = TSPSolver(generate_random_cities(10))
    large = TSPSolver(generate_random_cities(21))
    assert not small.exceeds_size_limit("brute_force")
    assert large.exceeds_size_limit("held_karp")
    assert not large.exceeds_size_limit("2-opt")
