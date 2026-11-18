import itertools
import os
import pathlib
import random
import threading
import time

import numpy as np
import pytest

import tsp_solver.solver as solver_module
from tsp_solver import (TSPLIB_OPTIMA, TSPBenchmark, TSPSolver,
                        generate_random_cities, load_tsplib)
from tsp_solver.solver import _TourSearch

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
                                    "three_opt", "iterated_local_search"])
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
BUNDLED = ["eil51", "berlin52", "st70", "kroA100"]


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


def test_exceeds_size_limit():
    small = TSPSolver(generate_random_cities(10))
    large = TSPSolver(generate_random_cities(21))
    assert not small.exceeds_size_limit("brute_force")
    assert large.exceeds_size_limit("held_karp")
    assert not large.exceeds_size_limit("2-opt")


# ---- tour representation and moves (_TourSearch) ----

def edge_set(tour):
    return {frozenset((tour[i], tour[(i + 1) % len(tour)])) for i in range(len(tour))}


@pytest.mark.parametrize("n", [8, 9, 13, 40, 200])
def test_double_bridge_changes_four_edges_and_reports_them(n):
    solver = TSPSolver(generate_random_cities(n, seed=n))
    rng = random.Random(n)
    max_piece = min(50, (n - 2) // 3)
    four_edge_kicks = 0
    for _ in range(60):
        tour = rng.sample(range(n), n)
        before = list(tour)
        search = _TourSearch(solver, tour)
        change, touched = search.double_bridge(rng, max_piece)
        assert sorted(tour) == list(range(n))
        assert all(search.pos[city] == i for i, city in enumerate(tour))
        removed = edge_set(before) - edge_set(tour)
        # Fewer than four only when pieces of one city put an old edge back.
        assert len(removed) <= 4
        four_edge_kicks += len(removed) == 4
        changed = edge_set(before) ^ edge_set(tour)
        assert {city for edge in changed for city in edge} <= set(touched)
        assert change == pytest.approx(solver.calculate_tour_distance(tour)
                                       - solver.calculate_tour_distance(before), abs=1e-9)
    assert four_edge_kicks > 20


@pytest.mark.parametrize("n", [5, 6, 7, 12, 60])
@pytest.mark.parametrize("prune", [False, True])
def test_tour_search_reports_the_exact_change(n, prune):
    solver = TSPSolver(generate_random_cities(n, seed=n))
    rng = random.Random(n)
    for _ in range(10):
        tour = rng.sample(range(n), n)
        before = solver.calculate_tour_distance(tour)
        search = _TourSearch(solver, tour)
        change = search.optimize(solver._neighbor_lists(), solver._neighbor_lists(10),
                                 until_optimal=True, prune=prune)
        assert sorted(tour) == list(range(n))
        assert all(search.pos[city] == i for i, city in enumerate(tour))
        assert not any(search.queued)
        assert change <= 0
        assert solver.calculate_tour_distance(tour) - before == pytest.approx(change, abs=1e-9)


def improving_or_opt_move_exists(solver, tour, neighbors=10, max_segment=3):
    """Every segment of up to max_segment cities, moved in either
    orientation into a tour edge at one of the nearest neighbors of its
    end cities."""
    D = solver.distance_matrix
    near = solver._neighbor_lists(neighbors)
    n = len(tour)
    pos = {city: i for i, city in enumerate(tour)}
    for length in range(1, max_segment + 1):
        if n < length + 3:
            break
        for i in range(n):
            s, e = tour[i], tour[(i + length - 1) % n]
            p, nx = tour[(i - 1) % n], tour[(i + length) % n]
            removal_gain = D[p, s] + D[e, nx] - D[p, nx]
            for x in (s, e):
                for c in near[x]:
                    if (pos[c] - i) % n < length:
                        continue
                    for u, v in ((c, tour[(pos[c] + 1) % n]), (tour[(pos[c] - 1) % n], c)):
                        if (pos[u] - i) % n < length or (pos[v] - i) % n < length:
                            continue
                        cost = min(D[u, s] + D[e, v], D[u, e] + D[s, v]) - D[u, v]
                        if removal_gain - cost > 1e-9:
                            return True
    return False


@pytest.mark.parametrize("seed", range(8))
def test_local_search_leaves_no_two_opt_or_or_opt_move(seed):
    solver = TSPSolver(generate_random_cities(70, seed=seed))
    start = list(range(70))
    random.Random(seed).shuffle(start)
    tour, distance = solver.local_search(start)
    assert not improving_two_opt_move_exists(solver, tour)
    assert not improving_or_opt_move_exists(solver, tour)
    assert distance <= solver.two_opt(start)[1] + 1e-9
    relocated, _ = solver.or_opt(start)
    assert not improving_or_opt_move_exists(solver, relocated)


def best_three_opt_gain(solver, tour):
    """Largest gain of any of the seven reconnections of any three edges."""
    D = solver.distance_matrix
    n = len(tour)
    best = 0.0
    for i, j, k in itertools.combinations(range(n), 3):
        a, b = tour[i], tour[i + 1]
        c, d = tour[j], tour[j + 1]
        e, f = tour[k], tour[(k + 1) % n]
        current = D[a, b] + D[c, d] + D[e, f]
        best = max(best, current - min(
            D[a, c] + D[b, d] + D[e, f], D[a, b] + D[c, e] + D[d, f],
            D[a, e] + D[c, d] + D[b, f], D[a, d] + D[e, b] + D[c, f],
            D[a, c] + D[b, e] + D[d, f], D[a, e] + D[d, b] + D[c, f],
            D[a, d] + D[e, c] + D[b, f]))
    return best


@pytest.mark.parametrize("n, seed", [(4, 0), (5, 1), (6, 2), (9, 3), (30, 4), (30, 5), (45, 6)])
def test_three_opt_is_locally_optimal(n, seed):
    # Only four of the seven reconnections used to be tried, and never for
    # neighboring edges: half the results could still be improved.
    solver = TSPSolver(generate_random_cities(n, seed=seed))
    start = list(range(n))
    random.Random(seed).shuffle(start)
    tour, distance = solver.three_opt(start)
    assert_valid(solver, tour, distance)
    assert tour[0] == start[0]
    assert best_three_opt_gain(solver, tour) < 1e-9
    assert not improving_two_opt_move_exists(solver, tour)


# ---- numerical robustness ----

def run_with_timeout(function, seconds):
    """Return function(), or fail if it is still running after `seconds`."""
    outcome = {}

    def target():
        try:
            outcome["value"] = function()
        except BaseException as error:  # reported in the calling thread
            outcome["error"] = error

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    assert not thread.is_alive(), f"still running after {seconds} s"
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


@pytest.mark.parametrize("method, side, starts", [
    ("two_opt", 10, 60), ("or_opt", 10, 80), ("local_search", 10, 30),
    ("three_opt", 7, 4), ("iterated_local_search", 10, 3)])
def test_local_search_terminates_on_ties_with_large_coordinates(method, side, starts):
    # A grid scaled like map coordinates in metres: equally long edges
    # differ in their last bits. With a fixed tolerance of 1e-10 the search
    # traded such tours back and forth forever.
    n = side * side
    grid = np.array([(i, j) for i in range(side) for j in range(side)], dtype=float)
    solver = TSPSolver(grid * 1234567.891, seed=1)
    unit = solver.distance_matrix[0, 1]
    rng = random.Random(0)
    kwargs = {"iterations": 100} if method == "iterated_local_search" else {}

    def run_all():
        return [getattr(solver, method)(rng.sample(range(n), n), **kwargs)
                for _ in range(starts)]

    for tour, distance in run_with_timeout(run_all, 120):
        assert sorted(tour) == list(range(n))
        assert distance / unit >= n - 1e-6        # a grid tour is at least n long


@pytest.mark.parametrize("scale", [1e-9, 1e9])
def test_results_do_not_depend_on_the_scale_of_the_coordinates(scale):
    # With the old fixed tolerance, coordinates of the order of 1e-10 were
    # not improved at all.
    cities = generate_random_cities(40, seed=5)
    for method in ("two_opt", "or_opt", "local_search", "three_opt"):
        assert getattr(TSPSolver(cities * scale), method)()[0] == \
            getattr(TSPSolver(cities), method)()[0]
    for method, kwargs in (("iterated_local_search", {"iterations": 200}),
                           ("simulated_annealing", {"max_iterations": 20000}),
                           ("genetic_algorithm", {"generations": 5})):
        assert getattr(TSPSolver(cities * scale, seed=1), method)(**kwargs)[0] == \
            getattr(TSPSolver(cities, seed=1), method)(**kwargs)[0]


def test_tolerance_scales_with_the_distances():
    cities = generate_random_cities(10, seed=1)
    small, large = TSPSolver(cities), TSPSolver(cities * 1e6)
    assert 0 < small.tolerance < 1e-9
    assert large.tolerance == pytest.approx(1e6 * small.tolerance)
    assert TSPSolver(np.ones((5, 2))).tolerance == 0


@pytest.mark.parametrize("coordinate", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_coordinates_are_rejected(coordinate):
    with pytest.raises(ValueError, match="finite"):
        TSPSolver([(0, 0), (1, coordinate), (2, 2)])


def test_overflowing_distances_are_rejected():
    with pytest.raises(ValueError, match="too large"):
        TSPSolver([(0, 0), (1e200, 1e200)])


@pytest.mark.parametrize("start_city", [-1, 6, 1.5, "0", None, True])
def test_start_city_is_validated(start_city):
    # -1 used to give brute force a tour with seven entries for six cities.
    solver = TSPSolver(generate_random_cities(6))
    for method in (solver.brute_force, solver.nearest_neighbor):
        with pytest.raises(ValueError, match="start_city"):
            method(start_city)


def test_start_city_accepts_numpy_integers():
    solver = TSPSolver(generate_random_cities(6))
    assert solver.nearest_neighbor(np.int64(4)) == solver.nearest_neighbor(4)
    assert solver.brute_force(np.int32(2))[0][0] == 2


def test_city_names_may_be_any_sequence():
    names = np.array(["a", "b", "c"])
    assert TSPSolver(generate_random_cities(3), city_names=names).city_names == ["a", "b", "c"]
    assert TSPSolver(generate_random_cities(2), city_names=(7, 8)).city_names == ["7", "8"]


# ---- constructive heuristics ----

@pytest.mark.parametrize("cities", [
    generate_random_cities(200, seed=3),
    np.vstack([generate_random_cities(60, seed=4)] * 3),             # duplicates
    np.array([(i, j) for i in range(14) for j in range(14)]),        # ties
    generate_random_cities(3, seed=1),
])
def test_vectorized_construction_matches_the_python_loops(cities, monkeypatch):
    solver = TSPSolver(cities)
    last = len(cities) - 1
    monkeypatch.setattr(solver_module, "_NUMPY_ROW_THRESHOLD", 10 ** 9)
    plain = [solver.nearest_neighbor(0), solver.nearest_neighbor(last),
             solver.nearest_insertion()]
    monkeypatch.setattr(solver_module, "_NUMPY_ROW_THRESHOLD", 0)
    assert [solver.nearest_neighbor(0), solver.nearest_neighbor(last),
            solver.nearest_insertion()] == plain


def test_nearest_insertion_starts_with_the_closest_pair():
    solver = TSPSolver([(0, 0), (10, 0), (10, 1), (0, 9), (5, 5)])
    tour, distance = solver.nearest_insertion()
    assert_valid(solver, tour, distance)
    assert tour[0] == 1                   # cities 1 and 2 are the closest pair
    # Of two equally close pairs, the one with the lower indices starts.
    tied = TSPSolver([(0, 0), (5, 5), (0, 1), (9, 9), (9, 8)])
    assert tied.nearest_insertion()[0][0] == 0


# ---- iterated local search ----

@pytest.mark.parametrize("name", BUNDLED)
def test_iterated_local_search_reaches_the_optimum_on_bundled_instances(name):
    solver = TSPSolver.from_tsplib(os.path.join(TSPLIB_DIR, f"{name}.tsp"), seed=42)
    tour, distance = solver.iterated_local_search()
    assert_valid(solver, tour, distance)
    assert distance == TSPLIB_OPTIMA[name]


def test_iterated_local_search_budgets():
    solver = TSPSolver(generate_random_cities(60, seed=2), seed=2)
    _, local = solver.local_search()
    # No kicks: the local search alone.
    assert solver.iterated_local_search(iterations=0)[1] == pytest.approx(local)
    assert solver.iterated_local_search(time_limit=0)[1] == pytest.approx(local)
    started = time.perf_counter()
    tour, distance = solver.iterated_local_search(time_limit=0.2)
    assert time.perf_counter() - started < 10
    assert_valid(solver, tour, distance)
    assert distance <= local + 1e-9
    # Whichever budget runs out first ends the search.
    tour, distance = solver.iterated_local_search(iterations=20, time_limit=60)
    assert_valid(solver, tour, distance)


@pytest.mark.parametrize("kwargs", [{"iterations": -1}, {"time_limit": -0.5}])
def test_iterated_local_search_rejects_negative_budgets(kwargs):
    solver = TSPSolver(generate_random_cities(20, seed=1), seed=1)
    with pytest.raises(ValueError):
        solver.iterated_local_search(**kwargs)


@pytest.mark.parametrize("neighbors", [None, 3])
def test_iterated_local_search_neighbor_options(neighbors):
    solver = TSPSolver(generate_random_cities(50, seed=6), seed=6)
    start, start_distance = solver.nearest_neighbor(9)
    tour, distance = solver.iterated_local_search(start, iterations=100, neighbors=neighbors)
    assert_valid(solver, tour, distance)
    assert tour[0] == 9
    assert distance <= solver.local_search(start)[1] + 1e-9


# ---- genetic algorithm ----

@pytest.mark.parametrize("kwargs", [{"population_size": 0}, {"generations": -1},
                                    {"mutation_rate": 1.5}, {"mutation_rate": -0.1},
                                    {"elite_size": -1}])
def test_genetic_algorithm_rejects_invalid_parameters(kwargs):
    solver = TSPSolver(generate_random_cities(12, seed=1), seed=1)
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        solver.genetic_algorithm(**kwargs)


@pytest.mark.parametrize("kwargs", [{"population_size": 1}, {"population_size": 2, "elite_size": 0},
                                    {"generations": 0}, {"population_size": 5, "elite_size": 9},
                                    {"population_size": 40}, {"mutation_rate": 1.0},
                                    {"mutation_rate": 0.0}, {"neighbors": None}])
def test_genetic_algorithm_parameter_ranges(kwargs):
    solver = TSPSolver(generate_random_cities(12, seed=1), seed=1)
    kwargs = {"generations": 5, **kwargs}
    tour, distance = solver.genetic_algorithm(**kwargs)
    assert_valid(solver, tour, distance)
    assert distance >= solver.held_karp()[1] - 1e-9


def test_genetic_algorithm_reaches_the_optimum_on_eil51():
    solver = TSPSolver.from_tsplib(os.path.join(TSPLIB_DIR, "eil51.tsp"), seed=42)
    tour, distance = solver.genetic_algorithm()
    assert_valid(solver, tour, distance)
    assert distance == TSPLIB_OPTIMA["eil51"]


def test_genetic_algorithm_result_is_locally_optimal():
    solver = TSPSolver(generate_random_cities(60, seed=3), seed=3)
    tour, _ = solver.genetic_algorithm(generations=3)
    assert not improving_two_opt_move_exists(solver, tour)
    assert not improving_or_opt_move_exists(solver, tour)


# ---- comparison and plotting ----

def test_compare_algorithms_passes_options():
    solver = TSPSolver(generate_random_cities(40, seed=3), seed=3)
    results = solver.compare_algorithms(
        ["2-opt", "iterated_local_search", "nearest_neighbor"],
        options={"2-opt": {"max_iterations": 0},
                 "iterated_local_search": {"iterations": 0}})
    assert results["2-opt"]["tour"] == results["nearest_neighbor"]["tour"]
    assert results["iterated_local_search"]["distance"] == pytest.approx(solver.local_search()[1])
    with pytest.raises(ValueError, match="no_such"):
        solver.compare_algorithms(["2-opt"], options={"no_such": {}})


def test_visualize_tour_labels_small_instances_only():
    import matplotlib.pyplot as plt

    small = TSPSolver(generate_random_cities(12))
    large = TSPSolver(generate_random_cities(80))
    for solver, labels, expected in ((small, None, 12), (large, None, 0),
                                     (large, True, 80), (small, False, 0)):
        fig = solver.visualize_tour(solver.nearest_neighbor()[0], show=False, labels=labels)
        assert len(fig.axes[0].texts) == expected
        plt.close(fig)
    with pytest.raises(ValueError, match="empty"):
        small.visualize_tour([], show=False)


# ---- TSPLIB reader ----

def test_tsplib_reader_tolerates_format_variations(tmp_path):
    path = tmp_path / "variations.tsp"
    path.write_bytes(b"name : tiny\ncomment : caf\xe9 tour\nTYPE :\ndimension : 3\n"
                     b"edge_weight_type : euc_2d\nnode_coord_section\n"
                     b"1.0 0 0\n2.0 3e0 0\n3.0 3 4\neof\nthis line is not read\n")
    instance = load_tsplib(path)
    assert instance["dimension"] == 3
    assert instance["city_names"] == ["1.0", "2.0", "3.0"]
    assert instance["edge_weight_type"] == "EUC_2D"
    assert instance["coordinates"].tolist() == [[0, 0], [3, 0], [3, 4]]
    assert TSPSolver.from_tsplib(instance).held_karp()[1] == 12


def test_tsplib_reader_finds_the_optimum_when_the_name_has_an_extension(tmp_path):
    text = open(os.path.join(TSPLIB_DIR, "eil51.tsp")).read()
    assert "eil51" in text.splitlines()[0]
    path = tmp_path / "renamed.tsp"
    path.write_text(text.replace("eil51", "eil51.tsp", 1))
    assert load_tsplib(path)["optimum"] == 426


def test_optima_are_known_for_instances_that_are_not_bundled(tmp_path):
    assert set(BUNDLED) <= set(TSPLIB_OPTIMA)
    assert all(isinstance(value, int) and value > 0 for value in TSPLIB_OPTIMA.values())
    path = tmp_path / "pr1002.tsp"
    path.write_text("NAME : pr1002\nTYPE : TSP\nDIMENSION : 3\nEDGE_WEIGHT_TYPE : EUC_2D\n"
                    "NODE_COORD_SECTION\n1 0 0\n2 3 0\n3 3 4\nEOF\n")
    assert load_tsplib(path)["optimum"] == 259045
    path.write_text(path.read_text().replace("pr1002", "my_cities"))
    assert load_tsplib(path)["optimum"] is None


def test_tsplib_reader_works_without_dimension(tmp_path):
    path = tmp_path / "nodim.tsp"
    path.write_text("NAME: nodim\nTYPE: TSP\nEDGE_WEIGHT_TYPE: EUC_2D\n"
                    "NODE_COORD_SECTION\n1 0 0\n2 5 0\n")
    assert load_tsplib(path)["dimension"] == 2


@pytest.mark.parametrize("body, message", [
    ("DIMENSION: 2\nNODE_COORD_SECTION\n1 0 0\n2 1 one\nEOF\n", "line 7"),
    ("DIMENSION: 2\nNODE_COORD_SECTION\n1 0 0\n2 1\nEOF\n", "line 7"),
    ("DIMENSION: 2\nNODE_COORD_SECTION\nEOF\n", "no node coordinates"),
    ("DIMENSION: two\nNODE_COORD_SECTION\n1 0 0\n2 1 1\nEOF\n", "DIMENSION must be an integer"),
    ("DIMENSION: 3\nNODE_COORD_SECTION\n1 0 0\n2 1 1\nEOF\n", "DIMENSION is 3"),
])
def test_tsplib_reader_reports_malformed_files(tmp_path, body, message):
    path = tmp_path / "bad.tsp"
    path.write_text("NAME: bad\nTYPE: TSP\nEDGE_WEIGHT_TYPE: EUC_2D\n" + body)
    with pytest.raises(ValueError, match=message):
        load_tsplib(path)
