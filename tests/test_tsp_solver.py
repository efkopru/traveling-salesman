import numpy as np
import pytest

from tsp_solver import TSPSolver, TSPBenchmark, generate_random_cities

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
        assert city not in row and len(row) == 11
        assert all(D[city, a] <= D[city, b] for a, b in zip(row, row[1:]))
    assert all(len(row) == 3 for row in solver._neighbor_lists(3))


def test_iterated_local_search_is_reproducible():
    cities = generate_random_cities(30)
    first = TSPSolver(cities, seed=9).iterated_local_search(iterations=100)
    second = TSPSolver(cities, seed=9).iterated_local_search(iterations=100)
    assert first == second


def test_two_opt_max_iterations_limits_moves():
    solver = TSPSolver(generate_random_cities(50, seed=2))
    start, start_distance = solver.nearest_neighbor()
    _, one_move = solver.two_opt(start, max_iterations=1)
    _, full = solver.two_opt(start)
    assert full <= one_move < start_distance


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
    t_small = small._annealing_temperatures(small.nearest_neighbor()[0])
    t_large = large._annealing_temperatures(large.nearest_neighbor()[0])
    assert t_small[0] > t_small[1] > 0
    assert t_large[0] == pytest.approx(1000 * t_small[0])
    assert t_large[1] == pytest.approx(1000 * t_small[1])


def test_simulated_annealing_without_polish_is_valid():
    solver = TSPSolver(generate_random_cities(25, seed=3), seed=3)
    tour, distance = solver.simulated_annealing(max_iterations=5000, polish=False)
    assert_valid(solver, tour, distance)
