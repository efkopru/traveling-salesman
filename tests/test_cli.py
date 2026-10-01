import os
import sys

import pytest

from tsp_solver.cli import main, read_csv_cities

TSPLIB_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "tsplib")


def test_solve_tsplib_reports_gap(capsys):
    assert main(["solve", os.path.join(TSPLIB_DIR, "berlin52.tsp"),
                 "-a", "2-opt", "-a", "iterated_local_search", "--seed", "42"]) == 0
    out = capsys.readouterr().out
    assert "berlin52 (52 cities, EUC_2D), optimum 7542" in out
    assert "Best: iterated_local_search, distance 7542.00" in out


def test_solve_csv_with_header_names_and_outputs(tmp_path, capsys):
    csv_path = tmp_path / "cities.csv"
    csv_path.write_text("x,y,name\n0,0,Home\n3,0,Shop\n3,4,Park\n0,4,School\n")
    tour_path = tmp_path / "tour.txt"
    plot_path = tmp_path / "tour.png"
    assert main(["solve", str(csv_path), "-a", "held_karp",
                 "--tour-out", str(tour_path), "--plot", str(plot_path)]) == 0
    out = capsys.readouterr().out
    assert "distance 14.00" in out
    assert sorted(tour_path.read_text().split()) == ["Home", "Park", "School", "Shop"]
    assert plot_path.exists()


def test_solve_random_is_reproducible(capsys):
    def best_lines():
        main(["solve", "--random", "25", "--seed", "3"])
        lines = capsys.readouterr().out.splitlines()
        return [line for line in lines if line.startswith(("Best:", "Tour:"))]

    first = best_lines()
    assert len(first) == 2
    assert first == best_lines()


def test_exact_algorithm_above_limit_is_skipped(capsys):
    assert main(["solve", "--random", "25", "-a", "brute_force"]) == 1
    assert "Skipping brute_force" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["solve"], ["solve", "x.csv", "--random", "5"],
                                  ["solve", "--random", "5", "-a", "nope"],
                                  ["solve", "--random", "0"],
                                  ["solve", "--random", "-4"]])
def test_invalid_arguments_exit_with_error(argv):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2


def test_missing_file_is_a_usage_error(tmp_path):
    with pytest.raises(SystemExit):
        main(["solve", str(tmp_path / "missing.csv")])


def test_read_csv_rejects_bad_rows(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("1,2\nfoo,bar\n")
    with pytest.raises(ValueError):
        read_csv_cities(str(path))


def test_demo_runs(capsys):
    assert main(["demo"]) == 0
    out = capsys.readouterr().out
    assert "held_karp" in out and "Exact Solution" in out


@pytest.mark.parametrize("argv", [["solve", "--random", "8", "--plot", "x.png"],
                                  ["demo", "--plot", "x.png"], ["demo", "--show"]])
def test_plot_without_matplotlib_is_a_usage_error(argv, monkeypatch, capsys, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    assert "tsp-solver[plot]" in capsys.readouterr().err
    assert not (tmp_path / "x.png").exists()
