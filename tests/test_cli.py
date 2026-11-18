import io
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


def test_csv_with_byte_order_mark_keeps_first_city(tmp_path):
    path = tmp_path / "excel.csv"
    path.write_bytes("1,2\n3,4\n5,6\n".encode("utf-8-sig"))
    cities, names = read_csv_cities(str(path))
    assert cities.tolist() == [[1, 2], [3, 4], [5, 6]]


def test_random_instance_without_seed_is_reproducible(capsys):
    def run():
        main(["solve", "--random", "60"])
        out = capsys.readouterr().out
        return [line for line in out.splitlines() if line.startswith(("Best:", "Tour:"))]

    first = run()
    assert first == run()


@pytest.mark.parametrize("option", ["--tour-out", "--plot"])
def test_unwritable_output_fails_before_solving(option, tmp_path, capsys):
    target = str(tmp_path / "missing_dir" / "out.txt")
    with pytest.raises(SystemExit) as excinfo:
        main(["solve", "--random", "8", option, target])
    assert excinfo.value.code == 2
    assert "Instance:" not in capsys.readouterr().out


def test_output_write_error_is_reported(tmp_path, capsys):
    # A directory passes the up-front check but cannot be opened as a file.
    assert main(["solve", "--random", "8", "--tour-out", str(tmp_path)]) == 1
    assert "could not write output" in capsys.readouterr().err


UNICODE_CSV = "x,y,name\n0,0,東京\n3,0,Łódź\n3,4,Zürich\n0,4,Oslo\n"


def test_tour_file_is_utf8_whatever_the_locale(tmp_path, capsys):
    # The platform default (cp1252 on Windows) cannot encode these names.
    csv_path = tmp_path / "cities.csv"
    csv_path.write_text(UNICODE_CSV, encoding="utf-8")
    tour_path = tmp_path / "tour.txt"
    assert main(["solve", str(csv_path), "-a", "2-opt", "--tour-out", str(tour_path)]) == 0
    names = tour_path.read_text(encoding="utf-8").split()
    assert sorted(names) == sorted(["東京", "Łódź", "Zürich", "Oslo"])


def test_names_the_terminal_cannot_encode_do_not_stop_the_output(tmp_path, monkeypatch):
    # A redirected stdout on Windows: printing the tour raised
    # UnicodeEncodeError after the instance had been solved.
    csv_path = tmp_path / "cities.csv"
    csv_path.write_text(UNICODE_CSV, encoding="utf-8")
    buffer = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", io.TextIOWrapper(buffer, encoding="cp1252"))
    assert main(["solve", str(csv_path), "-a", "2-opt"]) == 0
    sys.stdout.flush()
    printed = buffer.getvalue().decode("cp1252")
    assert "Zürich" in printed and "Oslo" in printed
    assert "\\u6771\\u4eac" in printed          # 東京, escaped


@pytest.mark.parametrize("text", [
    "x;y;name\n0;0;A\n1,5;0;B\n1,5;2,5;C\n0;2,5;D\n",      # semicolons, decimal commas
    "x\ty\tname\n0\t0\tA\n1.5\t0\tB\n1,5\t2,5\tC\n0\t2.5\tD\n",
    "0 0 A\n1.5 0 B\n1.5   2.5 C\n0 2.5 D\n",
    "\n\nx,y,name\n0,0,A\n1.5,0,B\n1.5,2.5,C\n0,2.5,D\n",    # blank lines, then a header
    '0,0,A\r\n1.5,0,"B"\r\n\r\n1.5,2.5,C\r\n0,2.5,D\r\n',
])
def test_csv_dialects(tmp_path, text):
    path = tmp_path / "cities.csv"
    path.write_bytes(text.encode("utf-8"))
    cities, names = read_csv_cities(str(path))
    assert cities.tolist() == [[0, 0], [1.5, 0], [1.5, 2.5], [0, 2.5]]
    assert names == ["A", "B", "C", "D"]


def test_csv_keeps_commas_inside_quoted_names(tmp_path):
    path = tmp_path / "cities.csv"
    path.write_text('1,2,"Washington, D.C."\n3,4,"a;b"\n')
    cities, names = read_csv_cities(str(path))
    assert cities.tolist() == [[1, 2], [3, 4]]
    assert names == ["Washington, D.C.", "a;b"]


def test_csv_that_is_not_utf8_is_read_as_windows_1252(tmp_path):
    path = tmp_path / "excel.csv"
    path.write_bytes("0,0,Zürich\n3,0,Köln\n3,4,Malmö\n".encode("cp1252"))
    cities, names = read_csv_cities(str(path))
    assert names == ["Zürich", "Köln", "Malmö"]


@pytest.mark.parametrize("row", ["nan,1", "1,inf"])
def test_csv_rejects_coordinates_that_are_not_finite(tmp_path, row):
    path = tmp_path / "bad.csv"
    path.write_text(f"0,0\n{row}\n2,2\n")
    with pytest.raises(ValueError, match=r"bad\.csv:2: coordinates must be finite"):
        read_csv_cities(str(path))


def test_csv_error_names_the_line(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("x,y\n\n1,2\n3,four\n")
    with pytest.raises(ValueError, match=r"bad\.csv:4: expected"):
        read_csv_cities(str(path))


def test_time_limit_runs_iterated_local_search(capsys):
    assert main(["solve", os.path.join(TSPLIB_DIR, "eil51.tsp"), "--seed", "1",
                 "--time-limit", "0.3"]) == 0
    out = capsys.readouterr().out
    assert "iterated_local_search" in out and "optimum 426" in out
    seconds = float(out.splitlines()[2].split()[-1])
    assert 0.2 < seconds < 10


def test_time_limit_applies_only_to_algorithms_that_take_one(capsys):
    assert main(["solve", "--random", "30", "-a", "2-opt", "-a", "iterated_local_search",
                 "--time-limit", "0.1"]) == 0
    assert "2-opt" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [
    ["solve", "--random", "20", "-a", "2-opt", "--time-limit", "1"],
    ["solve", "--random", "20", "--time-limit", "0"],
    ["solve", "--random", "20", "--time-limit", "-2"],
    ["solve", "--random", "20", "--time-limit", "soon"],
    ["solve", "--random", "20", "--time-limit", "nan"],
    ["solve", "--random", "20", "--time-limit", "inf"],
])
def test_invalid_time_limit_is_a_usage_error(argv, capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(argv)
    assert excinfo.value.code == 2
    assert "Instance:" not in capsys.readouterr().out


def test_version(capsys):
    import tsp_solver

    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"tsp-solver {tsp_solver.__version__}"
    # The version is written in two places; they must agree.
    pyproject = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")
    with open(pyproject, encoding="utf-8") as f:
        assert f'version = "{tsp_solver.__version__}"' in f.read()


def test_large_instance_plot_has_no_labels(tmp_path):
    plot_path = tmp_path / "tour.png"
    assert main(["solve", "--random", "120", "-a", "2-opt", "--plot", str(plot_path)]) == 0
    assert plot_path.stat().st_size > 0
