"""
Command-line interface.

    tsp-solver solve cities.csv -a 2-opt -a iterated_local_search --plot tour.png
    tsp-solver solve data/tsplib/berlin52.tsp
    tsp-solver solve --random 50 --seed 1
    tsp-solver solve --random 2000 --time-limit 10
    tsp-solver demo

Also available as `python -m tsp_solver`.
"""

import argparse
import csv
import io
import math
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

from .solver import TSPSolver, generate_random_cities
from .tsplib import load_tsplib

DEFAULT_ALGORITHM = 'iterated_local_search'
# Algorithms that can run for a given time instead of a fixed budget.
TIME_LIMITED = ('iterated_local_search',)
PLOT_HINT = "plotting needs matplotlib: pip install 'tsp-solver[plot]'"
# Tried in this order when working out how a CSV file separates its fields.
CSV_DELIMITERS = (',', ';', '\t', ' ')


def positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {value}")
    return value


def positive_seconds(text: str) -> float:
    value = float(text)
    if not value > 0 or math.isinf(value):
        raise argparse.ArgumentTypeError(f"must be a positive number of seconds, got {text}")
    return value


def require_writable(parser, path: Optional[str], option: str) -> None:
    """Fail before solving if an output file cannot be created."""
    if path is None:
        return
    directory = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(directory):
        parser.error(f"{option}: directory does not exist: {directory}")
    if not os.access(directory, os.W_OK):
        parser.error(f"{option}: directory is not writable: {directory}")


def require_matplotlib(parser) -> None:
    """Fail before solving if a plot is requested without matplotlib."""
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        parser.error(PLOT_HINT)


def _parse_coordinate(cell: str, delimiter: str) -> float:
    """A coordinate from a CSV cell. Files that separate fields with ';' or
    a tab often come from locales that write decimals with a comma."""
    if delimiter != ',':
        cell = cell.replace(',', '.')
    return float(cell)


def _read_rows(text: str, delimiter: str) -> List[Tuple[int, List[str]]]:
    """Non-empty CSV rows as (line number, cells)."""
    reader = csv.reader(io.StringIO(text, newline=''), delimiter=delimiter,
                        skipinitialspace=delimiter == ' ')
    rows = []
    for row in reader:
        row = [cell.strip() for cell in row]
        if any(row):
            rows.append((reader.line_num, row))
    return rows


def _detect_delimiter(text: str) -> str:
    """The first delimiter under which one of the leading rows starts with
    two numbers (the very first row may be a header)."""
    for delimiter in CSV_DELIMITERS:
        for _, row in _read_rows(text, delimiter)[:3]:
            try:
                _parse_coordinate(row[0], delimiter)
                _parse_coordinate(row[1], delimiter)
            except (ValueError, IndexError):
                continue
            return delimiter
    return ','


def read_csv_cities(path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Read cities from a CSV file with rows `x,y` or `x,y,name`.

    Fields may also be separated by semicolons, tabs or spaces (with ';'
    and tabs, decimal commas are accepted). A first row that is not
    numeric is treated as a header. The file is read as UTF-8, or as
    Windows-1252 if it is not valid UTF-8.
    """
    with open(path, 'rb') as f:
        raw = f.read()
    try:
        # utf-8-sig drops the byte-order mark Excel writes in "CSV UTF-8"
        # files; left in place it would make the first row look like a header.
        text = raw.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = raw.decode('cp1252', errors='replace')   # Excel's plain "CSV"

    delimiter = _detect_delimiter(text)
    coords, names = [], []
    for index, (line_number, row) in enumerate(_read_rows(text, delimiter)):
        try:
            x = _parse_coordinate(row[0], delimiter)
            y = _parse_coordinate(row[1], delimiter)
        except (ValueError, IndexError):
            if index == 0:
                continue  # header
            raise ValueError(f"{path}:{line_number}: expected 'x,y[,name]', "
                             f"got {delimiter.join(row)!r}")
        if not (math.isfinite(x) and math.isfinite(y)):
            raise ValueError(f"{path}:{line_number}: coordinates must be finite, "
                             f"got {delimiter.join(row)!r}")
        coords.append((x, y))
        names.append(row[2] if len(row) > 2 and row[2] else str(len(names)))
    if not coords:
        raise ValueError(f"{path}: no cities found")
    return np.array(coords), names


def build_solver(args) -> Tuple[TSPSolver, str, Optional[float]]:
    """Return (solver, description, known optimum or None)."""
    if args.random is not None:
        seed = 42 if args.seed is None else args.seed
        cities = generate_random_cities(args.random, seed=seed)
        # The printed seed also drives the randomized algorithms, so the
        # run is reproducible from its own output.
        solver = TSPSolver(cities, [f"C{i}" for i in range(args.random)],
                           seed=seed)
        return solver, f"{args.random} random cities (seed {seed})", None

    if args.input.lower().endswith('.tsp'):
        instance = load_tsplib(args.input)
        solver = TSPSolver.from_tsplib(instance, seed=args.seed)
        description = (f"{instance['name']} ({instance['dimension']} cities, "
                       f"{instance['edge_weight_type']})")
        return solver, description, instance['optimum']

    cities, names = read_csv_cities(args.input)
    solver = TSPSolver(cities, names, seed=args.seed)
    return solver, f"{args.input} ({len(names)} cities)", None


def cmd_solve(args, parser) -> int:
    if (args.input is None) == (args.random is None):
        parser.error("give either an input file or --random N")
    algorithms = args.algo or [DEFAULT_ALGORITHM]
    options = {}
    if args.time_limit is not None:
        options = {a: {'time_limit': args.time_limit}
                   for a in algorithms if a in TIME_LIMITED}
        if not options:
            parser.error(f"--time-limit applies to {', '.join(TIME_LIMITED)}; "
                         f"none of the chosen algorithms can use it")
    if args.plot:
        require_matplotlib(parser)
    require_writable(parser, args.plot, '--plot')
    require_writable(parser, args.tour_out, '--tour-out')
    try:
        solver, description, optimum = build_solver(args)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    skipped = [a for a in algorithms if solver.exceeds_size_limit(a)]
    for algo in skipped:
        print(f"Skipping {algo}: limited to {solver.EXACT_LIMITS[algo]} cities",
              file=sys.stderr)
    algorithms = [a for a in algorithms if a not in skipped]
    if not algorithms:
        return 1

    results = solver.compare_algorithms(algorithms, options)

    print(f"Instance: {description}"
          + (f", optimum {optimum}" if optimum is not None else ""))
    header = f"{'Algorithm':<24}{'Distance':>14}"
    if optimum is not None:
        header += f"{'Gap':>10}"
    print(header + f"{'Time (s)':>12}")
    for algo, result in results.items():
        line = f"{algo:<24}{result['distance']:>14.2f}"
        if optimum is not None:
            line += f"{100 * (result['distance'] - optimum) / optimum:>9.2f}%"
        print(line + f"{result['time']:>12.4f}")

    best_algo = min(results, key=lambda a: results[a]['distance'])
    best = results[best_algo]
    names = [solver.city_names[i] for i in best['tour']]
    print(f"\nBest: {best_algo}, distance {best['distance']:.2f}")
    print("Tour: " + " -> ".join(names + names[:1]))

    try:
        if args.tour_out:
            # UTF-8 whatever the platform's default: city names need not
            # fit the local code page.
            with open(args.tour_out, 'w', encoding='utf-8') as f:
                f.write("\n".join(names) + "\n")
        if args.plot:
            solver.visualize_tour(best['tour'], f"{description}: {best_algo}",
                                  save_path=args.plot, show=False)
    except OSError as error:
        print(f"tsp-solver: could not write output: {error}", file=sys.stderr)
        return 1
    return 0


def cmd_demo(args, parser) -> int:
    if args.plot or args.show:
        require_matplotlib(parser)
    require_writable(parser, args.plot, '--plot')
    from .demo import run_demo
    run_demo(show=args.show, save_path=args.plot)
    return 0


def build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog='tsp-solver',
        description='Solve Traveling Salesman Problem instances.')
    parser.add_argument('--version', action='version',
                        version=f'%(prog)s {__version__}')
    commands = parser.add_subparsers(dest='command')
    commands.required = True

    solve = commands.add_parser(
        'solve', help='solve an instance from a file or random cities',
        description='Solve a TSP instance and print the tour.')
    solve.add_argument('input', nargs='?',
                       help="TSPLIB .tsp file, or CSV with rows 'x,y[,name]'")
    solve.add_argument('--random', type=positive_int, metavar='N',
                       help='use N random cities instead of a file')
    solve.add_argument('-a', '--algo', action='append',
                       choices=list(TSPSolver.ALGORITHMS), metavar='ALGO',
                       help=f"algorithm to run; repeat to compare "
                            f"(default: {DEFAULT_ALGORITHM}). Choices: "
                            f"{', '.join(TSPSolver.ALGORITHMS)}")
    solve.add_argument('--seed', type=int,
                       help='seed for the randomized algorithms (and for --random '
                            'cities; --random defaults to 42)')
    solve.add_argument('--time-limit', type=positive_seconds, metavar='SECONDS',
                       help=f"let {', '.join(TIME_LIMITED)} keep improving the "
                            f"tour for this long instead of its default budget")
    solve.add_argument('--plot', metavar='PNG',
                       help='save a plot of the best tour (needs matplotlib)')
    solve.add_argument('--tour-out', metavar='FILE',
                       help='write the best tour, one city name per line (UTF-8)')
    solve.set_defaults(handler=cmd_solve)

    demo = commands.add_parser('demo', help='run the 20-city example comparison')
    demo.add_argument('--show', action='store_true', help='display the best tour')
    demo.add_argument('--plot', metavar='PNG', help='save a plot of the best tour')
    demo.set_defaults(handler=cmd_demo)
    return parser


def _tolerate_unencodable_output() -> None:
    """Print characters the terminal's encoding lacks as escapes instead of
    raising: a redirected stdout on Windows uses a legacy code page, and
    city names may contain anything."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors='backslashreplace')
        except (AttributeError, ValueError, OSError):
            pass  # replaced or closed stream: leave it as it is


def main(argv: Optional[List[str]] = None) -> int:
    _tolerate_unencodable_output()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.handler(args, parser)
