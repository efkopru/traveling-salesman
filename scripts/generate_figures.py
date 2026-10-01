"""
Regenerate the README figures and results table.

Usage (from the repository root, after `pip install -e ".[dev]"`):
    python scripts/generate_figures.py

Writes PNGs to images/ and prints the README results tables as Markdown.
Distances are deterministic (fixed instance and solver seeds); times depend
on the machine.
"""

import glob
import os
import time

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tsp_solver import (TSPBenchmark, TSPSolver,  # noqa: E402
                        generate_random_cities)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGES = os.path.join(ROOT, "images")
SIZES = (20, 50, 100)
SEED = 42
TIMING_RUNS = 3

ALGORITHMS = ["nearest_neighbor", "nearest_insertion", "2-opt", "3-opt",
              "2-opt+or-opt", "iterated_local_search", "simulated_annealing",
              "genetic_algorithm"]
LABELS = {
    "nearest_neighbor": "Nearest Neighbor",
    "nearest_insertion": "Nearest Insertion",
    "2-opt": "2-Opt",
    "3-opt": "3-Opt",
    "2-opt+or-opt": "2-Opt + Or-Opt",
    "iterated_local_search": "Iterated Local Search",
    "simulated_annealing": "Simulated Annealing",
    "genetic_algorithm": "Genetic Algorithm",
}
TSPLIB_FILES = sorted(glob.glob(os.path.join(ROOT, "data", "tsplib", "*.tsp")))
SCALING_SIZES = (200, 500, 1000)
SCALING_ALGORITHMS = ["2-opt", "2-opt+or-opt", "iterated_local_search"]

# Chart chrome (light surface) and series color
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "text.color": INK,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def run_all():
    """Solve every instance with every algorithm.

    Distances come from the first run; time is the fastest of TIMING_RUNS
    runs, each with a fresh solver so seeded results are identical.
    """
    results = {}
    for n in SIZES:
        cities = generate_random_cities(n)
        names = [f"C{i}" for i in range(n)]
        runs = [TSPSolver(cities, names, seed=SEED).compare_algorithms(ALGORITHMS)
                for _ in range(TIMING_RUNS)]
        best = min(r["distance"] for r in runs[0].values())
        results[n] = {
            "solver": TSPSolver(cities, names, seed=SEED),
            "best": best,
            "algos": {
                a: {
                    "tour": runs[0][a]["tour"],
                    "distance": runs[0][a]["distance"],
                    "gap": 100 * (runs[0][a]["distance"] - best) / best,
                    "time": min(r[a]["time"] for r in runs),
                }
                for a in ALGORITHMS
            },
        }
    return results


def save(fig, name):
    path = os.path.join(IMAGES, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {os.path.relpath(path, ROOT)}")


def plot_best_tour(results):
    """Best tour on the 20-city instance, using the library's own plot."""
    data = results[20]
    algo = min(ALGORITHMS, key=lambda a: data["algos"][a]["distance"])
    fig = data["solver"].visualize_tour(
        data["algos"][algo]["tour"], f"Best Tour, 20 cities ({LABELS[algo]})",
        show=False)
    save(fig, "best_tour_20.png")


def plot_tour_comparison(results, n=50):
    """Small multiples: the tour each algorithm finds on one instance."""
    data = results[n]
    cities = data["solver"].cities
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.6))
    for ax, algo in zip(axes.flat, ALGORITHMS):
        info = data["algos"][algo]
        tour = info["tour"] + info["tour"][:1]
        ax.plot(cities[tour, 0], cities[tour, 1], color=SERIES, linewidth=2,
                solid_joinstyle="round", zorder=2)
        ax.scatter(cities[:, 0], cities[:, 1], s=22, color=INK_SECONDARY,
                   edgecolor=SURFACE, linewidth=1.5, zorder=3)
        gap = "best" if info["gap"] < 0.005 else f"+{info['gap']:.1f}%"
        ax.set_title(f"{LABELS[algo]}\n", loc="left", fontsize=11,
                     fontweight="bold")
        ax.text(0, 1.02, f"{info['distance']:.1f}  ({gap})",
                transform=ax.transAxes, color=INK_SECONDARY, fontsize=10)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GRID)
    fig.suptitle(f"Tours found on a {n}-city instance (distance, gap to best)",
                 x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=3)
    save(fig, f"tour_comparison_{n}.png")


def gap_panels(panels, title, xlabel, zero_label, filename, panel_width):
    """
    Small multiples of horizontal bars, one panel per (panel title, gaps)
    pair; gaps are % values per algorithm in ALGORITHMS order.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(panel_width * len(panels), 3.9),
                             sharey=True)
    order = list(reversed(ALGORITHMS))
    max_gap = max(max(gaps.values()) for _, gaps in panels)
    for ax, (panel_title, gaps) in zip(axes, panels):
        values = [gaps[a] for a in order]
        ax.barh(range(len(order)), values, height=0.6, color=SERIES)
        for y, g in enumerate(values):
            text = zero_label if g < 0.005 else f"+{g:.1f}%"
            ax.text(g + max_gap * 0.02, y, text, va="center",
                    color=INK_SECONDARY, fontsize=9)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABELS[a] for a in order], color=INK_SECONDARY)
        ax.set_xlim(0, max_gap * 1.3)
        ax.set_title(panel_title, loc="left", fontsize=11, fontweight="bold")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_color(BASELINE)
        ax.set_xlabel(xlabel)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, filename)


def plot_gap_by_size(results):
    """% above the best result, one panel per instance size."""
    panels = [(f"{n} cities", {a: results[n]["algos"][a]["gap"] for a in ALGORITHMS})
              for n in SIZES]
    gap_panels(panels, "Solution quality by instance size",
               "% above best result (lower is better)", "best",
               "quality_by_size.png", panel_width=4)


def plot_quality_vs_time(results, n=100):
    """Scatter: runtime vs. gap to best on the largest instance."""
    data = results[n]["algos"]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [data[a]["time"] for a in ALGORITHMS]
    ys = [data[a]["gap"] for a in ALGORITHMS]
    ax.scatter(xs, ys, s=64, color=SERIES, edgecolor=SURFACE, linewidth=2,
               zorder=3)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} s"))
    ax.set_xlim(min(xs) / 3, max(xs) * 8)
    ax.set_ylim(-max(ys) * 0.12, max(ys) * 1.12)
    ax.set_xlabel("Runtime (log scale)")
    ax.set_ylabel("% above best result")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Quality vs. runtime, {n} cities (lower left is better)",
                 loc="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    place_labels(ax, xs, ys, [LABELS[a] for a in ALGORITHMS])
    save(fig, f"quality_vs_time_{n}.png")


def place_labels(ax, xs, ys, labels, pad_px=10, gap_px=3, dot_px=7):
    """
    Label each point beside it without overlapping other labels or points.

    For each point (bottom to top) try, in order: right of the point, left
    of the point, then right with increasing vertical shifts (alternating
    up and down). A thin leader line joins a label that had to move away
    from its point. Works in pixels, so it is independent of the log axis
    and of where the timing-dependent points land.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    from_px = ax.transData.inverted().transform
    points = [to_px((x, y)) for x, y in zip(xs, ys)]
    obstacles = [(px - dot_px, py - dot_px, px + dot_px, py + dot_px)
                 for px, py in points]

    def overlaps(box):
        x0, y0, x1, y1 = box
        return any(x0 < b1 and b0 < x1 and y0 < c1 and c0 < y1
                   for b0, c0, b1, c1 in obstacles)

    order = sorted(range(len(labels)), key=lambda i: points[i][1])
    for i in order:
        px, py = points[i]
        probe = ax.text(0, 0, labels[i], fontsize=9)
        box = probe.get_window_extent(renderer)
        probe.remove()
        w, h = box.width, box.height

        candidates = [(px + pad_px, py), (px - pad_px - w, py)]
        for step in range(1, 12):
            shift = step * (h + gap_px) / 2
            candidates += [(px + pad_px, py + shift), (px + pad_px, py - shift),
                           (px - pad_px - w, py + shift), (px - pad_px - w, py - shift)]
        for lx, ly in candidates:
            rect = (lx - gap_px, ly - h / 2 - gap_px, lx + w + gap_px, ly + h / 2 + gap_px)
            if not overlaps(rect):
                break
        obstacles.append(rect)

        x, y = from_px((px, py))
        tx, ty = from_px((lx, ly))
        moved = abs(ly - py) > 1
        ax.annotate(labels[i], (x, y), xytext=(tx, ty), textcoords="data",
                    va="center", ha="left", color=INK_SECONDARY, fontsize=9,
                    arrowprops=dict(arrowstyle="-", color=BASELINE, linewidth=0.8,
                                    shrinkA=2, shrinkB=5) if moved else None)


def print_table(results):
    print("\n| Algorithm | " + " | ".join(f"{n} cities" for n in SIZES)
          + f" | Time at {SIZES[-1]} cities (s) |")
    print("|-----------|" + "-----------|" * len(SIZES)
          + "------------------------|")
    for a in ALGORITHMS:
        cells = []
        for n in SIZES:
            info = results[n]["algos"][a]
            if info["gap"] < 0.005:
                cells.append(f"**{info['distance']:.2f}**")
            else:
                cells.append(f"{info['distance']:.2f} (+{info['gap']:.1f}%)")
        t = results[SIZES[-1]]["algos"][a]["time"]
        print(f"| {LABELS[a]} | " + " | ".join(cells) + f" | {t:.4f} |")


def run_tsplib():
    """Gap to the published optimum on the bundled TSPLIB instances."""
    df = TSPBenchmark.run_tsplib_benchmark(TSPLIB_FILES, ALGORITHMS, seed=SEED)
    order = {name: i for i, name in enumerate(
        df.drop_duplicates("Instance").sort_values("Cities")["Instance"])}
    return df.assign(order=df["Instance"].map(order)).sort_values("order")


def plot_tsplib_gap(df):
    """% above the known optimum, one panel per TSPLIB instance."""
    panels = []
    for name in dict.fromkeys(df["Instance"]):
        sub = df[df["Instance"] == name].set_index("Algorithm")
        panels.append((f"{name} (optimum {int(sub['Optimum'].iloc[0])})",
                       {a: sub.loc[a, "Gap (%)"] for a in ALGORITHMS}))
    gap_panels(panels, "TSPLIB instances: gap to the published optimal tour",
               "% above optimum", "optimal", "tsplib_gap.png", panel_width=3.75)


def print_tsplib_table(df):
    instances = list(dict.fromkeys(df["Instance"]))
    optima = df.drop_duplicates("Instance").set_index("Instance")["Optimum"]
    print("\n| Algorithm | " + " | ".join(
        f"{name} ({int(optima[name])})" for name in instances) + " |")
    print("|-----------|" + "-----------|" * len(instances))
    for a in ALGORITHMS:
        cells = []
        for name in instances:
            row = df[(df["Instance"] == name) & (df["Algorithm"] == a)].iloc[0]
            if row["Gap (%)"] < 0.005:
                cells.append(f"**{row['Distance']:.0f}**")
            else:
                cells.append(f"{row['Distance']:.0f} (+{row['Gap (%)']:.1f}%)")
        print(f"| {LABELS[a]} | " + " | ".join(cells) + " |")


def print_scaling_table():
    """Runtime and length of the local searches on larger random instances."""
    print("\n| Algorithm | " + " | ".join(f"{n} cities" for n in SCALING_SIZES) + " |")
    print("|-----------|" + "-----------|" * len(SCALING_SIZES))
    rows = {a: [] for a in SCALING_ALGORITHMS}
    for n in SCALING_SIZES:
        results = TSPSolver(generate_random_cities(n), seed=SEED) \
            .compare_algorithms(SCALING_ALGORITHMS)
        for a in SCALING_ALGORITHMS:
            rows[a].append(f"{results[a]['distance']:.1f} ({results[a]['time']:.2f} s)")
    for a in SCALING_ALGORITHMS:
        print(f"| {LABELS[a]} | " + " | ".join(rows[a]) + " |")


def print_exact_20():
    solver = TSPSolver(generate_random_cities(20))
    start = time.perf_counter()
    _, distance = solver.held_karp()
    print(f"\nHeld-Karp optimum, 20 cities: {distance:.2f} "
          f"({time.perf_counter() - start:.2f} s)")


def main():
    os.makedirs(IMAGES, exist_ok=True)
    results = run_all()
    plot_best_tour(results)
    plot_tour_comparison(results)
    plot_gap_by_size(results)
    plot_quality_vs_time(results)
    tsplib = run_tsplib()
    plot_tsplib_gap(tsplib)
    print_table(results)
    print_tsplib_table(tsplib)
    print_scaling_table()
    print_exact_20()


if __name__ == "__main__":
    main()
