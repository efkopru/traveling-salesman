"""
Regenerate the README figures and results table.

Usage (from the repository root, after `pip install -e ".[dev]"`):
    python scripts/generate_figures.py

Writes PNGs to images/ and prints the README results table as Markdown.
Distances are deterministic (fixed instance and solver seeds); times depend
on the machine.
"""

import os

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tsp_solver import TSPSolver, generate_random_cities  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IMAGES = os.path.join(ROOT, "images")
SIZES = (20, 50, 100)
SEED = 42
TIMING_RUNS = 3

ALGORITHMS = ["nearest_neighbor", "nearest_insertion", "2-opt", "3-opt",
              "simulated_annealing", "genetic_algorithm"]
LABELS = {
    "nearest_neighbor": "Nearest Neighbor",
    "nearest_insertion": "Nearest Insertion",
    "2-opt": "2-Opt",
    "3-opt": "3-Opt",
    "simulated_annealing": "Simulated Annealing",
    "genetic_algorithm": "Genetic Algorithm",
}

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
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.4))
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


def plot_gap_by_size(results):
    """Small multiples: % above the best result, one panel per instance size."""
    fig, axes = plt.subplots(1, len(SIZES), figsize=(12, 3.8), sharey=True)
    order = list(reversed(ALGORITHMS))
    max_gap = max(results[n]["algos"][a]["gap"] for n in SIZES for a in ALGORITHMS)
    for ax, n in zip(axes, SIZES):
        gaps = [results[n]["algos"][a]["gap"] for a in order]
        ax.barh(range(len(order)), gaps, height=0.6, color=SERIES)
        for y, g in enumerate(gaps):
            text = "best" if g < 0.005 else f"+{g:.1f}%"
            ax.text(g + max_gap * 0.02, y, text, va="center",
                    color=INK_SECONDARY, fontsize=9)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([LABELS[a] for a in order], color=INK_SECONDARY)
        ax.set_xlim(0, max_gap * 1.25)
        ax.set_title(f"{n} cities", loc="left", fontsize=11, fontweight="bold")
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_color(BASELINE)
        ax.set_xlabel("% above best result (lower is better)")
    fig.suptitle("Solution quality by instance size", x=0.02, ha="left",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "quality_by_size.png")


def plot_quality_vs_time(results, n=100):
    """Scatter: runtime vs. gap to best on the largest instance."""
    data = results[n]["algos"]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [data[a]["time"] for a in ALGORITHMS]
    ys = [data[a]["gap"] for a in ALGORITHMS]
    ax.scatter(xs, ys, s=64, color=SERIES, edgecolor=SURFACE, linewidth=2,
               zorder=3)
    # 3-Opt and Simulated Annealing sit close together near zero, so their
    # labels go on opposite sides.
    offsets = {"3-opt": (-8, 8, "right"), "simulated_annealing": (8, 8, "left")}
    for a, x, y in zip(ALGORITHMS, xs, ys):
        dx, dy, ha = offsets.get(a, (8, 4, "left"))
        ax.annotate(LABELS[a], (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, color=INK_SECONDARY, fontsize=9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g} s"))
    ax.set_xlim(min(xs) / 3, max(xs) * 8)
    ax.set_ylim(-5, max(ys) * 1.12)
    ax.set_xlabel("Runtime (log scale)")
    ax.set_ylabel("% above best result")
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(f"Quality vs. runtime, {n} cities (lower left is better)",
                 loc="left", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, f"quality_vs_time_{n}.png")


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


def main():
    os.makedirs(IMAGES, exist_ok=True)
    results = run_all()
    plot_best_tour(results)
    plot_tour_comparison(results)
    plot_gap_by_size(results)
    plot_quality_vs_time(results)
    print_table(results)


if __name__ == "__main__":
    main()
