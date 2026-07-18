"""
Task 14 — Analyze episode difficulty CSV and plot distribution.

Reads the per-episode CSV from difficulty_eval.py and produces:
  1. Console summary (easy / hard / boundary counts)
  2. Histogram PNG of pass_rate distribution
  3. Bucketed breakdown table

Usage:
  python vln/analyze_difficulty.py results/difficulty/episode_difficulty.csv \
    --output-plot results/difficulty/difficulty_distribution.png

Works locally (no Habitat / GPU needed).
"""

import csv
import argparse
import os


def load_csv(path):
    rows = []
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["pass_rate"] = float(row["pass_rate"])
            row["success_count"] = int(row["success_count"])
            row["pass_k"] = int(row["pass_k"])
            row["avg_ne"] = float(row["avg_ne"])
            row["avg_spl"] = float(row["avg_spl"])
            row["avg_steps"] = float(row["avg_steps"])
            rows.append(row)
    return rows


def print_summary(rows):
    n = len(rows)
    if n == 0:
        print("No episodes found.")
        return

    pass_rates = [r["pass_rate"] for r in rows]
    n_easy = sum(1 for p in pass_rates if p == 1.0)
    n_hard = sum(1 for p in pass_rates if p == 0.0)
    n_boundary = n - n_easy - n_hard
    avg_pr = sum(pass_rates) / n

    print(f"\n{'='*60}")
    print(f"Episode Difficulty Distribution (N={n})")
    print(f"{'='*60}")
    print(f"  Easy    (pass_rate = 1.0):  {n_easy:5d}  ({100*n_easy/n:5.1f}%)")
    print(f"  Hard    (pass_rate = 0.0):  {n_hard:5d}  ({100*n_hard/n:5.1f}%)")
    print(f"  Boundary(0 < rate < 1  ):  {n_boundary:5d}  ({100*n_boundary/n:5.1f}%)")
    print(f"  Avg pass_rate:             {avg_pr:.3f}")
    print()

    # bucketed breakdown
    buckets = [
        ("0.0      ", lambda p: p == 0.0),
        ("(0, 0.25]", lambda p: 0.0 < p <= 0.25),
        ("(0.25,0.5]", lambda p: 0.25 < p <= 0.5),
        ("(0.5,0.75]", lambda p: 0.5 < p <= 0.75),
        ("(0.75, 1)", lambda p: 0.75 < p < 1.0),
        ("1.0      ", lambda p: p == 1.0),
    ]
    print(f"  {'Bucket':<12} {'Count':>6} {'Pct':>7}  {'Bar'}")
    print(f"  {'-'*12} {'-'*6} {'-'*7}  {'-'*30}")
    for label, pred in buckets:
        cnt = sum(1 for p in pass_rates if pred(p))
        pct = 100 * cnt / n
        bar = '#' * int(pct / 2)
        print(f"  {label:<12} {cnt:6d} {pct:6.1f}%  {bar}")
    print(f"{'='*60}\n")

    # GRPO signal analysis
    print("GRPO Signal Quality:")
    print(f"  Episodes with gradient signal (0 < pass_rate < 1): {n_boundary}/{n} = {100*n_boundary/n:.1f}%")
    print(f"  Episodes wasted (pass_rate=0 or 1, advantage=0):   {n_easy+n_hard}/{n} = {100*(n_easy+n_hard)/n:.1f}%")
    if n_boundary > 0:
        boundary_rates = [p for p in pass_rates if 0 < p < 1]
        avg_boundary = sum(boundary_rates) / len(boundary_rates)
        print(f"  Avg pass_rate among boundary episodes:             {avg_boundary:.3f}")
    print()


def plot_histogram(rows, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot.")
        return

    pass_rates = [r["pass_rate"] for r in rows]
    n = len(pass_rates)
    n_easy = sum(1 for p in pass_rates if p == 1.0)
    n_hard = sum(1 for p in pass_rates if p == 0.0)
    n_boundary = n - n_easy - n_hard

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    bins = [i / 20 for i in range(21)]  # 0.0, 0.05, ..., 1.0
    ax.hist(pass_rates, bins=bins, edgecolor="black", alpha=0.7, color="#4C72B0")
    ax.set_xlabel("Episode Pass Rate (success_count / K)", fontsize=12)
    ax.set_ylabel("Number of Episodes", fontsize=12)
    ax.set_title(
        f"Episode Difficulty Distribution (N={n}, K={rows[0]['pass_k']})\n"
        f"Easy={n_easy} ({100*n_easy/n:.0f}%)  |  "
        f"Boundary={n_boundary} ({100*n_boundary/n:.0f}%)  |  "
        f"Hard={n_hard} ({100*n_hard/n:.0f}%)",
        fontsize=11,
    )
    ax.axvline(x=0.0, color="red", linestyle="--", alpha=0.5, label="Hard (no signal)")
    ax.axvline(x=1.0, color="red", linestyle="--", alpha=0.5, label="Easy (no signal)")
    ax.set_xlim(-0.05, 1.05)
    ax.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Histogram saved: {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze episode difficulty distribution")
    parser.add_argument("csv_path", type=str, help="Path to episode_difficulty.csv")
    parser.add_argument("--output-plot", type=str, default="",
                        help="Output histogram PNG path (default: same dir as CSV)")
    args = parser.parse_args()

    if not args.output_plot:
        base = os.path.dirname(args.csv_path)
        args.output_plot = os.path.join(base, "difficulty_distribution.png")

    rows = load_csv(args.csv_path)
    print_summary(rows)
    plot_histogram(rows, args.output_plot)


if __name__ == "__main__":
    main()
