import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"


def load_summary():
    file_path = OUTPUT_DIR / "strategy_summary.csv"
    return pd.read_csv(file_path)


# -----------------------------
# Utility: Add labels on bars
# -----------------------------
def add_labels(ax):
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.1f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom', fontsize=9)


# -----------------------------
# Plot 1: Matched vs Unmatched
# -----------------------------
def plot_match_comparison(summary):
    strategies = summary["strategy"]
    matched = summary["matched_passengers"]
    unmatched = summary["unmatched_passengers"]

    x = range(len(strategies))

    plt.figure(figsize=(10,6))
    plt.bar(x, matched, label="Matched", color="#2E86C1")
    plt.bar(x, unmatched, bottom=matched, label="Unmatched", color="#E74C3C")

    plt.xticks(x, strategies, rotation=20)
    plt.title("Matched vs Unmatched Passengers", fontsize=14)
    plt.ylabel("Number of Passengers")
    plt.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "match_comparison.png", dpi=300)
    plt.show()


# -----------------------------
# Plot 2: Average Wait Time
# -----------------------------
def plot_wait_time(summary):
    plt.figure(figsize=(10,6))

    ax = plt.bar(summary["strategy"], summary["avg_wait_time_min"], color="#F39C12")

    plt.title("Average Wait Time by Strategy", fontsize=14)
    plt.ylabel("Minutes")
    plt.xticks(rotation=20)

    add_labels(plt.gca())

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "wait_time_comparison.png", dpi=300)
    plt.show()


# -----------------------------
# Plot 3: Average Price
# -----------------------------
def plot_price(summary):
    plt.figure(figsize=(10,6))

    ax = plt.bar(summary["strategy"], summary["avg_price"], color="#27AE60")

    plt.title("Average Price by Strategy", fontsize=14)
    plt.ylabel("Price ($)")
    plt.xticks(rotation=20)

    add_labels(plt.gca())

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_comparison.png", dpi=300)
    plt.show()


# -----------------------------
# Plot 4: Total Cost
# -----------------------------
def plot_total_cost(summary):
    plt.figure(figsize=(10,6))

    ax = plt.bar(summary["strategy"], summary["total_price"], color="#8E44AD")

    plt.title("Total Cost by Strategy", fontsize=14)
    plt.ylabel("Total Price ($)")
    plt.xticks(rotation=20)

    add_labels(plt.gca())

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "total_cost_comparison.png", dpi=300)
    plt.show()


# -----------------------------
# Main
# -----------------------------
def main():
    print("Loading summary data...")
    summary = load_summary()

    print("Generating charts...")

    plot_match_comparison(summary)
    plot_wait_time(summary)
    plot_price(summary)
    plot_total_cost(summary)

    print("Charts saved in outputs folder!")


if __name__ == "__main__":
    main()
    