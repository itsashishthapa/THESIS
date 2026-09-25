"""Plot readable cookie-factory objective comparisons from saved evaluation data."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "cookie_factory_comparison"
OUTPUT = ROOT / "paper" / "figures" / "cookie_factory_objective_comparison.png"
N_WINDOWS = 10


def successful_rows(path: Path) -> pd.DataFrame:
    """Return successful comparisons in their recorded evaluation order."""
    data = pd.read_csv(path)
    return data.loc[data["status"].eq("success")].reset_index(drop=True)


def main() -> None:
    sac = successful_rows(RESULTS / "model_comparison_sac_summary.csv")
    td3 = successful_rows(RESULTS / "model_comparison_td3_summary.csv")

    if not sac["start_idx"].equals(td3["start_idx"]):
        raise ValueError("SAC and TD3 successful windows are not aligned.")

    n = min(N_WINDOWS, len(sac), len(td3))
    sac = sac.iloc[:n]
    td3 = td3.iloc[:n]

    x = np.arange(n)
    td3_cost = td3["rl_cost"].to_numpy(float)
    # The first optimization run (stored with SAC) is the common baseline.
    optimization_cost = sac["pyomo_cost"].to_numpy(float)
    td3_gap = np.abs(td3_cost - optimization_cost)

    window_labels = [
        f"Window {i + 1}\n({start}–{start + 24})"
        for i, start in enumerate(sac["start_idx"].astype(int))
    ]

    plt.rcParams.update({"font.size": 12})
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(16, 10.5),
        gridspec_kw={"height_ratios": [1.15, 0.93]},
    )
    figure.subplots_adjust(
        hspace=0.43, bottom=0.13, top=0.93, left=0.08, right=0.97
    )

    td3_color = "#0b5bb5"
    optimization_color = "#ff7a00"
    bar_color = "#1f62b5"
    for axis in axes:
        axis.set_axisbelow(True)
        axis.grid(True, linestyle="--", linewidth=0.8, alpha=0.4)

    axes[0].plot(
        x,
        td3_cost,
        marker="o",
        markersize=8,
        linewidth=2,
        color=td3_color,
        label="TD3 (RL)",
    )
    axes[0].plot(
        x,
        optimization_cost,
        marker="o",
        markersize=8,
        linewidth=2,
        color=optimization_color,
        label="Optimization baseline",
    )
    axes[0].set_title(
        "TD3 vs. Optimization Baseline Across 24-Hour Cookie-Factory Windows",
        fontsize=20,
        fontweight="bold",
        pad=14,
    )
    axes[0].set_ylabel("Objective value [EUR]", fontsize=14, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(window_labels, fontsize=10, fontweight="bold")
    axes[0].legend(loc="upper right", frameon=True, fancybox=True, framealpha=0.95)

    bars = axes[1].bar(
        x,
        td3_gap,
        width=0.58,
        color=bar_color,
        alpha=0.85,
    )

    mean_gap = float(td3_gap.mean())
    axes[1].axhline(
        mean_gap,
        color="#d7191c",
        linestyle="--",
        linewidth=1.4,
    )
    axes[1].text(
        max(0, n / 2 - 0.2),
        mean_gap + max(8.0, 0.035 * td3_gap.max()),
        f"Mean gap = EUR {mean_gap:.2f}",
        color="#c41218",
        fontsize=13,
        fontweight="bold",
        ha="center",
    )

    label_offset = max(float(td3_gap.max()) * 0.025, 4.0)
    for bar, gap in zip(bars, td3_gap):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{gap:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ranked = np.argsort(td3_gap)[::-1]
    for rank, index in enumerate(ranked[:2], start=1):
        annotation = "Largest gap" if rank == 1 else "2nd largest gap"
        axes[1].annotate(
            f"{annotation}\nEUR {td3_gap[index]:.2f}",
            xy=(index, td3_gap[index]),
            xytext=(0, 26),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.45", "fc": "white", "ec": "#e74c3c", "lw": 0.8},
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "black"},
        )

    axes[1].set_title(
        "Objective Gap to Optimization Baseline (|TD3 - Optimization baseline|)",
        fontsize=18,
        fontweight="bold",
        pad=12,
    )
    axes[1].set_ylabel("Objective gap [EUR]", fontsize=14, fontweight="bold")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"Window {i + 1}" for i in range(n)], fontweight="bold")
    axes[1].set_xlim(-0.6, n - 0.4)
    axes[1].set_ylim(0, td3_gap.max() * 1.25)

    figure.text(
        0.80,
        0.065,
        "Lower objective value = lower operating cost",
        fontsize=11,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.7", "fc": "#eef4ff", "ec": "#7aa0d5"},
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(figure)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
