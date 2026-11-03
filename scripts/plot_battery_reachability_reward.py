"""Plot the thesis reachability violation and reward for an idle battery."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


N = 24
P_MAX = 20.0
E_MAX = 200.0
ETA = 0.9
DT_H = 1.0
SOC_TERM = 0.5
LAMBDA_REACH = 10.0


def reachability_threshold(time_step: int) -> float:
    return max(SOC_TERM - (N - time_step) * ETA * P_MAX * DT_H / E_MAX, 0.0)


def main() -> None:
    plt.rcParams.update(
        {"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 10,
         "legend.fontsize": 8, "figure.dpi": 150}
    )
    soc = np.linspace(0.0, 1.0, 1001)
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    # All selected steps precede the final decision. For u_t=0, both
    # the grid cost and the baseline terminal penalty are zero: r_t=0.
    for time_step, color, linestyle in (
        (18, "#666666", ":"),
        (20, "#0072B2", "--"),
        (22, "#D55E00", "-"),
    ):
        threshold = reachability_threshold(time_step)
        # Include each kink explicitly, independent of the plotting resolution.
        curve_soc = np.unique(np.append(soc, threshold))
        violation = np.maximum(threshold - curve_soc, 0.0)
        reward = -LAMBDA_REACH * violation
        label = fr"$t={time_step}$, $\mathrm{{SOC}}_t^{{\mathrm{{reach}}}}={threshold:.2f}$"
        for axis, values in zip(axes, (violation, reward)):
            axis.plot(curve_soc, values, color=color, linestyle=linestyle,
                      linewidth=1.7, label=label)
            axis.plot(threshold, 0.0, marker="o", markersize=4,
                      markerfacecolor="white", markeredgecolor=color,
                      zorder=4)
            if threshold > 0.0:
                axis.axvline(threshold, color=color, linestyle=":",
                             linewidth=0.8, alpha=0.4)

    axes[0].set_title("(a) Reachability violation")
    axes[0].set_ylabel(r"Violation $v_{\mathrm{reach},t}$")
    axes[0].set_ylim(-0.02, 0.35)
    axes[0].legend(loc="upper right", frameon=True)
    axes[1].set_title(r"(b) Shaped reward for $u_t=0$")
    axes[1].set_ylabel(r"Reward $r_t^{\mathrm{reach}}$")
    axes[1].set_ylim(-3.5, 0.2)
    axes[1].text(0.97, 0.1, r"$r_t=0$, $\lambda_{\mathrm{reach}}=10$",
                 transform=axes[1].transAxes, ha="right", va="bottom")
    for axis in axes:
        axis.set_xlabel(r"State of charge $\mathrm{SOC}_t$")
        axis.set_xlim(0.0, 1.0)
        axis.grid(alpha=0.25)
    figure.tight_layout(w_pad=2.0)
    output = Path(__file__).resolve().parents[1] / "paper" / "figures" / "battery_reachability_reward.pdf"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
