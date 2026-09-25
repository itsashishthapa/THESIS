"""Plot the smooth battery power mappings used in the thesis."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


P_MAX = 20.0
ETA = 0.9
K = 700.0
DELTA = 0.0025
EPSILON = 1.0


def smooth_gate(value: np.ndarray | float) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(K * value))


def grid_side_power(action: np.ndarray, soc: float) -> np.ndarray:
    charge_action_gate = smooth_gate(action - DELTA)
    discharge_action_gate = 1.0 - smooth_gate(action + DELTA)
    charge_soc_gate = smooth_gate(0.995 - soc)
    discharge_soc_gate = smooth_gate(soc - 0.005)
    return P_MAX * action * (
        charge_action_gate * charge_soc_gate
        + discharge_action_gate * discharge_soc_gate
    )


def storage_side_power(grid_power: np.ndarray) -> np.ndarray:
    transition = np.tanh(grid_power / EPSILON)
    efficiency_factor = 0.5 * (
        (1.0 + transition) * ETA
        + (1.0 - transition) / ETA
    )
    return grid_power * efficiency_factor


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
        }
    )

    action = np.linspace(-1.0, 1.0, 2001)
    grid_power = np.linspace(-P_MAX, P_MAX, 2001)
    soc_values = (0.0, 0.005, 0.5, 0.995, 1.0)
    colors = ("#0072B2", "#56B4E9", "#222222", "#E69F00", "#D55E00")

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)

    for soc, color in zip(soc_values, colors, strict=True):
        axes[0].plot(
            action,
            grid_side_power(action, soc),
            color=color,
            linewidth=1.8,
            label=fr"$\mathrm{{SOC}}_t={soc:g}$",
        )
    axes[0].axhline(0.0, color="0.65", linewidth=0.7)
    axes[0].axvline(0.0, color="0.65", linewidth=0.7)
    axes[0].set_xlabel(r"Normalized action $u_t$")
    axes[0].set_title(r"(a) Action-to-power mapping")
    axes[0].set_xlim(-1.0, 1.0)
    axes[0].set_ylim(-23.0, 23.0)
    axes[0].set_yticks((-20.0, -10.0, 0.0, 10.0, 20.0))
    axes[0].legend(loc="upper left", ncol=2, frameon=True)

    actual_power = storage_side_power(grid_power)
    axes[1].plot(
        grid_power,
        actual_power,
        color="#0072B2",
        linewidth=2.0,
        label=r"Smooth mapping ($\eta=0.9$)",
    )
    axes[1].plot(
        grid_power,
        grid_power,
        color="0.45",
        linestyle="--",
        linewidth=1.0,
        label=r"Lossless: $P_{\mathrm{actual},t}=P_t^{\mathrm{S}}$",
    )
    axes[1].axhline(0.0, color="0.65", linewidth=0.7)
    axes[1].axvline(0.0, color="0.65", linewidth=0.7)
    axes[1].set_xlabel(r"Grid-side power $P_t^{\mathrm{S}}$ [kW]")
    axes[1].set_title(r"(b) Smooth efficiency mapping")
    axes[1].set_xlim(-P_MAX, P_MAX)
    axes[1].set_ylim(-23.0, 23.0)
    axes[1].set_yticks((-20.0, -10.0, 0.0, 10.0, 20.0))
    axes[1].legend(loc="upper left", frameon=True)

    for axis in axes:
        axis.grid(alpha=0.25)

    figure.supylabel("Power [kW]")
    figure.tight_layout(w_pad=2.0)
    output = Path(__file__).resolve().parents[1] / "paper" / "figures" / "battery_smooth_mappings.pdf"
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
