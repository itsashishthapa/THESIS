"""Draw the thesis HTF flow chart using the manuscript's notation.

Topology adapted from Fig. 3 of Kyriakidis et al. (2024), as cited in main.tex.
Only diagram geometry and labels are generated; no simulation data are used.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle


def draw_flow_chart():
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm"})
    figure, axis = plt.subplots(figsize=(14, 3.4))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 3.4)
    axis.axis("off")

    def line(points, dashed=False):
        xs, ys = zip(*points)
        axis.plot(xs, ys, color="black", linewidth=1.5,
                  linestyle=(0, (4, 3)) if dashed else "-")

    def arrow(start, end):
        axis.add_patch(FancyArrowPatch(start, end, arrowstyle="->",
                                      mutation_scale=13, linewidth=1.5,
                                      color="black", shrinkA=0, shrinkB=0))

    def label(x, y, text, **kwargs):
        axis.text(x, y, text, fontsize=12, ha="center", va="center", **kwargs)

    def valve(x, y, branch_direction):
        # Three triangles meet at the centre; the third faces the bypass pipe.
        triangles = (
            [(x, y), (x - 0.19, y - 0.12), (x - 0.19, y + 0.12)],
            [(x, y), (x + 0.19, y - 0.12), (x + 0.19, y + 0.12)],
            [(x, y), (x - 0.12, y + branch_direction * 0.19),
             (x + 0.12, y + branch_direction * 0.19)],
        )
        for vertices in triangles:
            axis.add_patch(Polygon(vertices,
                                   closed=True, facecolor="white",
                                   edgecolor="black", linewidth=1, zorder=3))

    for x, width, name in ((0.7, 2.0, "HTHP"), (6.0, 2.0, "TES"),
                           (11.1, 2.0, "SG")):
        axis.add_patch(Rectangle((x, 0.75), width, 1.65, facecolor="white",
                                 edgecolor="black", linewidth=1.2))
        axis.text(x + width / 2, 1.575, name, ha="center", va="center",
                  fontsize=24, fontfamily="serif")

    # HTHX outlet -> charging valve -> TES -> mixing valve -> SG.
    arrow((2.7, 1.15), (4.11, 1.15))
    arrow((4.49, 1.15), (6.0, 1.15))
    arrow((8.0, 1.15), (9.46, 1.15))
    arrow((9.84, 1.15), (11.1, 1.15))
    label(3.4, 0.91, r"$T_{\mathrm{II},t},\dot m_{\mathrm{II},t}$")
    label(5.2, 0.91, r"$(1-\beta_{1,t})\dot m_{\mathrm{II},t}$")
    label(8.7, 0.91, r"$T_{1,t}$")
    label(10.5, 0.91, r"$T_{2,t},\dot m_{\mathrm{II},t}$")

    # SG return -> discharging valve -> TES -> mixing valve -> HTHX inlet.
    arrow((11.1, 2.0), (9.84, 2.0))
    line([(9.46, 2.0), (8.0, 2.0)], dashed=True)
    line([(6.0, 2.0), (4.49, 2.0)], dashed=True)
    arrow((4.11, 2.0), (2.7, 2.0))
    label(10.5, 1.77, r"$T_{3,t},\dot m_{\mathrm{I},t}$")
    label(8.7, 1.77, r"$(1-\beta_{2,t})\dot m_{\mathrm{I},t}$")
    label(5.2, 1.77, r"$T_{4,t}$")
    label(3.4, 1.77, r"$T_{\mathrm{I},t},\dot m_{\mathrm{I},t}$")

    # Bypass branches and mixing valves.
    line([(9.65, 2.0), (9.65, 2.98), (4.3, 2.98), (4.3, 2.24)])
    arrow((4.3, 2.24), (4.3, 2.19))
    label(8.7, 3.17, r"$\beta_{2,t}\dot m_{\mathrm{I},t}$")
    line([(4.3, 1.15), (4.3, 0.27), (9.65, 0.27), (9.65, 0.91)],
         dashed=True)
    arrow((9.65, 0.91), (9.65, 0.96))
    label(5.7, 0.48, r"$\beta_{1,t}\dot m_{\mathrm{II},t}$")
    for x in (4.3, 9.65):
        for y, branch_direction in ((1.15, -1), (2.0, 1)):
            valve(x, y, branch_direction)

    # Low-temperature air side and electrical supply.
    arrow((1.3, 3.12), (1.3, 2.4))
    label(0.6, 2.94, r"$T_{\mathrm{III},t}$")
    arrow((1.3, 0.75), (1.3, 0.12))
    label(0.6, 0.41, r"$T_{\mathrm{IV},t}$")
    arrow((2.25, 3.12), (2.25, 2.4))
    label(3.05, 2.91, r"$P_t^{\mathrm{S}}+W_t$")

    figure.tight_layout(pad=0.15)
    destination = Path(__file__).resolve().parents[1] / "paper" / "figures"
    for extension in ("pdf", "svg"):
        figure.savefig(destination / f"cookie_detailed_flow_chart.{extension}",
                       bbox_inches="tight", pad_inches=0.05)
    plt.close(figure)


if __name__ == "__main__":
    draw_flow_chart()
