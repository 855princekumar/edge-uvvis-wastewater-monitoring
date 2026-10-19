#!/usr/bin/env python3
"""Graphical abstract (journal requirement): pipeline and key results. Numbers are those of the paper:
time-blocked R2 from results/batch-8_smoothsnv_blocked_215ch, Pi timings from fleet/real_pi and fleet run B,
soak from fleet/runs/soak_24h_1hz, Raman threshold from simulation/. Usage: python figures/make_graphical_abstract.py figures/out"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "figures/out"); OUT.mkdir(parents=True, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "grey": "#6b7280", "ink": "#1f2937"}
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig = plt.figure(figsize=(13, 5.2)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 26); ax.set_ylim(0, 10.4); ax.axis("off")


def panel(x, w, title, lines, ec, fc, ls="-"):
    ax.add_patch(FancyBboxPatch((x, 1.0), w, 7.6, boxstyle="round,pad=0.05,rounding_size=0.3", fc=fc, ec=ec, lw=2, ls=ls))
    ax.text(x + w / 2, 7.95, title, ha="center", va="center", fontsize=15, weight="bold", color=C["ink"])
    y = 6.9
    for t, fs, col in lines:
        ax.text(x + w / 2, y, t, ha="center", va="top", fontsize=fs, color=col, linespacing=1.3); y -= 0.42 * fs / 10 * (t.count("\n") + 1) + 0.55


def arrow(x1, x2):
    ax.add_patch(FancyArrowPatch((x1, 4.8), (x2, 4.8), arrowstyle="-|>", mutation_scale=26, lw=2.4, color=C["ink"]))


ax.text(13, 9.7, "Edge-enabled UV-Vis wastewater monitoring: tested on real data and real edge hardware", ha="center", fontsize=17, weight="bold", color=C["ink"])
panel(0.3, 5.4, "Real wastewater", [("25 weeks of raw municipal\nwastewater (open data)", 12.5, C["ink"]), ("510 UV-Vis spectra\n200-735 nm, matched to\nlaboratory grab samples", 12.5, C["ink"])], C["grey"], "#f3f4f6")
arrow(5.8, 6.6)
# accuracy panel with bars
x0 = 6.7; panel(x0, 6.8, "Time-blocked R$^2$", [], C["blue"], "#eaf2fc")
bx = fig.add_axes([0.295, 0.29, 0.2, 0.38]); vals = {"NH$_4^+$": (0.821, 0.596), "PO$_4^{3-}$": (0.564, 0.352)}
xs = range(len(vals)); w = 0.36
bx.bar([i - w / 2 for i in xs], [v[0] for v in vals.values()], w, color=C["blue"], label="PLS")
bx.bar([i + w / 2 for i in xs], [v[1] for v in vals.values()], w, color=C["orange"], label="compact 1D-CNN")
for i, v in enumerate(vals.values()):
    bx.text(i - w / 2, v[0] + 0.02, f"{v[0]:.2f}", ha="center", fontsize=11); bx.text(i + w / 2, v[1] + 0.02, f"{v[1]:.2f}", ha="center", fontsize=11)
bx.set_xticks(list(xs)); bx.set_xticklabels(list(vals), fontsize=12); bx.set_ylim(0, 1); bx.tick_params(axis="y", labelsize=10)
bx.spines[["top", "right"]].set_visible(False); bx.set_facecolor("none"); bx.legend(fontsize=10, frameon=False, loc="upper right")
ax.text(x0 + 3.4, 1.35, "DOC not recoverable\nPLS: about 1,450x fewer operations", ha="center", fontsize=11.5, color=C["ink"], linespacing=1.3)
arrow(13.6, 14.4)
panel(14.5, 6.0, "Raspberry Pi 4 node", [("4.2 ms per spectrum\n(complete PLS pipeline)", 12.5, C["ink"]), ("24 h over VPN with\nbroker restarts and outages:\n0 messages lost", 12.5, C["ink"]), ("about 1% of 4 CPU cores", 12.5, C["ink"])], C["green"], "#e9f8f1")
arrow(20.6, 21.3)
panel(21.4, 4.3, "Next: Raman", [("simulation, not\nvalidation", 11.5, C["grey"]), ("helps nitrite,\nnot nitrate", 12.5, C["ink"]), ("hardware target:\nSNR >= 100", 12.5, C["ink"])], C["orange"], "#fdf0ea", ls="--")
for e in ("png", "pdf", "tiff"):
    fig.savefig(OUT / f"graphical_abstract.{e}", dpi=200 if e != "pdf" else None)
print("graphical abstract written", OUT)
