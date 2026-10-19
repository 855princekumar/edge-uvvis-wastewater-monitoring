#!/usr/bin/env python3
"""Graphical abstract (journal requirement: readable at 13 cm width). Drawn at 13.2 cm x 5.8 cm with all text >= 7 pt.
Numbers are those of the paper: time-blocked R2 (results/batch-8_smoothsnv_blocked_215ch), Pi pipeline time
(fleet run B), 24 h test (fleet/runs/soak_24h_1hz), Raman threshold (simulation/). Usage: python figures/make_graphical_abstract.py figures/out"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "figures/out"); OUT.mkdir(parents=True, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "grey": "#6b7280", "ink": "#1f2937"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7})
W, H = 13.2 / 2.54, 5.8 / 2.54
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 26); ax.set_ylim(0, 11.4); ax.axis("off")


def panel(x, w, title, lines, ec, fc, ls="-"):
    ax.add_patch(FancyBboxPatch((x, 0.5), w, 8.9, boxstyle="round,pad=0.05,rounding_size=0.3", fc=fc, ec=ec, lw=1.2, ls=ls))
    ax.text(x + w / 2, 8.55, title, ha="center", va="center", fontsize=8, weight="bold", color=C["ink"])
    for y, t, col in lines:
        ax.text(x + w / 2, y, t, ha="center", va="center", fontsize=7.5, color=col, linespacing=1.25)


def arrow(x1, x2):
    ax.add_patch(FancyArrowPatch((x1, 5.0), (x2, 5.0), arrowstyle="-|>", mutation_scale=9, lw=1.3, color=C["ink"]))


ax.text(13, 10.5, "Edge UV-Vis wastewater monitoring: real data, real edge hardware", ha="center", va="center", fontsize=9, weight="bold", color=C["ink"])
panel(0.2, 5.2, "Real data", [(6.4, "Raw municipal\nwastewater,\n25 weeks", C["ink"]), (3.0, "510 spectra\nwith lab\nreferences", C["ink"])], C["grey"], "#f3f4f6")
arrow(5.5, 6.3)
x0 = 6.4; panel(x0, 7.0, "Time-blocked R$^2$", [(1.15, "DOC not recoverable", C["ink"])], C["blue"], "#eaf2fc")
bx = fig.add_axes([(x0 + 0.9) / 26, 3.0 / 11.4, 5.6 / 26, 4.6 / 11.4])
vals = {"NH$_4^+$": (0.821, 0.596), "PO$_4^{3-}$": (0.564, 0.352)}; w = 0.38
for i, (k, (p, c)) in enumerate(vals.items()):
    bx.bar(i - w / 2, p, w, color=C["blue"], label="PLS" if i == 0 else None); bx.bar(i + w / 2, c, w, color=C["orange"], label="1D-CNN" if i == 0 else None)
    bx.text(i - w / 2, p + 0.03, f"{p:.2f}", ha="center", fontsize=7); bx.text(i + w / 2, c + 0.03, f"{c:.2f}", ha="center", fontsize=7)
bx.set_xticks([0, 1]); bx.set_xticklabels(list(vals), fontsize=7.5); bx.set_ylim(0, 1.15); bx.set_yticks([])
bx.spines[["top", "right", "left"]].set_visible(False); bx.set_facecolor("none")
bx.legend(fontsize=7, frameon=False, loc="upper center", ncol=2, handlelength=1, columnspacing=0.8, borderaxespad=0)
arrow(13.5, 14.3)
panel(14.4, 5.6, "Raspberry Pi 4", [(6.4, "4.2 ms per\nspectrum (PLS)", C["ink"]), (3.0, "24 h with outages:\n0 messages lost", C["ink"])], C["green"], "#e9f8f1")
arrow(20.1, 20.8)
panel(20.9, 4.9, "Next: Raman", [(6.2, "Simulation:\nhelps nitrite,\nnot nitrate", C["ink"]), (2.9, "Target:\nSNR ≥ 100", C["ink"])], C["orange"], "#fdf0ea", ls="--")
for e in ("png", "tiff"): fig.savefig(OUT / f"graphical_abstract.{e}", dpi=600)
fig.savefig(OUT / "graphical_abstract.pdf")
print("graphical abstract written", OUT, "size cm", round(W * 2.54, 1), "x", round(H * 2.54, 1))
