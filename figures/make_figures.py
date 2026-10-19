#!/usr/bin/env python3
"""Draws the manuscript schematics (Fig. 1 architecture, Fig. 2 complementarity) and the VIP figure\n(Fig. S7) as vector graphics. Fig. 1 and Fig. 2 are conceptual; Fig. 2 uses illustrative model curves,\nnot measured spectra. Fig. S7 uses results/validation_v131/baselines_forward.json.\nUsage (repo root): python figures/make_figures.py figures/out"""
import json, sys
from pathlib import Path
import numpy as np, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "figures/out"); OUT.mkdir(parents=True, exist_ok=True)
C = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "grey": "#6b7280", "ink": "#1f2937", "light": "#eef2f7"}
plt.rcParams.update({"font.size": 8.5, "font.family": "DejaVu Sans"})


def box(ax, x, y, w, h, text, fc="#ffffff", ec=C["ink"], ls="-", bold=False, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc, ec=ec, lw=1.2, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=C["ink"], weight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, text=None, color=C["ink"], ls="-", off=(0, 0.12)):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=color, ls=ls))
    if text: ax.text((x1 + x2) / 2 + off[0], (y1 + y2) / 2 + off[1], text, ha="center", va="bottom", fontsize=7.2, color=color)


# ---------------- Fig. 1: architecture ----------------
fig, ax = plt.subplots(figsize=(7.4, 4.1)); ax.set_xlim(0, 16); ax.set_ylim(0, 8.4); ax.axis("off")
fs = 7.6
box(ax, 0.1, 5.5, 2.5, 1.6, "STP inlet / outlet\n(wastewater)", fc=C["light"], fs=fs)
box(ax, 3.0, 5.5, 2.8, 1.6, "Automated\nsampling\n(solenoid valve)", fs=fs)
box(ax, 6.2, 5.5, 2.6, 1.6, "Inline\nfiltration", fs=fs)
box(ax, 9.2, 5.2, 3.0, 2.2, "Flow cell:\nUV-Vis spectrometer\n200-735 nm", fc="#e8f1fc", ec=C["blue"], fs=fs)
box(ax, 12.7, 5.5, 3.2, 1.6, "Raman probe\n(Stage 3, not\nimplemented)", fc="#fdf0ea", ec=C["orange"], ls="--", fs=fs)
box(ax, 4.75, 2.0, 3.55, 2.4, "Edge node (Raspberry Pi)\nSG + SNV preprocessing\nPLS inference, QA flags\nlocal MQTT buffer", fc="#e9f8f1", ec=C["green"], fs=fs)
box(ax, 9.2, 2.3, 3.0, 1.9, "Physico-chemical\nprobes (pH, EC, DO,\nturbidity, T)", fs=fs)
box(ax, 12.7, 2.3, 3.2, 1.9, "Distilled-water\nflush between\ncycles (to waste)", fs=fs)
box(ax, 0.1, 2.0, 3.55, 2.4, "Plant dashboard,\nSCADA / telemetry\n(results and\nsummaries only)", fc=C["light"], fs=fs)
box(ax, 0.1, 0.1, 3.55, 1.3, "Operator decision\n(human in the loop)", fs=fs)
arrow(ax, 2.6, 6.3, 3.0, 6.3); arrow(ax, 5.8, 6.3, 6.2, 6.3); arrow(ax, 8.8, 6.3, 9.2, 6.3)
arrow(ax, 12.2, 6.3, 12.7, 6.3, color=C["orange"], ls="--")
arrow(ax, 10.2, 5.2, 7.6, 4.4); ax.text(8.25, 4.95, "spectra", fontsize=7.2, color=C["ink"], ha="right")
arrow(ax, 13.9, 4.2, 11.7, 5.2); ax.text(13.05, 4.85, "flush", fontsize=7.2, color=C["ink"], ha="left")
arrow(ax, 9.2, 3.25, 8.3, 3.25); ax.text(8.75, 3.4, "data", fontsize=6.6, color=C["ink"], ha="center", va="bottom")
arrow(ax, 4.75, 3.2, 3.65, 3.2); ax.text(4.2, 3.35, "results", fontsize=6.6, color=C["ink"], ha="center", va="bottom")
arrow(ax, 1.85, 2.0, 1.85, 1.4); ax.text(2.05, 1.6, "alerts", fontsize=7.0, color=C["ink"], ha="left")
ax.text(8.0, 7.85, "Measurement (Sections 3.1, 3.4)", ha="center", fontsize=8.2, color=C["grey"])
ax.text(6.5, 1.55, "Edge processing (Sections 4.5, 5.4)", ha="center", fontsize=8.2, color=C["grey"])
fig.tight_layout(); [fig.savefig(OUT / f"fig1_architecture.{e}", dpi=300) for e in ("png", "svg", "pdf")]; plt.close(fig)

# ---------------- Fig. 2: complementarity (illustrative curves) ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.7))
wl = np.linspace(200, 450, 400)
org = 1.2 * np.exp(-0.015 * (wl - 254)); nit = 1.6 * np.exp(-0.5 * ((wl - 205) / 9) ** 2)
ax[0].plot(wl, org + nit, color=C["blue"], lw=2, label="total absorbance")
ax[0].plot(wl, org, color=C["grey"], lw=1.2, ls="--", label="organic matter (exponential)")
ax[0].plot(wl, nit, color=C["green"], lw=1.2, ls=":", label="nitrate band (~205 nm)")
ax[0].set_xlabel("Wavelength (nm)"); ax[0].set_ylabel("Absorbance (arbitrary units)"); ax[0].set_yticks([])
ax[0].set_title("(a) UV-Vis: broad, overlapping", fontsize=9); ax[0].legend(fontsize=6.8, frameon=False)
wn = np.linspace(800, 1800, 800); lor = lambda c, w, a: a / (1 + ((wn - c) / w) ** 2)
fl = 0.5 + 0.0004 * (wn - 800)
for c, w, a, lab in ((981, 6, 0.8, "SO$_4^{2-}$ 981"), (1047, 6, 1.0, "NO$_3^-$ 1047"), (1330, 8, 0.45, "NO$_2^-$ 1330"), (1640, 45, 0.35, "H$_2$O 1640")):
    dx = {981: -45, 1047: 45}.get(c, 0)
    ax[1].text(c + dx, fl[np.argmin(abs(wn - c))] + a + 0.08, lab, ha="center", fontsize=6.6, color=C["ink"])
spec = fl + lor(981, 6, 0.8) + lor(1047, 6, 1.0) + lor(1330, 8, 0.45) + lor(1640, 45, 0.35)
ax[1].plot(wn, spec, color=C["orange"], lw=1.6, label="Raman signal"); ax[1].plot(wn, fl, color=C["grey"], lw=1.2, ls="--", label="fluorescence background")
ax[1].set_xlabel("Raman shift (cm$^{-1}$)"); ax[1].set_ylabel("Intensity (arbitrary units)"); ax[1].set_yticks([]); ax[1].set_ylim(0, 2.3)
ax[1].set_title("(b) Raman: narrow, molecule-specific", fontsize=9); ax[1].legend(fontsize=6.8, frameon=False, loc="upper right")
for a in ax: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); [fig.savefig(OUT / f"fig2_complementarity.{e}", dpi=300) for e in ("png", "svg", "pdf")]; plt.close(fig)

# ---------------- Fig. S7: VIP / coefficients ----------------
V = json.load(open("results/validation_v131/baselines_forward.json"))["vip"]
fig, ax = plt.subplots(2, 2, figsize=(7.2, 3.9), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
for k, (t, lab) in enumerate((("nh4", "Ammonium (NH$_4^+$), 12 LV"), ("po4", "Phosphate (PO$_4^{3-}$), 14 LV"))):
    w = np.array(V[t]["wl"]); v = np.array(V[t]["vip"]); sp = np.array(V[t]["mean_raw_spectrum"])
    ax[0, k].plot(w, v, color=C["blue"], lw=1.6); ax[0, k].axhline(1, color=C["grey"], ls="--", lw=1)
    ax[0, k].text(w.max(), 1.03, "VIP = 1", ha="right", va="bottom", fontsize=7, color=C["grey"])
    ax[0, k].set_title(lab, fontsize=9); ax[0, k].set_ylabel("VIP score")
    ax[1, k].plot(w, sp, color=C["ink"], lw=1.2); ax[1, k].set_ylabel("Mean raw\nabsorbance (m$^{-1}$)", fontsize=7.5)
    ax[1, k].set_xlabel("Wavelength (nm)")
for a in ax.flat: a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); [fig.savefig(OUT / f"fig_s7_vip.{e}", dpi=300) for e in ("png", "svg", "pdf")]; plt.close(fig)
print("figures written to", OUT)
