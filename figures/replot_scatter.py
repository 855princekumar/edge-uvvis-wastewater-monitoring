#!/usr/bin/env python3
"""Fig. 2 and Figs. S1-S6: predicted versus measured on the 75/25 hold-out split, re-plotted with units from the
predictions that run_experiment.py stores in results/<batch>/dashboard.html (same values as pred_vs_measured.png).
Usage (repo root): python figures/replot_scatter.py figures/out"""
import json, re, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "figures/out"); OUT.mkdir(parents=True, exist_ok=True)
NAME = {"lab_nh4_mg_l": ("NH$_4^+$", "s"), "lab_po4_mg_l": ("PO$_4^{3-}$", "s"), "lab_doc_mg_l": ("DOC", "s")}
COL = {"PLS": "#2a78d6", "CNN": "#eb6834"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
for batch, tag in (("batch-7_smoothsnv_215ch", "s"), ("batch-6_raw_215ch", "r")):
    html = Path(f"results/{batch}/dashboard.html").read_text(encoding="utf-8")
    D = json.loads(re.search(r'(?:const|var|let)\s+D\s*=\s*(\{"stamp".*?\});', html, re.S).group(1))
    assert "REAL" in D["stamp"], D["stamp"]
    for r in D["runs"]:
        name = NAME[r["target"]][0]; y = r["test_y"]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharex=True, sharey=True)
        lo = min(min(y), *(min(v) for v in r["preds"].values())); hi = max(max(y), *(max(v) for v in r["preds"].values()))
        for ax, (k, yp) in zip(axes, r["preds"].items()):
            m = r["models"][k]; key = "PLS" if k.upper().startswith("PLS") else "CNN"
            ax.scatter(y, yp, s=14, alpha=.7, color=COL[key], edgecolor="none")
            ax.plot([lo, hi], [lo, hi], "--", color="#6b7280", lw=1)
            ax.set_title(f"{'PLS' if key == 'PLS' else '1D-CNN'}: R$^2$ = {m['R2']:.3f}, RMSE = {m['RMSE']:.2f} mg/L", fontsize=9.5)
            ax.set_xlabel(f"Measured {name} (mg/L)"); ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(f"Predicted {name} (mg/L)")
        fig.tight_layout()
        stem = f"{tag}_{r['target'].split('_')[1]}"
        for e in ("png", "pdf"): fig.savefig(OUT / f"scatter_{stem}.{e}", dpi=300)
        plt.close(fig); print("written", stem, {k: round(v["R2"], 3) for k, v in r["models"].items()})
