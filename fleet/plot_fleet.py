#!/usr/bin/env python3
"""Fig. S8: end-to-end latency CDF per profile (fleet run). Fig. S7: inference ms per profile
(microbenchmark, log scale), PLS vs NumPy 1D-CNN. All values EMULATED (QEMU + Docker cgroups)
except the native amd64 reference.  Usage: python plot_fleet.py <runs/fleet_run or -> out_dir"""
COL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
LS = {"linux/amd64": ":", "linux/arm64": "-", "linux/arm/v7": "--"}
import json, sys, glob, os
import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888", "font.size": 8})
run, dst = sys.argv[1], sys.argv[2]; os.makedirs(dst, exist_ok=True)
prof = {k: v for k, v in json.load(open("profiles.json")).items() if not k.startswith("_")}
if run != "-":
  R = pd.read_json(f"{run}/results.jsonl", lines=True).drop_duplicates(["node", "seq"])
  R["e2e_ms"] = (R.t_coll - R.t_send) * 1e3
  fig, ax = plt.subplots(figsize=(6.5, 5))
  for i, n in enumerate(prof):
    g = np.sort(R.loc[R.node == n, "e2e_ms"].to_numpy())
    if len(g): ax.plot(g, np.arange(1, len(g) + 1) / len(g), color=COL[i], ls=LS[prof[n]["platform"]], lw=2, label=f"{n} ({prof[n]['platform'].split('/',1)[1]}, {prof[n]['mem']}, {prof[n]['cpus']} cpu)")
  if len(sys.argv) > 3:   # optional: physical node(s) from a second run, e.g. a real Pi over VPN
    R2 = pd.read_json(f"{sys.argv[3]}/results.jsonl", lines=True).drop_duplicates(["node", "seq"])
    R2["e2e_ms"] = (R2.t_coll - R2.t_send) * 1e3
    for n in sorted(set(R2.node) - set(prof)):
      g = np.sort(R2.loc[R2.node == n, "e2e_ms"].to_numpy())
      ax.plot(g, np.arange(1, len(g) + 1) / len(g), color="#222222", lw=2.4, label=f"{n} (physical Raspberry Pi 4, 4 GB, over VPN; separate 10-min run)")
  ax.set_xscale("log");   ax.set_xlabel("End-to-end latency, spectrum published to result collected (ms)"); ax.set_ylabel("Cumulative fraction")
  ax.set_title("Fleet end-to-end latency, 1 Hz per node (coloured: emulated, QEMU + cgroup limits)", fontsize=9); ax.grid(color="#ddd", lw=.5, which="major"); ax.legend(fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
  fig.tight_layout(); fig.savefig(f"{dst}/fig_s8_latency_cdf.png", dpi=200); fig.savefig(f"{dst}/fig_s8_latency_cdf.svg")
rows = []
if os.path.exists("real_pi/bench_pi4_real_1thread.json"):
    d = json.load(open("real_pi/bench_pi4_real_1thread.json")); rows.append(("pi4_real (physical)", d["PLS"]["median_ms"], d["CNN1D_numpy"]["median_ms"], d["PLS"]["p95_ms"], d["CNN1D_numpy"]["p95_ms"]))
for n in prof:
    f = f"out_bench/bench_{n}.json"
    if os.path.exists(f):
        d = json.load(open(f)); rows.append((n, d["PLS"]["median_ms"], d["CNN1D_numpy"]["median_ms"], d["PLS"]["p95_ms"], d["CNN1D_numpy"]["p95_ms"]))
B = pd.DataFrame(rows, columns=["node", "pls", "cnn", "pls95", "cnn95"])
fig, ax = plt.subplots(figsize=(6.5, 4)); x = np.arange(len(B)); w = .38
hat = ["//" if "physical" in n else None for n in B.node]
b1 = ax.bar(x - w/2, B.pls, w, label="PLS (sklearn predict)", color=COL[0], edgecolor="white", linewidth=1); b2 = ax.bar(x + w/2, B.cnn, w, label="1D-CNN (NumPy)", color=COL[1], edgecolor="white", linewidth=1)
for bars in (b1, b2):
    for r, h in zip(bars, hat):
        if h: r.set_hatch(h)
ax.errorbar(x - w/2, B.pls, yerr=[np.zeros(len(B)), B.pls95 - B.pls], fmt="none", ecolor="k", capsize=2, lw=.8)
ax.errorbar(x + w/2, B.cnn, yerr=[np.zeros(len(B)), B.cnn95 - B.cnn], fmt="none", ecolor="k", capsize=2, lw=.8)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(B.node, rotation=30, ha="right", fontsize=7); ax.set_ylabel("Single-spectrum inference (ms), median; whisker = p95")
ax.set_title("Inference by profile: hatched = physical Raspberry Pi 4; node_ref_amd64 = native; others emulated", fontsize=8); from matplotlib.patches import Patch
ax.grid(color="#ddd", lw=.5, axis="y"); ax.legend(handles=[Patch(color=COL[0], label="PLS (sklearn predict)"), Patch(color=COL[1], label="1D-CNN (NumPy)")], fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{dst}/fig_s7_inference_by_profile.png", dpi=200); fig.savefig(f"{dst}/fig_s7_inference_by_profile.svg"); print("plots written")
