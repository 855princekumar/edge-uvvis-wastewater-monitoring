#!/usr/bin/env python3
"""Fig. S10: robustness of the in-silico design study (simulation + semi-synthetic, not hardware).
(a) minimum Raman SNR giving >= 10% nitrite RMSE reduction across the Morris parameter sets;
(b) Morris mu* of the nitrite gain at SNR 50 (which assumptions drive it);
(c) nitrite gain vs SNR at fluorescence-to-signal 3: fully simulated vs real-matrix (semi-synthetic)."""
import json
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

C = ["#2a78d6", "#eb6834", "#1baf7a"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#888", "font.size": 8})
sens = json.load(open("robust_sensitivity.json")); semi = pd.read_csv("robust_realmatrix.csv")
sim = pd.read_csv("sim_summary.csv")
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
d = sens["summary"]["min_snr_no2_distribution"]; keys = ["10", "20", "50", "100", "300", "not_reached"]
ax[0].bar(range(len(keys)), [100 * d[k] for k in keys], color=C[0], edgecolor="white")
ax[0].set_xticks(range(len(keys)), ["10", "20", "50", "100", "300", ">300"]); ax[0].set_xlabel("Minimum Raman SNR for >= 10% nitrite gain")
ax[0].set_ylabel("Parameter sets (%)"); ax[0].set_title(f"(a) Sensitivity, {sens['n_samples']} parameter sets (rho = 3)", fontsize=9)
for i, k in enumerate(keys): ax[0].text(i, 100 * d[k] + 1, f"{100 * d[k]:.0f}", ha="center", fontsize=8)
mu = sens["indices"]["gain_no2_snr50"]; items = sorted(mu.items(), key=lambda kv: kv[1]["mu_star"])
NAMES = {"no2_raman_xsec": "Nitrite Raman cross-section", "no2_sigma_nm": "Nitrite UV band width", "unk_amp": "Unknown UV absorber", "eps_no2_peak": "Nitrite UV absorptivity", "rm_foul_loss": "Raman fouling loss", "uv_noise": "UV detector noise", "doc_slope_sd": "Organic-matter spectral variability", "rm_turb_att": "Raman turbidity attenuation", "lab_abs_err_no2": "Lab error, nitrite", "turb_rain": "Rain turbidity"}
ax[1].barh([NAMES[k] for k, _ in items], [v["mu_star"] for _, v in items], color=C[1], edgecolor="white")
ax[1].set_xlabel("Morris mu* of nitrite gain at SNR 50 (percentage points)"); ax[1].set_title("(b) Which assumptions matter", fontsize=9)
ax[1].tick_params(axis="y", labelsize=7)
s3 = sim[(sim.target == "no2") & (sim.rho == 3)].sort_values("snr")
r3 = semi[(semi.kind == "fused") & (semi.target == "no2") & (semi.rho == 3)].groupby("snr").gain_pct.agg(["mean", "std"]).reset_index()
ax[2].errorbar(s3.snr, s3.gain, yerr=s3.gain_sd, color=C[0], marker="o", lw=2, capsize=2, label="fully simulated")
ax[2].errorbar(r3.snr, r3["mean"], yerr=r3["std"], color=C[2], marker="s", lw=2, capsize=2, label="real Eawag matrix + added bands")
ax[2].axhline(10, color="#888", ls="--", lw=1); ax[2].text(3.2, 12, "10% criterion", fontsize=7, color="#555")
ax[2].set_xscale("log"); ax[2].set_xlabel("Raman SNR at 10 mg N/L nitrate"); ax[2].set_ylabel("Nitrite RMSE reduction (%)")
ax[2].set_title("(c) Fully simulated vs real-matrix UV-Vis (rho = 3)", fontsize=9); ax[2].legend(fontsize=7, frameon=False)
for f in ("fig_sim_robustness.png", "fig_sim_robustness.svg"): fig.savefig(f, dpi=200)
print("written")
