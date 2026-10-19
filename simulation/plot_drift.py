"""Fig. S10: simulated fouling drift at 700 nm with the alarm rule drawn (threshold per cycle and first flag).
Recomputes exactly the drift demonstration of analyse.py (same seed and rule) and checks that the
flag count and median lead time match sim_drift.json. Usage (from simulation/): python plot_drift.py"""
import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
import sim_fusion as S

p = dict(S.PARAMS); p["n_samples"] = p["weeks"] * 7 * 24
rng = np.random.default_rng(7); s = S.timeseries(rng, p)
wl, A = S.uv_spectra(rng, s, p)
a700 = A[:, np.argmin(abs(wl - 700))]
t = s["t"]; cyc = (t // p["cleaning_days"]).astype(int)
roll = pd.Series(a700).groupby(cyc).transform(lambda x: x.rolling(24, min_periods=6).median()).values
fig, ax = plt.subplots(figsize=(11, 3.2), constrained_layout=True)
ax.plot(t, a700, lw=0.3, color="0.7", label="Absorbance at 700 nm (hourly)")
ax.plot(t, roll, lw=1.2, color="#2a78d6", label="24 h trailing median")
flags, leads = 0, []
for c in np.unique(cyc)[:-1]:
    m = cyc == c; tt, rr = t[m], roll[m]
    ref0 = np.nanmedian(rr[(tt - tt[0] >= 1.0) & (tt - tt[0] < 2.0)])
    ax.hlines(ref0 + 0.03, tt[0], tt[-1], color="#eb6834", lw=1, ls="--", label="alarm threshold (reference + 0.03 AU)" if c == 0 else None)
    hit = np.where(rr - ref0 > 0.03)[0]
    if len(hit):
        flags += 1; leads.append(p["cleaning_days"] - (tt[hit[0]] - c * p["cleaning_days"]))
        ax.plot(tt[hit[0]], rr[hit[0]], "v", color="#1f2937", ms=6, label="first alarm in cycle" if flags == 1 else None)
for c in range(0, int(t.max()) + 1, p["cleaning_days"]):
    ax.axvline(c, color="#6b7280", lw=0.6, ls=":")
ax.set_xlabel("Day of campaign (dotted lines: scheduled cleaning)"); ax.set_ylabel("A at 700 nm (AU)")
ax.set_ylim(0, 1.05 * np.nanmax(a700)); ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper left", fontsize=7.5, ncol=4, frameon=False)
ax.set_title("Simulated window fouling between fortnightly cleanings (simulation, not measured data)", fontsize=10)
fig.savefig("fig_sim_drift.png", dpi=200); fig.savefig("fig_sim_drift.svg")
ref = json.load(open("sim_drift.json"))
assert flags == ref["flagged"] and abs(np.median(leads) - ref["median_lead_days"]) < 1e-9, (flags, np.median(leads), ref)
print("drift figure ok:", flags, "of", ref["cycles"], "cycles flagged, median lead", round(float(np.median(leads)), 2), "d")
