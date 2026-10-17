import json, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, ".")
import sim_fusion as S

res = json.load(open("sim_results.json"))
g = pd.DataFrame(res["grid"])
b = g[g.blocked]
agg = b.groupby(["target", "snr", "rho"]).agg(
    uv=("uv_rmse", "mean"), rm=("rm_rmse", "mean"), fu=("fu_rmse", "mean"),
    uv_r2=("uv_r2", "mean"), fu_r2=("fu_r2", "mean"), rm_r2=("rm_r2", "mean"),
    gain=("gain_pct", "mean"), gain_sd=("gain_pct", "std")).reset_index()
agg.to_csv("sim_summary.csv", index=False)

names = {"no3": "Nitrate (NO$_3^-$-N)", "no2": "Nitrite (NO$_2^-$-N)", "doc": "DOC"}
snrs = sorted(b.snr.unique()); rhos = sorted(b.rho.unique())
fig, ax = plt.subplots(1, 3, figsize=(13, 3.9), constrained_layout=True)
for k, t in enumerate(("no2", "no3", "doc")):
    M = np.array([[agg[(agg.target == t) & (agg.snr == s) & (agg.rho == r)].gain.values[0] for s in snrs] for r in rhos])
    im = ax[k].imshow(M, origin="lower", cmap="RdBu", vmin=-70, vmax=70, aspect="auto")
    for i in range(len(rhos)):
        for j in range(len(snrs)):
            ax[k].text(j, i, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8,
                       color="k" if abs(M[i, j]) < 45 else "w")
    ax[k].set_xticks(range(len(snrs)), snrs); ax[k].set_yticks(range(len(rhos)), rhos)
    ax[k].set_xlabel("Raman SNR at 10 mg N/L nitrate"); ax[k].set_title(names[t], fontsize=11)
    if k == 0: ax[k].set_ylabel("Fluorescence-to-signal ratio")
fig.colorbar(im, ax=ax, label="RMSE reduction of UV-Vis + Raman\nvs UV-Vis alone (%), blocked CV")
fig.savefig("fig_sim_gain.png", dpi=200)

# requirement table: lowest SNR with mean gain >= 10 % (acceptance criterion), per rho
rows = []
for t in ("no2", "no3", "doc"):
    for r in rhos:
        sub = agg[(agg.target == t) & (agg.rho == r)].sort_values("snr")
        ok = sub[sub.gain >= 10]
        rows.append(dict(target=t, rho=r, min_snr=int(ok.snr.iloc[0]) if len(ok) else None,
                         max_gain=round(sub.gain.max(), 1)))
req = pd.DataFrame(rows); req.to_csv("sim_requirements.csv", index=False)

# random vs blocked at reference point (SNR 20, rho 10)
ref = g[(g.snr == 20) & (g.rho == 10)].groupby(["target", "blocked"]).agg(
    uv_r2=("uv_r2", "mean"), fu_r2=("fu_r2", "mean"), uv=("uv_rmse", "mean"), fu=("fu_rmse", "mean")).reset_index()
ref.to_csv("sim_ref_random_vs_blocked.csv", index=False)

# baseline UV-only performance (identical across grid) and lab floor
base = b.groupby(["target"]).agg(uv=("uv_rmse", "mean"), uv_sd=("uv_rmse", "std"), uv_r2=("uv_r2", "mean")).reset_index()
base.to_csv("sim_uv_baseline.csv", index=False)

# ---------------- drift detection demo (hourly spectra) ----------------
p = dict(S.PARAMS); p["n_samples"] = p["weeks"] * 7 * 24
rng = np.random.default_rng(7); s = S.timeseries(rng, p)
wl, A = S.uv_spectra(rng, s, p)
a700 = A[:, np.argmin(abs(wl - 700))]
t = s["t"]; cyc = (t // p["cleaning_days"]).astype(int)
roll = pd.Series(a700).groupby(cyc).transform(
    lambda x: x.rolling(24, min_periods=6).median()).values    # trailing 24 h median, within cycle
leads, flags = [], 0
for c in np.unique(cyc)[:-1]:
    m = cyc == c
    tt, rr = t[m], roll[m]
    ref0 = np.nanmedian(rr[(tt - tt[0] >= 1.0) & (tt - tt[0] < 2.0)])   # day 2 after cleaning (first full 24 h window)
    hit = np.where(rr - ref0 > 0.03)[0]                        # 0.03 AU alert threshold
    if len(hit):
        flags += 1; leads.append(p["cleaning_days"] - (tt[hit[0]] - c * p["cleaning_days"]))
drift = dict(cycles=int(len(np.unique(cyc)) - 1), flagged=flags,
             median_lead_days=float(np.median(leads)) if leads else None,
             corr_a700_foul=float(np.corrcoef(roll[~np.isnan(roll)], s["foul"][~np.isnan(roll)])[0, 1]))
json.dump(drift, open("sim_drift.json", "w"))
fig, ax = plt.subplots(figsize=(11, 3), constrained_layout=True)
ax.plot(t, a700, lw=0.3, color="0.7", label="Absorbance at 700 nm (hourly)")
ax.plot(t, roll, lw=1.2, color="C0", label="24 h rolling median")
for c in range(0, int(t.max()) + 1, p["cleaning_days"]):
    ax.axvline(c, color="C3", lw=0.6, ls="--")
ax.set_xlabel("Day of campaign"); ax.set_ylabel("A (AU)"); ax.set_ylim(0, np.nanpercentile(a700, 99.5))
ax.legend(loc="upper left", fontsize=8); ax.set_title("Simulated window fouling between cleanings (dashed)", fontsize=10)
fig.savefig("fig_sim_drift.png", dpi=200)

print(base.to_string()); print(req.to_string()); print(ref.to_string()); print(drift)
for t_ in ("no2", "no3", "doc"):
    sub = agg[(agg.target == t_) & (agg.snr == 100) & (agg.rho == 1)]
    print(t_, "SNR100 rho1:", sub[["uv", "rm", "fu", "uv_r2", "fu_r2", "gain", "gain_sd"]].round(3).to_dict("records"))
