#!/usr/bin/env python3
"""Robustness check R2: semi-synthetic test on REAL spectra. Nitrate and nitrite absorption
bands (Beer-Lambert, absorptivities and band shapes of sim_fusion.py) are added to the 510
real Eawag raw-wastewater UV-Vis spectra (abs/m) in time order, so the real matrix, noise and
interferences replace the simulated UV background. Concentration series and laboratory error
are those of the simulator (nitrifying-effluent levels). Then, under time-blocked CV:
  (a) UV-Vis only: can nitrate and nitrite be recovered from real-matrix spectra?
  (b) hybrid: real-matrix UV-Vis fused with SIMULATED Raman (same concentrations) at
      SNR 20..300 and fluorescence ratios 1, 3, 10: does the nitrite gain survive?
Assumptions: added bands are linear (no extra detector saturation), which favours UV-Vis;
raw sewage is a harsher background than effluent. Not a validation of Raman.
Usage: python robust_realmatrix.py <real_spectra.npy>  -> robust_realmatrix.json/.csv"""
import json, sys, time
import numpy as np, pandas as pd
import sim_fusion as S

p = S.PARAMS
Xreal = np.load(sys.argv[1])                       # (510, 215) abs/m, time-ordered (export_real.py)
wl = np.load(sys.argv[1].replace(".npy", "_wl.npy")); assert len(wl) == Xreal.shape[1] and len(set(wl)) == len(wl)
eps3 = p["eps_no3_peak"] * np.exp(-0.5 * ((wl - p["no3_peak_nm"]) / p["no3_sigma_nm"]) ** 2)
eps2 = p["eps_no2_peak"] * np.exp(-0.5 * ((wl - p["no2_peak_nm"]) / p["no2_sigma_nm"]) ** 2)
rows = []; t0 = time.time()
for seed in (1, 2, 3, 4, 5):
    rng = np.random.default_rng(seed); s = S.timeseries(rng, p)
    y = {t: s[t] * (1 + rng.normal(0, p["lab_rel_err"], len(s[t]))) + rng.normal(0, p["lab_abs_err"][t], len(s[t])) for t in ("no2", "no3")}
    add = (eps3[None] * (s["no3"][:, None] / 14000.0) + eps2[None] * (s["no2"][:, None] / 14000.0)) * 100.0   # 1/cm -> 1/m
    Xuv = S.prep_uv(Xreal + add)
    _, Asim = S.uv_spectra(rng, s, p); Xsim = S.prep_uv(Asim)                                          # fully simulated reference
    base = {t: S.cv_rmse(Xuv, y[t], True, seed) for t in ("no2", "no3")}
    simb = {t: S.cv_rmse(Xsim, y[t], True, seed) for t in ("no2", "no3")}
    for t in ("no2", "no3"):
        rows.append(dict(seed=seed, kind="uv_only", target=t, snr=None, rho=None, r2_real_matrix=base[t][1], rmse_real_matrix=base[t][0],
                         r2_fully_simulated=simb[t][1]))
    for snr in (20, 50, 100, 300):
        for rho in (1, 3, 10):
            wn, R = S.raman_spectra(np.random.default_rng(seed * 1000 + snr * 7 + rho), s, p, snr, rho)
            Xr = S.prep_rm(R, wn)
            for t in ("no2", "no3"):
                f = S.cv_rmse(np.hstack([Xuv, Xr]), y[t], True, seed, blocks=[Xuv.shape[1], Xr.shape[1]])
                rows.append(dict(seed=seed, kind="fused", target=t, snr=snr, rho=rho, r2_real_matrix=f[1], rmse_real_matrix=f[0],
                                 gain_pct=100 * (base[t][0] - f[0]) / base[t][0]))
    print("seed", seed, f"{time.time() - t0:.0f}s", flush=True)
D = pd.DataFrame(rows); D.to_csv("robust_realmatrix.csv", index=False)
u = D[D.kind == "uv_only"].groupby("target")[["r2_real_matrix", "r2_fully_simulated"]].agg(["mean", "std"])
g = D[D.kind == "fused"].groupby(["target", "snr", "rho"]).gain_pct.agg(["mean", "std"]).reset_index()
res = {"uv_only": {t: {"r2_real_matrix_mean": float(u.loc[t, ("r2_real_matrix", "mean")]), "r2_real_matrix_sd": float(u.loc[t, ("r2_real_matrix", "std")]),
                       "r2_fully_simulated_mean": float(u.loc[t, ("r2_fully_simulated", "mean")])} for t in ("no2", "no3")},
       "fused_gain": g.to_dict(orient="records"), "runtime_s": time.time() - t0, "n_spectra": int(Xreal.shape[0])}
json.dump(res, open("robust_realmatrix.json", "w"), indent=1)
print(json.dumps(res["uv_only"], indent=1)); print(g.to_string(index=False))
