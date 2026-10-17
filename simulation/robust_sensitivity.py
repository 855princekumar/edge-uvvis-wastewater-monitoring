#!/usr/bin/env python3
"""Robustness check R1: global sensitivity analysis (Morris elementary effects, SALib) of the
in-silico UV-Vis + Raman design study (sim_fusion.py). Ten modelling assumptions are varied
jointly over plausible ranges; for every parameter set we compute, under time-blocked CV:
  - the nitrite RMSE reduction of UV-Vis + Raman vs UV-Vis alone at SNR 10..300 (rho = 3),
  - the minimum Raman SNR giving >= 10% reduction (the Stage 3 entry criterion),
  - UV-only nitrite R2 and the nitrate gain at SNR 300 (rho = 3).
Simulation only; it quantifies how much the design conclusions depend on the assumptions.
Usage: python robust_sensitivity.py [trajectories]  -> robust_sensitivity.json / .csv"""
import copy, json, sys, time
import numpy as np, pandas as pd
from SALib.sample import morris as ms
from SALib.analyze import morris as ma
import sim_fusion as S

PROBLEM = {"num_vars": 10, "names": [
    "eps_no2_peak", "no2_sigma_nm", "no2_raman_xsec", "doc_slope_sd", "unk_amp",
    "uv_noise", "turb_rain", "rm_turb_att", "rm_foul_loss", "lab_abs_err_no2"],
    "bounds": [[3500, 7500], [8, 16], [0.3, 1.2], [0.001, 0.004], [0.025, 0.10],
               [0.001, 0.004], [7.5, 30.0], [0.01, 0.04], [0.10, 0.40], [0.015, 0.06]]}
SNRS = (10, 20, 50, 100, 300); RHO = 3


def params_from(x):
    p = copy.deepcopy(S.PARAMS)
    d = dict(zip(PROBLEM["names"], x))
    for k in ("eps_no2_peak", "no2_sigma_nm", "doc_slope_sd", "unk_amp", "uv_noise", "turb_rain", "rm_turb_att", "rm_foul_loss"):
        p[k] = float(d[k])
    c, w, _ = p["bands"]["no2"]; p["bands"]["no2"] = (c, w, float(d["no2_raman_xsec"]))
    p["lab_abs_err"] = dict(p["lab_abs_err"], no2=float(d["lab_abs_err_no2"]))
    return p


def evaluate(x, seed):
    p = params_from(x)
    rng = np.random.default_rng(seed); s = S.timeseries(rng, p)
    y = {t: s[t] * (1 + rng.normal(0, p["lab_rel_err"], len(s[t]))) + rng.normal(0, p["lab_abs_err"][t], len(s[t])) for t in ("no2", "no3")}
    _, A = S.uv_spectra(rng, s, p); Xuv = S.prep_uv(A)
    uv2 = S.cv_rmse(Xuv, y["no2"], True, seed); uv3 = S.cv_rmse(Xuv, y["no3"], True, seed)
    out = {"uv_r2_no2": uv2[1], "uv_r2_no3": uv3[1]}
    for snr in SNRS:
        wn, R = S.raman_spectra(np.random.default_rng(seed * 1000 + snr * 7 + RHO), s, p, snr, RHO)
        Xf = np.hstack([Xuv, S.prep_rm(R, wn, p)]); blocks = [Xuv.shape[1], Xf.shape[1] - Xuv.shape[1]]
        f2 = S.cv_rmse(Xf, y["no2"], True, seed, blocks=blocks)
        out[f"gain_no2_snr{snr}"] = 100 * (uv2[0] - f2[0]) / uv2[0]
        if snr == 300:
            f3 = S.cv_rmse(Xf, y["no3"], True, seed, blocks=blocks)
            out["gain_no3_snr300"] = 100 * (uv3[0] - f3[0]) / uv3[0]
    ok = [snr for snr in SNRS if out[f"gain_no2_snr{snr}"] >= 10]
    out["min_snr_no2"] = min(ok) if ok else np.nan
    return out


if __name__ == "__main__":
    r = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    X = ms.sample(PROBLEM, N=r, num_levels=4, seed=42)
    k = PROBLEM["num_vars"]; t0 = time.time(); rows = []
    for i, x in enumerate(X):
        seed = 100 + i // (k + 1)                     # common random numbers within each trajectory
        rows.append(dict(zip(PROBLEM["names"], x), seed=seed, **evaluate(x, seed)))
        if i % 22 == 0: print(f"{i}/{len(X)} {time.time() - t0:.0f}s", flush=True)
    D = pd.DataFrame(rows); D.to_csv("robust_sensitivity_samples.csv", index=False)
    res = {"n_samples": len(D), "trajectories": r, "problem": PROBLEM, "rho": RHO, "snrs": SNRS, "indices": {}}
    for out in ("gain_no2_snr50", "uv_r2_no2", "gain_no3_snr300"):
        Si = ma.analyze(PROBLEM, X, D[out].to_numpy(), num_levels=4, seed=42)
        res["indices"][out] = {n: {"mu_star": float(a), "sigma": float(b)} for n, a, b in zip(PROBLEM["names"], Si["mu_star"], Si["sigma"])}
    q = lambda v: {"p5": float(np.nanpercentile(v, 5)), "median": float(np.nanmedian(v)), "p95": float(np.nanpercentile(v, 95))}
    m = D["min_snr_no2"]
    res["summary"] = {
        "min_snr_no2_distribution": {str(s): float((m == s).mean()) for s in SNRS} | {"not_reached": float(m.isna().mean())},
        "frac_min_snr_le_50": float((m <= 50).mean()), "frac_min_snr_le_100": float((m <= 100).mean()),
        "gain_no2_snr50": q(D["gain_no2_snr50"]), "gain_no2_snr300": q(D["gain_no2_snr300"]),
        "uv_r2_no2": q(D["uv_r2_no2"]), "uv_r2_no3": q(D["uv_r2_no3"]),
        "gain_no3_snr300": q(D["gain_no3_snr300"]), "frac_no3_gain_ge_10": float((D["gain_no3_snr300"] >= 10).mean()),
        "runtime_s": time.time() - t0}
    json.dump(res, open("robust_sensitivity.json", "w"), indent=1)
    print(json.dumps(res["summary"], indent=1))
