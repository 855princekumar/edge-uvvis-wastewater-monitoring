#!/usr/bin/env python3
"""
In-silico design study: when does adding a Raman channel to UV-Vis improve
quantification in STP secondary effluent?  NOT a validation; every physical
parameter is an explicit, stated assumption (see PARAMS) and the result is a
requirement on Raman hardware (signal-to-noise, fluorescence tolerance).

Scenario: nitrifying STP secondary effluent, 25 weeks, 510 grab samples
(same n as the Eawag analysis), diurnal + slow stochastic variation, rain
events, probe fouling between fortnightly cleanings.
"""
import json, itertools, time
import numpy as np
from scipy.signal import savgol_filter
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PARAMS = dict(
    weeks=25, n_samples=510, cleaning_days=14,
    # concentrations (secondary effluent, assumed)
    no3n_base=8.0, no3n_diurnal=2.0, no3n_ar_sd=2.5,          # mg N/L
    doc_base=10.0, doc_diurnal=2.0, doc_ar_sd=2.5,            # mg C/L
    so4_range=(40, 120), hco3_range=(150, 400),               # mg/L
    turb_base=3.0, turb_rain=15.0,                             # NTU
    suva=(2.0, 4.0),                                           # L/(mg m)
    rain_events=10,
    # UV-Vis
    uv_wl=(200.0, 735.0, 2.5), path_cm=0.2,
    eps_no3_peak=9900.0, no3_peak_nm=201.0, no3_sigma_nm=9.0,  # L/(mol cm)
    doc_slope=0.015,                                           # 1/nm
    turb_k=0.01, turb_exp=1.2,                                 # 1/cm per NTU at 550 nm
    stray=1e-3, uv_noise=0.002,                                # fraction, AU
    uv_foul_max=0.10,                                          # AU offset at end of cycle
    # Raman (relative units, shift 800-1800 1/cm)
    rm_wn=(800.0, 1800.0, 2.0),
    bands={"no3": (1047, 6, 1.0), "so4": (981, 6, 0.8), "hco3": (1017, 8, 0.25),
           "no2": (1330, 8, 0.6), "water": (1640, 45, 0.0)},                         # centre, HWHM, rel. x-sec per mM
    rm_turb_att=0.02, rm_foul_loss=0.20,
    # non-idealities
    no2n_range=(0.05, 1.5), eps_no2_peak=5400.0, no2_peak_nm=210.0, no2_sigma_nm=12.0,
    doc_slope_sd=0.002, unk_amp=0.05,                          # AU, random extra UV band
    lab_rel_err=0.05, lab_abs_err={"no3": 0.2, "doc": 0.5, "no2": 0.03},    # grab-sample reference error
)
MW = {"no2": 14.0, "no3": 14.0, "so4": 96.06, "hco3": 61.02}


def timeseries(rng, p):
    days = p["weeks"] * 7
    t = np.sort(rng.uniform(0, days, p["n_samples"]))
    def ar(sd, tau=3.0):
        g = np.linspace(0, days, days * 24 + 1); x = np.zeros_like(g)
        a = np.exp(-1 / (24 * tau))
        for i in range(1, len(g)):
            x[i] = a * x[i - 1] + np.sqrt(1 - a * a) * rng.normal(0, sd)
        return np.interp(t, g, x)
    rain = np.zeros_like(t)
    for c in rng.uniform(0, days, p["rain_events"]):
        rain += np.exp(-np.clip(t - c, 0, None) / 0.8) * (t >= c)
    di = np.sin(2 * np.pi * (t % 1) - 1.0)
    no3 = np.clip(p["no3n_base"] + p["no3n_diurnal"] * di + ar(p["no3n_ar_sd"]) - 4 * rain, 0.3, None)
    doc = np.clip(p["doc_base"] + p["doc_diurnal"] * di + ar(p["doc_ar_sd"]) + 8 * rain, 2, None)
    turb = p["turb_base"] * np.exp(ar(0.4)) + p["turb_rain"] * rain
    so4 = np.clip(80 + ar(18), *p["so4_range"]); hco3 = np.clip(275 + ar(60), *p["hco3_range"])
    suva = np.clip(3.0 + ar(0.4), *p["suva"])
    no2 = np.clip(0.4 + ar(0.35) + 0.5 * rain, *p["no2n_range"])
    foul = (t % p["cleaning_days"]) / p["cleaning_days"]          # 0 after cleaning -> 1
    return dict(t=t, no2=no2, no3=no3, doc=doc, turb=turb, so4=so4, hco3=hco3, suva=suva, foul=foul)


def uv_spectra(rng, s, p):
    wl = np.arange(*p["uv_wl"][:2], p["uv_wl"][2]); wl = np.append(wl, p["uv_wl"][1])
    L = p["path_cm"]
    eps = p["eps_no3_peak"] * np.exp(-0.5 * ((wl - p["no3_peak_nm"]) / p["no3_sigma_nm"]) ** 2)
    eps2 = p["eps_no2_peak"] * np.exp(-0.5 * ((wl - p["no2_peak_nm"]) / p["no2_sigma_nm"]) ** 2)
    n = len(s["t"])
    slope = p["doc_slope"] + rng.normal(0, p["doc_slope_sd"], n)
    uc = rng.uniform(210, 290, n); uw = rng.uniform(8, 25, n); ua = rng.exponential(p["unk_amp"], n)
    A = (eps[None] * (s["no3"][:, None] / 14000.0) + eps2[None] * (s["no2"][:, None] / 14000.0)
         + (s["suva"] * s["doc"] / 100.0)[:, None] * np.exp(-slope[:, None] * (wl[None] - 254))
         + (ua / L)[:, None] * np.exp(-0.5 * ((wl[None] - uc[:, None]) / uw[:, None]) ** 2)
         + (p["turb_k"] * s["turb"])[:, None] * (wl[None] / 550.0) ** (-p["turb_exp"])) * L
    A += (p["uv_foul_max"] * s["foul"])[:, None]                 # window fouling offset
    T = 10 ** (-A) + p["stray"]                                  # stray light -> saturation
    Am = -np.log10(T / (1 + p["stray"])) + rng.normal(0, p["uv_noise"], A.shape)
    return wl, Am


def raman_spectra(rng, s, p, snr10, rho):
    wn = np.arange(p["rm_wn"][0], p["rm_wn"][1] + 1e-9, p["rm_wn"][2])
    lor = lambda c, w: 1 / (1 + ((wn - c) / w) ** 2)
    npk = snr10 ** 2                       # nitrate peak counts at 10 mg N/L (shot-noise SNR = snr10)
    k = npk / (10 / 14.0)                  # counts per mM nitrate
    att = np.exp(-p["rm_turb_att"] * s["turb"]) * (1 - p["rm_foul_loss"] * s["foul"])
    sig = np.zeros((len(s["t"]), len(wn)))
    for sp in ("no3", "no2", "so4", "hco3"):
        c, w, x = p["bands"][sp]
        mm = s[sp] / MW[sp]
        sig += (k * x * mm)[:, None] * lor(c, w)[None]
    sig += 0.5 * npk * lor(*p["bands"]["water"][:2])[None]
    # fluorescence: broad, proportional to DOC, random slope; rho = background/nitrate peak at DOC 10, NO3-N 10
    slope = rng.uniform(-0.5, 0.5, len(s["t"]))[:, None]
    shape = 1 + slope * (wn[None] - 1300) / 500
    fl = (rho * npk * s["doc"] / 10.0)[:, None] * shape
    tot = (sig + fl) * att[:, None]
    return wn, tot + rng.normal(0, 1, tot.shape) * np.sqrt(np.clip(tot, 1, None))


def prep_uv(A):  return savgol_filter(A, 11, 2, axis=1)
def prep_rm(R, wn, p=None):
    """Raman features by classical least-squares (CLS) unmixing: each spectrum is fitted
    with the known band shapes (NO3-, NO2-, SO4 2-, HCO3-, water bend) plus a cubic
    polynomial fluorescence baseline. Band coefficients are normalised to the water
    coefficient (internal standard); the log baseline-to-water ratio is kept as a
    fluorescence feature."""
    p = p or PARAMS
    lor = lambda c, w: 1 / (1 + ((wn - c) / w) ** 2)
    z = (wn - wn.mean()) / (wn.max() - wn.min())
    cols = [lor(*p["bands"][k][:2]) for k in ("no3", "no2", "so4", "hco3", "water")]
    cols += [z ** k for k in range(4)]
    M = np.column_stack(cols)
    coef = np.linalg.lstsq(M, R.T, rcond=None)[0].T            # n x 9
    wat = np.clip(coef[:, 4], 0.25 * np.median(coef[:, 4]), None)   # robust internal standard
    bands = coef[:, :4] / wat[:, None]
    fl = np.log(np.clip(coef[:, 5] / wat, 1e-6, None))
    return np.column_stack([bands, fl])


def block_concat(*Xs):
    return np.hstack([x / np.sqrt(x.shape[1]) for x in Xs])


def cv_rmse(X, y, blocked, seed, blocks=None):
    """PLS, 5-fold outer CV (random or contiguous time blocks), inner CV for LV count.
    blocks: list of column counts; each block is autoscaled then weighted 1/sqrt(size)."""
    kf = KFold(5, shuffle=not blocked, random_state=None if blocked else seed)
    se = np.zeros_like(y)
    w = None
    if blocks:
        w = np.concatenate([np.full(b, 1 / np.sqrt(b)) for b in blocks])
    for tr, va in kf.split(X):
        sc = StandardScaler().fit(X[tr]); Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        if w is not None: Xtr, Xva = Xtr * w, Xva * w
        best, bnc = 1e18, 1
        ik = KFold(3, shuffle=not blocked, random_state=None if blocked else seed)
        for nc in [n for n in (1, 2, 3, 4, 6, 8, 10, 12, 15) if n <= X.shape[1]]:
            e = []
            for a, b in ik.split(Xtr):
                m = PLSRegression(nc, scale=False).fit(Xtr[a], y[tr][a])
                e.append(np.mean((m.predict(Xtr[b]).ravel() - y[tr][b]) ** 2))
            if np.mean(e) < best: best, bnc = np.mean(e), nc
        m = PLSRegression(bnc, scale=False).fit(Xtr, y[tr])
        se[va] = (m.predict(Xva).ravel() - y[va]) ** 2
    rmse = float(np.sqrt(se.mean())); r2 = 1 - se.sum() / ((y - y.mean()) ** 2).sum()
    return rmse, float(r2)


def run(seeds=(1, 2, 3, 4, 5), snrs=(3, 5, 10, 20, 50, 100, 300), rhos=(1, 3, 10, 30, 100)):
    p = PARAMS; out = {"params": p, "grid": [], "drift": None}
    for seed in seeds:
        rng = np.random.default_rng(seed)
        s = timeseries(rng, p)
        y = {tg: s[tg] * (1 + rng.normal(0, p["lab_rel_err"], len(s[tg]))) + rng.normal(0, p["lab_abs_err"][tg], len(s[tg]))
             for tg in ("no3", "doc", "no2")}      # lab reference values (spectra use true s)
        _, A = uv_spectra(rng, s, p); Xuv = prep_uv(A)
        base = {}
        for tgt in ("no3", "doc", "no2"):
            for blk in (False, True):
                base[(tgt, blk)] = cv_rmse(Xuv, y[tgt], blk, seed)
        for snr, rho in itertools.product(snrs, rhos):
            wn, R = raman_spectra(np.random.default_rng(seed * 1000 + snr * 7 + rho), s, p, snr, rho)
            Xrm = prep_rm(R, wn)
            for tgt in ("no3", "doc", "no2"):
                for blk in ((False, True) if (snr, rho) == (20, 10) else (True,)):
                    ru = base[(tgt, blk)]
                    rr = cv_rmse(Xrm, y[tgt], blk, seed)
                    rf = cv_rmse(np.hstack([Xuv, Xrm]), y[tgt], blk, seed, blocks=[Xuv.shape[1], Xrm.shape[1]])
                    out["grid"].append(dict(seed=seed, snr=snr, rho=rho, target=tgt, blocked=blk,
                        uv_rmse=ru[0], uv_r2=ru[1], rm_rmse=rr[0], rm_r2=rr[1],
                        fu_rmse=rf[0], fu_r2=rf[1], gain_pct=100 * (ru[0] - rf[0]) / ru[0]))
        print("seed", seed, "done", flush=True)
        if seed == seeds[0]:
            out["series"] = {k: s[k].tolist() for k in ("t", "no3", "no2", "doc", "turb", "foul")}
            out["example_uv"] = {"wl": wl_ex.tolist(), "A": A[:5].tolist()} if False else None
            # drift detection on a band free of analyte absorption (700 nm)
            wl, A = uv_spectra(np.random.default_rng(99), s, p)
            out["drift"] = {"t": s["t"].tolist(), "a700": A[:, np.argmin(abs(wl - 700))].tolist(),
                            "cleaning_days": p["cleaning_days"]}
    return out


if __name__ == "__main__":
    t0 = time.time(); res = run()
    json.dump(res, open("sim_results.json", "w"))
    print("done in %.0f s" % (time.time() - t0))
