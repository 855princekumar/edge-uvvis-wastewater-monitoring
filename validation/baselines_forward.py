#!/usr/bin/env python3
"""Reviewer-proofing checks of the UV-Vis models (215 spectral channels, smoothed + SNV unless stated).

(a) Simple baselines vs full-spectrum PLS, under time-blocked 5-fold CV and random 5-fold CV (seed 42):
    - training mean (null model)
    - absorbance at 254 nm (raw spectrum, linear regression)  ["SAC254" proxy]
    - best single wavelength chosen inside each training fold (max |r| with the target, raw spectrum)
    - online turbidity (Turbimax, nearest valid reading within 10 min of the grab sample), log-linear
    - full-spectrum PLS (pipeline fit_pls; inner CV blocked for the blocked scheme)
(b) Forward-chaining (expanding window): train on periods 1..k, test on period k+1 (k = 1..4), PLS,
    with a purge gap of 0, 1 or 2 days removed from the training data next to the test period.
(c) Leave-one-period-out with a purge gap of 1 and 2 days on both sides of the held-out period.
(d) VIP scores and regression coefficients of PLS fitted on all 510 samples (NH4, PO4), for Fig. S7.
Fold sd uses ddof = 0, as in the main pipeline.
Usage (repo root): python validation/baselines_forward.py data/raw/extracted results/validation_v131
"""
import argparse, json, logging, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as RE


def load(root):
    a_raw = argparse.Namespace(targets="nh4,po4,doc", target=None, join_tol="5min", smooth=False, snv=False, spectral_only=True)
    a_snv = argparse.Namespace(targets="nh4,po4,doc", target=None, join_tol="5min", smooth=True, snv=True, spectral_only=True)
    runs = RE.load_eawag(Path(root), a_raw, logging.getLogger("x"))
    T = pd.read_csv(Path(root) / "2_data/3_sensor_data/flume_turbimax_turbidity.csv", usecols=["timestamp_iso", "turbimax_turbidity_fnu", "valid_data"], parse_dates=["timestamp_iso"])
    T = T[T.valid_data == 1].dropna().sort_values("timestamp_iso")
    out = {}
    for r in runs:
        t = pd.DataFrame({"t": pd.to_datetime(r["t"])})
        m = pd.merge_asof(t, T.rename(columns={"timestamp_iso": "t"}), on="t", tolerance=pd.Timedelta("10min"), direction="nearest")
        out[r["target"][4:7]] = dict(Xraw=np.asarray(r["X"], float), Xsnv=RE.preprocess(np.asarray(r["X"], float), a_snv), y=np.asarray(r["y"], float),
                                     wl=np.asarray(r["wl"], float), t=pd.to_datetime(r["t"]), turb=m.turbimax_turbidity_fnu.to_numpy())
    return out


def pls_pred(Xtr, ytr, Xte, blocked):
    sc = StandardScaler().fit(Xtr); p, nc = RE.fit_pls(sc.transform(Xtr), ytr, 42, blocked=blocked)
    return p.predict(sc.transform(Xte)).ravel(), nc


def baselines(d, tr, va, blocked):
    y = d["y"]; out = {}
    out["mean"] = np.full(len(va), y[tr].mean())
    i254 = int(np.argmin(abs(d["wl"] - 254)))
    out["a254"] = LinearRegression().fit(d["Xraw"][tr][:, [i254]], y[tr]).predict(d["Xraw"][va][:, [i254]])
    r = [abs(np.corrcoef(d["Xraw"][tr][:, j], y[tr])[0, 1]) for j in range(d["Xraw"].shape[1])]
    j = int(np.nanargmax(r)); out["best_wl"] = LinearRegression().fit(d["Xraw"][tr][:, [j]], y[tr]).predict(d["Xraw"][va][:, [j]]); out["_best_wl_nm"] = float(d["wl"][j])
    tb = d["turb"]; ok_tr = np.isfinite(tb[tr]) & (tb[tr] > 0)
    if ok_tr.sum() > 20:
        lr = LinearRegression().fit(np.log(tb[tr][ok_tr])[:, None], y[tr][ok_tr])
        xt = np.where(np.isfinite(tb[va]) & (tb[va] > 0), np.log(np.clip(tb[va], 1e-3, None)), np.log(np.nanmedian(tb[tr][ok_tr])))
        out["turbidity"] = lr.predict(xt[:, None])
    out["pls"], _ = pls_pred(d["Xsnv"][tr], y[tr], d["Xsnv"][va], blocked)
    return out


def vip(pls):
    t, w, q = pls.x_scores_, pls.x_weights_, pls.y_loadings_
    p, h = w.shape; s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    wn = w / np.linalg.norm(w, axis=0, keepdims=True)
    return np.sqrt(p * (wn ** 2 @ s).ravel() / s.sum())


if __name__ == "__main__":
    root, out_dir = sys.argv[1], Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
    D = load(root); res = {"baselines": {}, "forward": {}, "purged_lopo": {}, "vip": {}, "turbidity_coverage": {}}
    for tgt, d in D.items():
        y = d["y"]; n = len(y)
        res["turbidity_coverage"][tgt] = float(np.isfinite(d["turb"]).mean())
        for scheme in ("blocked", "random"):
            kf = KFold(5, shuffle=False) if scheme == "blocked" else KFold(5, shuffle=True, random_state=42)
            per = {}; wls = []
            for tr, va in kf.split(d["Xraw"]):
                b = baselines(d, tr, va, scheme == "blocked"); wls.append(b.pop("_best_wl_nm"))
                for k, pred in b.items():
                    per.setdefault(k, []).append((r2_score(y[va], pred), float(np.sqrt(mean_squared_error(y[va], pred)))))
            res["baselines"][f"{tgt}|{scheme}"] = {k: {"r2_mean": float(np.mean([v[0] for v in vs])), "r2_sd": float(np.std([v[0] for v in vs])),
                                                       "rmse_mean": float(np.mean([v[1] for v in vs]))} for k, vs in per.items()}
            res["baselines"][f"{tgt}|{scheme}"]["best_wl_selected_nm"] = wls
            print(tgt, scheme, {k: round(v["r2_mean"], 3) for k, v in res["baselines"][f"{tgt}|{scheme}"].items() if isinstance(v, dict)}, flush=True)
        blocks = list(KFold(5, shuffle=False).split(d["Xraw"]))
        for gap_d in (0, 1, 2):
            gap = pd.Timedelta(days=gap_d); fw = []
            for k in range(1, 5):
                te = blocks[k][1]; tr = np.concatenate([blocks[i][1] for i in range(k)])
                tr = tr[d["t"][tr] < d["t"][te].min() - gap]
                pred, nc = pls_pred(d["Xsnv"][tr], y[tr], d["Xsnv"][te], True); fw.append({"test_period": k + 1, "n_train": int(len(tr)), "r2": float(r2_score(y[te], pred)), "lv": int(nc)})
            res["forward"][f"{tgt}|gap{gap_d}d"] = {"folds": fw, "r2_mean": float(np.mean([f["r2"] for f in fw])), "r2_sd": float(np.std([f["r2"] for f in fw]))}
        for gap_d in (1, 2):
            gap = pd.Timedelta(days=gap_d); lo = []
            for tr, te in blocks:
                lo_t, hi_t = d["t"][te].min() - gap, d["t"][te].max() + gap
                keep = tr[(d["t"][tr] < lo_t) | (d["t"][tr] > hi_t)]
                pred, _ = pls_pred(d["Xsnv"][keep], y[keep], d["Xsnv"][te], True); lo.append(float(r2_score(y[te], pred)))
            res["purged_lopo"][f"{tgt}|gap{gap_d}d"] = {"r2_mean": float(np.mean(lo)), "r2_sd": float(np.std(lo)), "folds": lo}
        print(tgt, "forward", {k: round(v["r2_mean"], 3) for k, v in res["forward"].items() if k.startswith(tgt)},
              "purged", {k: round(v["r2_mean"], 3) for k, v in res["purged_lopo"].items() if k.startswith(tgt)}, flush=True)
        if tgt in ("nh4", "po4"):
            sc = StandardScaler().fit(d["Xsnv"]); p, nc = RE.fit_pls(sc.transform(d["Xsnv"]), y, 42)
            v = vip(p); coef = np.asarray(p.coef_).ravel()
            top = np.argsort(-v)[:8]
            res["vip"][tgt] = {"lv": int(nc), "wl": d["wl"].tolist(), "vip": v.tolist(), "coef": coef.tolist(), "mean_raw_spectrum": d["Xraw"].mean(0).tolist(),
                               "top_wl_nm": [float(d["wl"][i]) for i in top], "frac_vip_gt1_below_300nm": float(((v > 1) & (d["wl"] < 300)).sum() / max((v > 1).sum(), 1))}
        json.dump(res, open(out_dir / "baselines_forward.json", "w"), indent=1)
    print("done")
