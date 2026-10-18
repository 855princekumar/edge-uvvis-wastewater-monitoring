#!/usr/bin/env python3
"""Statistical validation of the PLS vs 1D-CNN comparison (215 spectral channels).

(a) Repeated random CV: R repeats x 5 folds; per fold R2 of PLS and CNN (pipeline functions
    fit_pls / train_predict_nn); paired difference tested with the corrected resampled t-test
    of Nadeau & Bengio (2003): var correction (1/(k*R) + n_test/n_train).
(b) Time-blocked CV (5 contiguous periods) with S CNN training seeds: per-period R2, seed spread,
    and a 95% CI for the pooled out-of-period R2 difference (PLS - CNN, CNN averaged over seeds)
    from a moving-block bootstrap (block = 12 consecutive samples, about one day) with B resamples.
Usage (repo root): python validation/stats_repeated_cv.py data/raw/extracted results/validation_v130
"""
import argparse, json, logging, sys, time
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as RE

R_REP, S_SEEDS, K, B, BLOCK = 10, 10, 5, 2000, 12
torch_threads = 4


def load(root):
    out = {}
    for cfg, (sm, snv) in {"SG+SNV": (True, True), "Raw": (False, False)}.items():
        a = argparse.Namespace(targets="nh4,po4,doc", target=None, join_tol="5min", smooth=sm, snv=snv, spectral_only=True)
        for r in RE.load_eawag(Path(root), a, logging.getLogger("x")):
            out[(cfg, r["target"])] = (RE.preprocess(r["X"], a), np.asarray(r["y"], float))
    return out


def fold_scores(X, y, tr, va, seed, blocked):
    sc = StandardScaler().fit(X[tr]); Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
    p, nc = RE.fit_pls(Xtr, y[tr], seed, blocked=blocked)
    yp = p.predict(Xva).ravel()
    yc = RE.train_predict_nn(Xtr, y[tr], Xva, "cnn1d", 60, seed, None)
    return yp, yc, nc


def nadeau_bengio(d, n_tr, n_te, k, r):
    d = np.asarray(d); m = d.mean(); v = d.var(ddof=1)
    se = np.sqrt((1.0 / (k * r) + n_te / n_tr) * v); t = m / se; df = k * r - 1
    p = 2 * stats.t.sf(abs(t), df); h = stats.t.ppf(0.975, df) * se
    return {"mean_diff": float(m), "ci95": [float(m - h), float(m + h)], "t": float(t), "df": df, "p": float(p)}


def block_bootstrap_r2_diff(y, yp, yc, rng):
    n = len(y); nb = int(np.ceil(n / BLOCK)); diffs = []
    for _ in range(B):
        starts = rng.integers(0, n, nb)
        idx = np.concatenate([(s + np.arange(BLOCK)) % n for s in starts])[:n]
        diffs.append(r2_score(y[idx], yp[idx]) - r2_score(y[idx], yc[idx]))
    d = np.array(diffs)
    return {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_le_0": float((d <= 0).mean())}


if __name__ == "__main__":
    import torch; torch.set_num_threads(torch_threads)
    data_root, out_dir = sys.argv[1], Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); D = load(data_root); print(f"loaded {len(D)} runs in {time.time() - t0:.0f}s", flush=True)
    res = {"settings": {"repeats": R_REP, "folds": K, "cnn_seeds_blocked": S_SEEDS, "bootstrap": B, "block_len": BLOCK, "channels": 215}, "random": {}, "blocked": {}}
    rows = []
    for (cfg, tgt), (X, y) in D.items():
        key = f"{cfg}|{tgt}"; n = len(y)
        # (a) repeated random CV
        r2p, r2c, d = [], [], []
        for rep in range(R_REP):
            for tr, va in KFold(K, shuffle=True, random_state=rep).split(X):
                yp, yc, nc = fold_scores(X, y, tr, va, rep, False)
                a, b = r2_score(y[va], yp), r2_score(y[va], yc); r2p.append(a); r2c.append(b); d.append(a - b)
                rows.append(dict(scheme="random", cfg=cfg, target=tgt, repeat=rep, r2_pls=a, r2_cnn=b, pls_lv=nc))
        nb = nadeau_bengio(d, n * (K - 1) / K, n / K, K, R_REP)
        res["random"][key] = {"pls_r2_mean": float(np.mean(r2p)), "pls_r2_sd": float(np.std(r2p)), "cnn_r2_mean": float(np.mean(r2c)),
                              "cnn_r2_sd": float(np.std(r2c)), "diff_pls_minus_cnn": nb}
        print(key, "random", json.dumps(res["random"][key]), f"{time.time() - t0:.0f}s", flush=True)
        # (b) blocked CV, CNN seed variation
        oof_p = np.zeros(n); oof_c = np.zeros((S_SEEDS, n)); per = []
        for bi, (tr, va) in enumerate(KFold(K, shuffle=False).split(X)):
            sc = StandardScaler().fit(X[tr]); Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
            p, nc = RE.fit_pls(Xtr, y[tr], 42, blocked=True); oof_p[va] = p.predict(Xva).ravel()
            cn = []
            for s in range(S_SEEDS):
                oof_c[s, va] = RE.train_predict_nn(Xtr, y[tr], Xva, "cnn1d", 60, s, None); cn.append(r2_score(y[va], oof_c[s, va]))
            per.append({"block": bi + 1, "pls_r2": float(r2_score(y[va], oof_p[va])), "pls_lv": int(nc),
                        "cnn_r2_mean": float(np.mean(cn)), "cnn_r2_sd": float(np.std(cn)), "cnn_r2_min": float(np.min(cn)), "cnn_r2_max": float(np.max(cn))})
            for s, v in enumerate(cn): rows.append(dict(scheme="blocked", cfg=cfg, target=tgt, repeat=s, block=bi + 1, r2_pls=per[-1]["pls_r2"], r2_cnn=v, pls_lv=nc))
        yc_mean = oof_c.mean(0)
        res["blocked"][key] = {"per_block": per,
                               "pooled_r2_pls": float(r2_score(y, oof_p)), "pooled_r2_cnn_seedmean": float(r2_score(y, yc_mean)),
                               "pooled_r2_cnn_per_seed": [float(r2_score(y, oof_c[s])) for s in range(S_SEEDS)],
                               "block_mean_r2_pls": float(np.mean([b["pls_r2"] for b in per])),
                               "block_mean_r2_cnn": float(np.mean([b["cnn_r2_mean"] for b in per])),
                               "pls_better_in_blocks": int(sum(b["pls_r2"] > b["cnn_r2_mean"] for b in per)),
                               "diff_bootstrap": block_bootstrap_r2_diff(y, oof_p, yc_mean, np.random.default_rng(7))}
        print(key, "blocked", json.dumps({k: v for k, v in res["blocked"][key].items() if k != "per_block"}), f"{time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(out_dir / "stats_repeated_cv.json", "w"), indent=1)
    import pandas as pd; pd.DataFrame(rows).to_csv(out_dir / "stats_repeated_cv_folds.csv", index=False)
    res["runtime_s"] = time.time() - t0; json.dump(res, open(out_dir / "stats_repeated_cv.json", "w"), indent=1); print("done", res["runtime_s"])
