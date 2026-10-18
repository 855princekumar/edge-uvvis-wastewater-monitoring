#!/usr/bin/env python3
"""(c) Tuned 1D-CNN vs PLS and (d) learning curves, smoothed + SNV, 215 spectral channels.

(c) Nested tuning: in every outer fold (random 5-fold, seed 42; and time-blocked 5-fold) a grid of
    CNN settings (learning rate x epochs x width multiplier) is scored on an inner validation split
    taken from the training fold only (random 20% for random CV; the most recent 20% of the training
    period for blocked CV); the best setting is refitted on the full training fold and evaluated on
    the outer fold. PLS uses the pipeline's fit_pls (inner CV for latent variables).
(d) Learning curves: random 5-fold (seed 42); PLS and the default CNN trained on a fraction
    (0.1 to 1.0) of each training fold, 3 random subsamples per fraction, evaluated on the full fold.
Usage (repo root): python validation/cnn_tuning_learning_curve.py data/raw/extracted results/validation_v130
"""
import argparse, itertools, json, logging, sys, time
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as RE

GRID = list(itertools.product((1e-3, 3e-3), (60, 150), (1, 2)))      # lr, epochs, width multiplier
FRACS = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0); SUBS = 3


def cnn_fit_predict(Xtr, ytr, Xte, lr, epochs, width, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    w = lambda c: int(c * width)
    net = nn.Sequential(nn.Conv1d(1, w(16), 7, padding=3), nn.ReLU(), nn.BatchNorm1d(w(16)), nn.MaxPool1d(2),
                        nn.Conv1d(w(16), w(32), 5, padding=2), nn.ReLU(), nn.BatchNorm1d(w(32)), nn.MaxPool1d(2),
                        nn.Conv1d(w(32), w(64), 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                        nn.Flatten(), nn.Linear(w(64), 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))
    ys = StandardScaler().fit(ytr.reshape(-1, 1)); yt = torch.tensor(ys.transform(ytr.reshape(-1, 1)), dtype=torch.float32)
    Xt = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1); Xv = torch.tensor(Xte, dtype=torch.float32).unsqueeze(1)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4); lossf = nn.MSELoss(); n = len(yt); bs = min(32, n)
    for _ in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 2: continue
            opt.zero_grad(); lossf(net(Xt[idx]), yt[idx]).backward(); opt.step()
    net.eval()
    with torch.no_grad(): p = net(Xv).numpy().ravel()
    return ys.inverse_transform(p.reshape(-1, 1)).ravel()


def load(root):
    a = argparse.Namespace(targets="nh4,po4,doc", target=None, join_tol="5min", smooth=True, snv=True, spectral_only=True)
    return {r["target"]: (RE.preprocess(r["X"], a), np.asarray(r["y"], float)) for r in RE.load_eawag(Path(root), a, logging.getLogger("x"))}


if __name__ == "__main__":
    import torch; torch.set_num_threads(4)
    data_root, out_dir = sys.argv[1], Path(sys.argv[2]); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); D = load(data_root); res = {"grid": GRID, "tuning": {}, "learning_curve": {}}
    for tgt, (X, y) in D.items():
        for scheme in ("random", "blocked"):
            blocked = scheme == "blocked"
            kf = KFold(5, shuffle=False) if blocked else KFold(5, shuffle=True, random_state=42)
            folds = []
            for fi, (tr, va) in enumerate(kf.split(X)):
                sc = StandardScaler().fit(X[tr]); Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va]); ytr = y[tr]
                ntr = len(tr); cut = int(0.8 * ntr)
                if blocked: itr, iva = np.arange(cut), np.arange(cut, ntr)                 # most recent 20% of training period
                else: perm = np.random.default_rng(fi).permutation(ntr); itr, iva = perm[:cut], perm[cut:]
                isc = StandardScaler().fit(X[tr][itr]); Xi, Xiv = isc.transform(X[tr][itr]), isc.transform(X[tr][iva])
                scores = {g: r2_score(ytr[iva], cnn_fit_predict(Xi, ytr[itr], Xiv, *g, seed=42)) for g in GRID}
                best = max(scores, key=scores.get)
                yc = cnn_fit_predict(Xtr, ytr, Xva, *best, seed=42)
                yd = RE.train_predict_nn(Xtr, ytr, Xva, "cnn1d", 60, 42, None)
                p, nc = RE.fit_pls(Xtr, ytr, 42, blocked=blocked); yp = p.predict(Xva).ravel()
                folds.append({"fold": fi + 1, "best": {"lr": best[0], "epochs": best[1], "width": best[2]}, "inner_r2_best": float(scores[best]),
                              "r2_tuned_cnn": float(r2_score(y[va], yc)), "r2_default_cnn": float(r2_score(y[va], yd)), "r2_pls": float(r2_score(y[va], yp)), "pls_lv": int(nc)})
                print(tgt, scheme, folds[-1], f"{time.time() - t0:.0f}s", flush=True)
            agg = lambda k: {"mean": float(np.mean([f[k] for f in folds])), "sd": float(np.std([f[k] for f in folds]))}
            res["tuning"][f"{tgt}|{scheme}"] = {"folds": folds, "tuned_cnn": agg("r2_tuned_cnn"), "default_cnn": agg("r2_default_cnn"), "pls": agg("r2_pls"),
                                                "pls_better_than_tuned_in_folds": int(sum(f["r2_pls"] > f["r2_tuned_cnn"] for f in folds))}
            json.dump(res, open(out_dir / "cnn_tuning_learning_curve.json", "w"), indent=1)
        # (d) learning curve
        lc = {fr: {"pls": [], "cnn": [], "n_train": []} for fr in FRACS}
        for fi, (tr, va) in enumerate(KFold(5, shuffle=True, random_state=42).split(X)):
            for fr in FRACS:
                for sub in range(SUBS if fr < 1.0 else 1):
                    m = max(int(round(fr * len(tr))), 20)
                    sel = np.sort(np.random.default_rng(100 * fi + sub).choice(tr, m, replace=False)) if fr < 1.0 else tr
                    sc = StandardScaler().fit(X[sel]); Xs, Xv = sc.transform(X[sel]), sc.transform(X[va])
                    p, _ = RE.fit_pls(Xs, y[sel], 42); lc[fr]["pls"].append(float(r2_score(y[va], p.predict(Xv).ravel())))
                    lc[fr]["cnn"].append(float(r2_score(y[va], RE.train_predict_nn(Xs, y[sel], Xv, "cnn1d", 60, 42, None))))
                    lc[fr]["n_train"].append(int(m))
        res["learning_curve"][tgt] = {str(fr): {"n_train_mean": float(np.mean(v["n_train"])), "pls_mean": float(np.mean(v["pls"])), "pls_sd": float(np.std(v["pls"])),
                                                "cnn_mean": float(np.mean(v["cnn"])), "cnn_sd": float(np.std(v["cnn"]))} for fr, v in lc.items()}
        print(tgt, "learning curve", json.dumps(res["learning_curve"][tgt]), f"{time.time() - t0:.0f}s", flush=True)
        json.dump(res, open(out_dir / "cnn_tuning_learning_curve.json", "w"), indent=1)
    res["runtime_s"] = time.time() - t0; json.dump(res, open(out_dir / "cnn_tuning_learning_curve.json", "w"), indent=1); print("done", res["runtime_s"])
