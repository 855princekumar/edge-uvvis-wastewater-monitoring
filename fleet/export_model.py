#!/usr/bin/env python3
"""Run on the HOST from the repo root (needs the extracted Eawag 2_data):
    python3 fleet/export_model.py --data-root data/raw/extracted
Trains PLS (smoothed + SNV, LV count by inner CV) on all 510 matched NH4 samples, exports
deployable coefficients (JSON), REAL replay spectra (consecutive valid flume spectra) and
host reference predictions for cross-architecture agreement checks."""
import argparse, json, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as RE

ap = argparse.ArgumentParser(); ap.add_argument("--data-root", default="data/raw/extracted")
ap.add_argument("--n-replay", type=int, default=3600); a = ap.parse_args()
args = argparse.Namespace(targets="nh4", target=None, join_tol="5min", smooth=True, snv=True)
log = logging.getLogger("export"); logging.basicConfig(level=logging.INFO)
runs = RE.load_eawag(Path(a.data_root), args, log); r = runs[0]
X = RE.preprocess(r["X"], args); y = r["y"]
from sklearn.preprocessing import StandardScaler
sc = StandardScaler().fit(X); pls, nc = RE.fit_pls(sc.transform(X), y, 42)
coef = np.asarray(pls.coef_).ravel(); icpt = float(np.ravel(pls.intercept_)[0]) if hasattr(pls, "intercept_") else float(y.mean())
out = Path(__file__).parent / "models"; out.mkdir(exist_ok=True)
# sanity: exported linear form reproduces sklearn predictions
lin = ((X - sc.mean_) / sc.scale_) @ coef + icpt
assert np.allclose(lin, pls.predict(sc.transform(X)).ravel(), atol=1e-6), "export mismatch"
json.dump({"target": "lab_nh4_mg_l", "n_components": int(nc), "smooth": True, "snv": True,
           "x_mean": sc.mean_.tolist(), "x_scale": sc.scale_.tolist(), "coef": coef.tolist(),
           "intercept": icpt, "wavelengths": np.asarray(r["wl"]).tolist(), "n_train": int(len(y))},
          open(out / "pls_nh4.json", "w"))
# real replay spectra: first n consecutive valid spectra from the flume file
import pandas as pd
spec_path = next(Path(a.data_root).rglob("flume_scan_absorbance.csv"))
hdr = pd.read_csv(spec_path, nrows=0).columns.tolist()
abs_cols = [c for c in hdr if RE._abs_wavelength(c) is not None]
abs_cols = [abs_cols[i] for i in np.argsort([RE._abs_wavelength(c) for c in abs_cols])]
use = abs_cols + (["valid_data"] if "valid_data" in hdr else [])
D = pd.read_csv(spec_path, usecols=use, nrows=a.n_replay * 3)
if "valid_data" in D.columns: D = D[D["valid_data"] == 1]   # same filter as training
D = D.dropna(subset=abs_cols).head(a.n_replay)
S = D[abs_cols].to_numpy(float)
np.save(out / "replay.npy", S)
ref = ((RE.preprocess(S, args) - sc.mean_) / sc.scale_) @ coef + icpt
np.save(out / "replay_ref_pred.npy", ref)
print(f"exported PLS ({nc} LV, {coef.size} coefficients), replay {S.shape}, models/ ready")
