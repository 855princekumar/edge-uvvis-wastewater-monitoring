#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiment.py  (v2)
=======================
Self-bootstrapping benchmarking pipeline for UV-Vis wastewater spectroscopy.
Proposed model (1D-CNN) vs control (PLS regression), with k-fold cross-
validation, automated dataset download/extract, multi-file discovery, a
spectra<->grab-sample timestamp join, and a unified cross-run benchmark
dashboard (Apache ECharts) plus 300-dpi matplotlib figures.

ONE COMMAND does everything:
  1. Builds ./.venv, installs deps, re-launches itself inside it.
  2. Acquires data (local file / folder / --url zip download+extract / SYNTHETIC).
  3. Preprocesses spectra; trains PLS (control) + 1D-CNN (proposed; MLP fallback
     if PyTorch is unavailable).
  4. Evaluates on a held-out test split AND with k-fold CV (mean +/- sd).
  5. Saves per-run outputs and a UNIFIED benchmark (metrics, figures,
     dashboard.html, report.md) into ./results/.

INTEGRITY SAFEGUARD
  If real data is not supplied, a clearly-labelled SYNTHETIC set is used and
  every output is stamped "SYNTHETIC - NOT FOR PUBLICATION". Never paste
  synthetic numbers into a manuscript.

------------------------------------------------------------------------------
THE DATASET (cited as ref [37] in the manuscript)
  Package DOI:  https://doi.org/10.25678/000D3C   (Eawag ERIC/open)
  Download ONLY the resource "2_data.zip" (UV-Vis spectra + grab-sample labs).
  Do NOT download hyperspectral_acquisitions_part01..12.zip (image cubes; huge).
  On the ERIC page, right-click the "2_data.zip" download button -> Copy link,
  then pass it as --url "<link>".
------------------------------------------------------------------------------

Examples:
  python run_experiment.py                                  # SYNTHETIC demo
  python run_experiment.py --url "<2_data.zip link>" --target ammonium --kfold 5
  python run_experiment.py --data-dir ./data/raw/2_data --targets "ammonium,DOC"
  python run_experiment.py --spectra-file spectra.csv --labels-file grab.csv \
        --time-col datetime --target ammonium --join-tol 2min --kfold 5
"""

import os
import sys
import subprocess
import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV_DIR = HERE / ".venv"
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"
OUT_DIR = None          # set per-run to results/batch-N
VENV_FLAG = "RUN_EXPERIMENT_INSIDE_VENV"

CORE_DEPS = ["numpy", "pandas", "scipy", "scikit-learn",
             "matplotlib", "openpyxl", "requests"]


# ============================================================================
# 1. BOOTSTRAP
# ============================================================================
def _venv_python() -> Path:
    return VENV_DIR / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def bootstrap_and_reexec(argv):
    print("=" * 70, "\nBOOTSTRAP: local environment in", VENV_DIR, "\n" + "=" * 70)
    if not VENV_DIR.exists():
        try:
            import venv
            venv.EnvBuilder(with_pip=True).create(VENV_DIR)
            print("[ok] venv created")
        except Exception as e:
            sys.exit(f"[fatal] cannot create venv: {e}")
    else:
        print("[ok] reusing .venv")
    py = _venv_python()
    if not py.exists():
        sys.exit(f"[fatal] venv python missing at {py}")
    subprocess.run([str(py), "-m", "pip", "install", "--upgrade",
                    "pip", "setuptools", "wheel"], check=False)
    print("[..] installing core deps")
    if subprocess.run([str(py), "-m", "pip", "install", *CORE_DEPS]).returncode:
        sys.exit("[fatal] core dependency install failed")
    print("[..] installing PyTorch CPU (for the 1D-CNN; optional)")
    if subprocess.run([str(py), "-m", "pip", "install", "torch",
                       "--index-url", "https://download.pytorch.org/whl/cpu"],
                      check=False).returncode:
        subprocess.run([str(py), "-m", "pip", "install", "torch"], check=False)
    env = dict(os.environ); env[VENV_FLAG] = "1"
    print("=" * 70, "\nRe-launching inside venv...\n" + "=" * 70)
    sys.exit(subprocess.run([str(py), str(Path(__file__).resolve()), *argv],
                            env=env).returncode)


# ============================================================================
# 2. HELPERS (imported only after venv active)
# ============================================================================
def _is_float(s):
    try:
        float(s); return True
    except (ValueError, TypeError):
        return False


def make_logger():
    import logging
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger("uvvis"); lg.setLevel(logging.INFO); lg.handlers.clear()
    fmt = __import__("logging").Formatter("%(asctime)s %(levelname)-7s %(message)s",
                                          "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); lg.addHandler(sh)
    fh = logging.FileHandler(OUT_DIR / "run_log.txt", "w", "utf-8")
    fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def download(url, dest, logger):
    """curl -> requests fallback."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        logger.info("Downloading with curl...")
        r = subprocess.run(["curl", "-L", "--fail", "-o", str(dest), url])
        if r.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            return dest
        logger.warning("curl failed (%s); trying requests", r.returncode)
    import requests
    logger.info("Downloading with requests...")
    with requests.get(url, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(1 << 16):
                f.write(chunk)
    return dest


def discover_tables(root: Path):
    """Find candidate tabular files, skipping obvious non-data."""
    exts = {".csv", ".tsv", ".txt", ".xlsx", ".xls"}
    files = []
    for p in root.rglob("*"):
        if p.suffix.lower() in exts and p.is_file():
            if p.stat().st_size > 800 * 1024 * 1024:      # skip >800MB
                continue
            files.append(p)
    return sorted(files)


def read_table(path: Path):
    import pandas as pd
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=None, engine="python")


def wavelength_cols(df, exclude=()):
    def is_wl(c):
        return _is_float(str(c).replace("nm", "").strip())
    return [c for c in df.columns if c not in exclude and is_wl(c)]


def find_datetime_col(df):
    import pandas as pd
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in ("time", "date", "stamp", "datetime")):
            try:
                pd.to_datetime(df[c], errors="raise")
                return c
            except Exception:
                continue
    return None


# --- Eawag ERIC dataset (DOI 10.25678/000D3C) specific handling ----------
EAWAG_TARGET_ALIAS = {
    "doc": "lab_doc_mg_l", "toc": "lab_toc_mg_l", "tss": "lab_tss_mg_l",
    "nh4": "lab_nh4_mg_l", "ammonium": "lab_nh4_mg_l",
    "po4": "lab_po4_mg_l", "phosphate": "lab_po4_mg_l",
    "so4": "lab_so4_mg_l", "sulfate": "lab_so4_mg_l",
    "ntot": "lab_ntot_mg_l", "tn": "lab_ntot_mg_l",
    "nsol": "lab_nsol_mg_l", "turbidity": "lab_turbidity_ntu",
}
EAWAG_DEFAULT_TARGETS = ["doc", "nh4", "tss", "po4"]


def _abs_wavelength(colname):
    """'scan_absorbance2000__m' -> 200.0 nm ; returns None if not absorbance."""
    import re
    m = re.search(r"absorbance(\d+)", str(colname).lower())
    if not m:
        return None
    return int(m.group(1)) / 10.0


def find_eawag_root(root: Path):
    """Return the folder containing laboratory_measurements.csv, or None."""
    hit = next(root.rglob("laboratory_measurements.csv"), None)
    return hit.parent.parent if hit else None   # .../2_data/


def load_eawag(root: Path, args, logger):
    """Build spectra<->grab-sample runs from the Eawag 2_data layout."""
    import re
    import numpy as np
    import pandas as pd

    lab_path = next(root.rglob("laboratory_measurements.csv"), None)
    if lab_path is None:
        return None
    spec_path = None
    for pat in ("flume_scan_absorbance.csv", "flume_isa_absorbance.csv"):
        c = next(root.rglob(pat), None)
        if c:
            spec_path = c
            break
    if spec_path is None:
        logger.warning("Eawag layout: no scan/isa absorbance file found.")
        return None
    logger.info("Eawag mode: labels=%s  spectra=%s",
                lab_path.name, spec_path.name)

    # ---- spectra: read only absorbance + timestamp + valid_data ----
    hdr = pd.read_csv(spec_path, nrows=0).columns.tolist()
    abs_cols = [c for c in hdr if _abs_wavelength(c) is not None]
    if getattr(args, "spectral_only", False):
        # v1.0.1 also matched 'validity_test3_range_scan_absorbance2550_m' (a validity-test
        # field, not a spectral channel); --spectral-only keeps the scan_absorbance channels only
        dropped = [c for c in abs_cols if not str(c).lower().startswith("scan_absorbance")]
        abs_cols = [c for c in abs_cols if str(c).lower().startswith("scan_absorbance")]
        logger.info("--spectral-only: dropped non-spectral columns %s", dropped)
    if len(abs_cols) < 10:
        logger.warning("Only %d absorbance columns detected.", len(abs_cols))
        return None
    usecols = ["timestamp_iso"] + abs_cols + (
        ["valid_data"] if "valid_data" in hdr else [])
    logger.info("Reading %d absorbance columns from spectra (may take a moment)",
                len(abs_cols))
    spec = pd.read_csv(spec_path, usecols=usecols)
    if "valid_data" in spec.columns:
        before = len(spec)
        spec = spec[spec["valid_data"] == 1]
        logger.info("Kept %d/%d valid spectra rows.", len(spec), before)
    spec["timestamp_iso"] = pd.to_datetime(spec["timestamp_iso"], errors="coerce")
    spec = spec.dropna(subset=["timestamp_iso"]).sort_values("timestamp_iso")

    wl = np.array([_abs_wavelength(c) for c in abs_cols], float)
    order = np.argsort(wl)
    wl = wl[order]
    abs_cols = [abs_cols[i] for i in order]

    # ---- labels ----
    lab = pd.read_csv(lab_path)
    lab["timestamp_iso"] = pd.to_datetime(lab["timestamp_iso"], errors="coerce")
    lab = lab.dropna(subset=["timestamp_iso"])

    # resolve targets
    req = ([t.strip() for t in args.targets.split(",")] if args.targets
           else ([args.target] if args.target else EAWAG_DEFAULT_TARGETS))
    tcols = []
    for t in req:
        col = t if str(t).startswith("lab_") else EAWAG_TARGET_ALIAS.get(t.lower())
        if col and col in lab.columns:
            tcols.append(col)
        else:
            logger.warning("target '%s' -> no column; available: %s", t,
                           [c for c in lab.columns if c.startswith("lab_")])
    if not tcols:
        sys.exit("[fatal] none of the requested targets exist in the lab file.")

    tol = pd.Timedelta(args.join_tol)
    runs = []
    for tcol in tcols:
        sub = lab[["timestamp_iso", tcol]].dropna().sort_values("timestamp_iso")
        if len(sub) < 40:
            logger.warning("%s: only %d grab samples; skip", tcol, len(sub))
            continue
        merged = pd.merge_asof(sub, spec[["timestamp_iso"] + abs_cols],
                               on="timestamp_iso", direction="nearest",
                               tolerance=tol)
        merged = merged.dropna(subset=abs_cols)
        if len(merged) < 40:
            logger.warning("%s: only %d matched within %s; skip",
                           tcol, len(merged), args.join_tol)
            continue
        X = merged[abs_cols].to_numpy(float)
        y = merged[tcol].to_numpy(float)
        logger.info("%s: %d grab samples matched to spectra", tcol, len(merged))
        t = merged["timestamp_iso"].to_numpy()
        gaps_h = np.diff(t).astype("timedelta64[s]").astype(float) / 3600.0
        logger.info("%s: span %s -> %s; median gap between grab samples %.2f h",
                    tcol, str(t[0])[:16], str(t[-1])[:16], float(np.median(gaps_h)))
        runs.append({"label": f"{spec_path.stem}/{tcol}", "X": X, "y": y,
                     "wl": wl, "target": tcol, "t": t})
    return runs or None


def extract_local_zips(logger):
    """Auto-detect *.zip beside the script; extract the DATA zip only.
    Returns the extraction root, or None if no data zip found."""
    import zipfile
    zips = sorted(HERE.glob("*.zip"))
    if not zips:
        return None
    ext_root = DATA_DIR / "raw" / "extracted"
    ext_root.mkdir(parents=True, exist_ok=True)
    found_data = None
    for z in zips:
        try:
            with zipfile.ZipFile(z) as zf:
                names = zf.namelist()
                is_data = any("laboratory_measurements.csv" in n for n in names)
                is_codes = any(n.startswith("1_codes/") for n in names)
                if is_data:
                    marker = ext_root / (z.stem + ".extracted")
                    if not marker.exists():
                        logger.info("Extracting data zip: %s", z.name)
                        zf.extractall(ext_root)
                        marker.write_text("ok")
                    else:
                        logger.info("Data zip already extracted: %s", z.name)
                    found_data = ext_root
                elif is_codes:
                    logger.info("Found %s (author code/demo) - reference only, "
                                "not used for analysis.", z.name)
                else:
                    logger.info("Ignoring unrecognised zip: %s", z.name)
        except zipfile.BadZipFile:
            logger.warning("Not a valid zip: %s", z.name)
    return found_data


# ============================================================================
# 3. METRICS
# ============================================================================
def metrics(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    sd = float(np.std(y_true))
    return {"R2": float(r2_score(y_true, y_pred)), "RMSE": rmse,
            "MAE": float(mean_absolute_error(y_true, y_pred)),
            "RPD": float(sd / rmse) if rmse > 0 else float("nan"),
            "n": int(len(y_true))}


# ============================================================================
# 4. MODELS
# ============================================================================
def fit_pls(Xtr, ytr, seed, max_nc=20, blocked=False):
    import numpy as np
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    best_nc, best = 2, -1e18
    mx = min(max_nc, Xtr.shape[1], Xtr.shape[0] - 1)
    kf = (KFold(n_splits=min(5, len(ytr)), shuffle=False) if blocked else
          KFold(n_splits=min(5, len(ytr)), shuffle=True, random_state=seed))
    for nc in range(2, max(3, mx) + 1):
        sc = []
        for tr, va in kf.split(Xtr):
            m = PLSRegression(n_components=nc).fit(Xtr[tr], ytr[tr])
            sc.append(r2_score(ytr[va], m.predict(Xtr[va]).ravel()))
        if np.mean(sc) > best:
            best, best_nc = float(np.mean(sc)), nc
    return PLSRegression(n_components=best_nc).fit(Xtr, ytr), best_nc


def build_cnn(n_in, seed):
    import torch, torch.nn as nn
    torch.manual_seed(seed)

    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(16),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(32),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(64, 32), nn.ReLU(),
                                      nn.Dropout(0.2), nn.Linear(32, 1))

        def forward(self, x):
            return self.head(self.net(x))
    return CNN1D()


def train_predict_nn(Xtr, ytr, Xte, model_kind, epochs, seed, logger):
    """Return predictions on Xte (original y units)."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    ys = StandardScaler().fit(ytr.reshape(-1, 1))
    ytr_s = ys.transform(ytr.reshape(-1, 1)).ravel()
    if model_kind == "cnn1d":
        import torch, torch.nn as nn
        net = build_cnn(Xtr.shape[1], seed)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.MSELoss()
        Xt = torch.tensor(Xtr, dtype=torch.float32).unsqueeze(1)
        yt = torch.tensor(ytr_s, dtype=torch.float32).unsqueeze(1)
        Xv = torch.tensor(Xte, dtype=torch.float32).unsqueeze(1)
        n = len(yt); bs = min(32, n)
        for ep in range(epochs):
            net.train(); perm = torch.randperm(n)
            for i in range(0, n, bs):
                idx = perm[i:i + bs]; opt.zero_grad()
                loss = lossf(net(Xt[idx]), yt[idx]); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            pred_s = net(Xv).cpu().numpy().ravel()
    else:
        from sklearn.neural_network import MLPRegressor
        mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=epochs * 10,
                           random_state=seed, early_stopping=True).fit(Xtr, ytr_s)
        pred_s = mlp.predict(Xte)
    return ys.inverse_transform(pred_s.reshape(-1, 1)).ravel()


# ============================================================================
# 5. BENCHMARK ONE (X, y): held-out split + optional k-fold
# ============================================================================
def benchmark(X, y, args, logger, model_kind):
    import numpy as np
    from sklearn.model_selection import train_test_split, KFold
    from sklearn.preprocessing import StandardScaler

    out = {"models": {}, "predictions": {}, "test_y": None}

    # ---- held-out split for plots ----
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size,
                                          random_state=args.seed)
    xs = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = xs.transform(Xtr), xs.transform(Xte)
    out["test_y"] = yte

    # PLS control
    pls, nc = fit_pls(Xtr_s, ytr, args.seed)
    yhat = pls.predict(Xte_s).ravel()
    out["models"]["PLS"] = {**metrics(yte, yhat),
                            "label": "PLS (control)", "n_components": nc}
    out["predictions"]["PLS"] = yhat

    # proposed NN
    yhat2 = train_predict_nn(Xtr_s, ytr, Xte_s, model_kind, args.epochs,
                             args.seed, logger)
    lbl = "1D-CNN (proposed)" if model_kind == "cnn1d" else \
          "MLP (proposed; torch unavailable)"
    key = "CNN1D" if model_kind == "cnn1d" else "MLP"
    out["models"][key] = {**metrics(yte, yhat2), "label": lbl}
    out["predictions"][key] = yhat2

    # ---- k-fold CV (mean +/- sd) ----
    if args.kfold and args.kfold >= 2:
        blocked = getattr(args, "cv", "random") == "blocked"
        logger.info("  running %d-fold CV (%s)...", args.kfold,
                    "time-blocked, contiguous periods" if blocked else "random")
        # data are time-sorted by the merge, so unshuffled KFold = contiguous
        # periods (leave-one-period-out); inner PLS CV is blocked as well
        kf = (KFold(n_splits=args.kfold, shuffle=False) if blocked else
              KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed))
        fold_log = []
        agg = {"PLS": {"R2": [], "RMSE": [], "MAE": [], "RPD": []},
               key: {"R2": [], "RMSE": [], "MAE": [], "RPD": []}}
        for fold, (tr, va) in enumerate(kf.split(X)):
            sca = StandardScaler().fit(X[tr])
            Xtr_f, Xva_f = sca.transform(X[tr]), sca.transform(X[va])
            p, ncf = fit_pls(Xtr_f, y[tr], args.seed, blocked=blocked)
            mp = metrics(y[va], p.predict(Xva_f).ravel())
            for k in agg["PLS"]:
                agg["PLS"][k].append(mp[k])
            pn = train_predict_nn(Xtr_f, y[tr], Xva_f, model_kind,
                                  args.epochs, args.seed, logger)
            mn = metrics(y[va], pn)
            for k in agg[key]:
                agg[key][k].append(mn[k])
            logger.info("    fold %d: n_val=%d  PLS(nc=%d) R2=%.3f  %s R2=%.3f",
                        fold + 1, len(va), ncf, mp["R2"], key, mn["R2"])
        for mdl in (["PLS", key]):
            cv = {}
            for k, vals in agg[mdl].items():
                cv[f"cv_{k}_mean"] = float(np.mean(vals))
                cv[f"cv_{k}_sd"] = float(np.std(vals))
            out["models"][mdl].update(cv)
            m = out["models"][mdl]
            logger.info("  %s CV: R2=%.3f+/-%.3f  RMSE=%.3f+/-%.3f",
                        mdl, m["cv_R2_mean"], m["cv_R2_sd"],
                        m["cv_RMSE_mean"], m["cv_RMSE_sd"])
    return out


# ============================================================================
# 6. ACQUIRE -> list of (label, X, y, wl) runs
# ============================================================================
def synth_dataset(seed, n=600, n_wl=200):
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    wl = np.linspace(200, 400, n_wl)
    conc = rng.uniform(5, 400, n); turb = rng.uniform(0, 1, n)
    X = np.zeros((n, n_wl))
    for i in range(n):
        p1 = np.exp(-0.5 * ((wl - 254) / 12) ** 2)
        p2 = np.exp(-0.5 * ((wl - 285) / 20) ** 2)
        X[i] = (conc[i] / 400) * (0.9 * p1 + 0.6 * p2) + turb[i] * (400 - wl) / 200
        X[i] += rng.normal(0, 0.01, n_wl)
    df = pd.DataFrame(X, columns=[f"{w:.1f}" for w in wl]); df["COD"] = conc
    return df


def load_runs(args, logger):
    """Return (runs, data_source_label, is_synthetic). runs: list of dicts
    {label, X, y, wl, target}."""
    import numpy as np, pandas as pd
    DATA_DIR.mkdir(exist_ok=True)
    targets = ([t.strip() for t in args.targets.split(",")] if args.targets
               else ([args.target] if args.target else []))

    # ----- explicit join mode -----
    if args.spectra_file and args.labels_file:
        logger.info("Join mode: spectra=%s labels=%s",
                    args.spectra_file, args.labels_file)
        sp = read_table(Path(args.spectra_file))
        lb = read_table(Path(args.labels_file))
        tcol = args.time_col or find_datetime_col(sp)
        tcol_l = args.time_col or find_datetime_col(lb)
        if not tcol or not tcol_l:
            sys.exit("[fatal] could not find a datetime column in both files; "
                     "pass --time-col <name>.")
        sp[tcol] = pd.to_datetime(sp[tcol]); lb[tcol_l] = pd.to_datetime(lb[tcol_l])
        sp = sp.sort_values(tcol); lb = lb.sort_values(tcol_l)
        merged = pd.merge_asof(lb, sp, left_on=tcol_l, right_on=tcol,
                               direction="nearest",
                               tolerance=pd.Timedelta(args.join_tol))
        merged = merged.dropna(subset=wavelength_cols(sp, exclude=(tcol,)))
        logger.info("Joined rows: %d", len(merged))
        runs = []
        for tgt in targets:
            if tgt not in merged.columns:
                logger.warning("target %s not in joined data; skip", tgt); continue
            runs += _rows_to_run(merged, tgt, "joined", logger)
        return runs, f"REAL (join: {Path(args.labels_file).name})", False

    # ----- acquire files -----
    files = []
    is_syn = False
    label = None
    if args.data:
        files = [Path(args.data)]; label = f"REAL ({Path(args.data).name})"
    elif args.url:
        dest = DATA_DIR / "raw" / ("download.zip" if args.url.lower().endswith(".zip")
                                   else "download.dat")
        download(args.url, dest, logger)
        if dest.suffix == ".zip" or args.url.lower().endswith(".zip"):
            ext = DATA_DIR / "raw" / "extracted"
            ext.mkdir(parents=True, exist_ok=True)
            import zipfile
            with zipfile.ZipFile(dest) as z:
                z.extractall(ext)
            logger.info("Extracted to %s", ext)
            eaw = load_eawag(ext, args, logger) if find_eawag_root(ext) else None
            if eaw:
                return eaw, f"REAL (url zip: {Path(args.url).name})", False
            files = discover_tables(ext)
        else:
            files = [dest]
        label = f"REAL (url: {Path(args.url).name})"
    elif args.data_dir:
        d = Path(args.data_dir)
        if find_eawag_root(d):
            eaw = load_eawag(d, args, logger)
            if eaw:
                return eaw, f"REAL (dir: {d.name})", False
        files = discover_tables(d)
        label = f"REAL (dir: {d.name})"
    else:
        # No explicit source: try zips sitting next to the script.
        ext = extract_local_zips(logger)
        if ext and find_eawag_root(ext):
            eaw = load_eawag(ext, args, logger)
            if eaw:
                return eaw, "REAL (local 2_data.zip)", False
        if ext:
            files = discover_tables(ext)
            label = "REAL (local zip)"
            if not files:
                is_syn = True; label = "SYNTHETIC - NOT FOR PUBLICATION"
        else:
            is_syn = True; label = "SYNTHETIC - NOT FOR PUBLICATION"

    if is_syn:
        df = synth_dataset(args.seed)
        return _rows_to_run(df, "COD", "SYNTHETIC", logger), label, True

    if not files:
        sys.exit("[fatal] no tabular files found to process.")
    if not targets:
        # show what's available to help the user choose
        sample = read_table(files[0])
        sys.exit("[fatal] specify --target or --targets. Columns in "
                 f"{files[0].name}: {list(sample.columns)[:40]}")

    # classify + build runs
    runs = []
    spectra_like, labels_like = [], []
    for f in files:
        try:
            df = read_table(f)
        except Exception as e:
            logger.warning("skip %s (%s)", f.name, e); continue
        wl = wavelength_cols(df)
        has_tgt = [t for t in targets if t in df.columns]
        if len(wl) >= 10 and has_tgt:
            for t in has_tgt:
                runs += _rows_to_run(df, t, f.stem, logger)
        elif len(wl) >= 10:
            spectra_like.append(f)
        elif has_tgt:
            labels_like.append(f)
    if not runs and spectra_like and labels_like:
        logger.warning("No single file had BOTH spectra and a target.")
        logger.warning("Spectra-like: %s", [p.name for p in spectra_like])
        logger.warning("Labels-like : %s", [p.name for p in labels_like])
        logger.warning("Use join mode, e.g.: --spectra-file %s --labels-file %s "
                       "--target %s --time-col <datetime col> --join-tol %s",
                       spectra_like[0], labels_like[0],
                       targets[0], args.join_tol)
        sys.exit("[fatal] need spectra<->labels join; see guidance above.")
    if not runs:
        sys.exit("[fatal] no usable (spectra+target) runs found.")
    return runs, label, False


def _rows_to_run(df, target, label, logger):
    import numpy as np, pandas as pd
    wl = wavelength_cols(df, exclude=(target,))
    if len(wl) < 10 or target not in df.columns:
        return []
    sub = df[wl + [target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 40:
        logger.warning("run '%s/%s': only %d rows; skipping", label, target, len(sub))
        return []
    X = sub[wl].to_numpy(float); y = sub[target].to_numpy(float)
    wl_axis = (np.array([float(str(c).replace("nm", "").strip()) for c in wl])
               if all(_is_float(str(c).replace("nm", "").strip()) for c in wl)
               else np.arange(len(wl), dtype=float))
    logger.info("run '%s/%s': %d samples x %d wl", label, target, X.shape[0], len(wl))
    return [{"label": f"{label}/{target}", "X": X, "y": y, "wl": wl_axis,
             "target": target}]


# ============================================================================
# 7. PREPROCESS a run's X
# ============================================================================
def preprocess(X, args):
    import numpy as np
    from scipy.signal import savgol_filter
    Xp = X.copy()
    if args.smooth and Xp.shape[1] >= 11:
        Xp = savgol_filter(Xp, 11, 2, axis=1)
    if args.snv:
        mu = Xp.mean(1, keepdims=True); sd = Xp.std(1, keepdims=True) + 1e-8
        Xp = (Xp - mu) / sd
    return Xp


# ============================================================================
# 8. MAIN EXPERIMENT
# ============================================================================
def next_batch_dir(tag=None):
    """Return a fresh results/batch-N (auto-incrementing) directory."""
    RESULTS_DIR.mkdir(exist_ok=True)
    existing = []
    for p in RESULTS_DIR.glob("batch-*"):
        if p.is_dir():
            head = p.name.split("_")[0]          # 'batch-3' from 'batch-3_tag'
            num = head.replace("batch-", "")
            if num.isdigit():
                existing.append(int(num))
    n = (max(existing) + 1) if existing else 1
    name = f"batch-{n}" + (f"_{tag}" if tag else "")
    d = RESULTS_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def append_batch_summary(dfm, meta):
    """Append this batch's key metrics to results/batches_summary.csv."""
    import csv
    summary = RESULTS_DIR / "batches_summary.csv"
    new = not summary.exists()
    cols = ["batch", "timestamp", "data_source", "model_kind", "kfold",
            "smooth", "snv", "run", "model", "R2", "RMSE", "RPD", "n",
            "cv_R2_mean", "cv_R2_sd", "cv_RMSE_mean", "cv_RMSE_sd"]
    with open(summary, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        for _, r in dfm.iterrows():
            row = {"batch": OUT_DIR.name, "timestamp": meta["timestamp"],
                   "data_source": meta["data_source"],
                   "model_kind": meta["model_kind"], "kfold": meta["kfold"],
                   "smooth": meta["preprocessing"]["savgol"],
                   "snv": meta["preprocessing"]["snv"]}
            for c in cols:
                if c in r and c not in row:
                    row[c] = r[c]
            w.writerow(row)
    return summary


def run_experiment(args):
    import json, datetime as dt
    import numpy as np
    global OUT_DIR
    OUT_DIR = next_batch_dir(args.tag)
    logger = make_logger()
    np.random.seed(args.seed)
    logger.info("Batch output dir: %s", OUT_DIR)

    # model kind (with torch check up front)
    model_kind = args.model
    if model_kind == "cnn1d":
        try:
            import torch  # noqa
        except Exception:
            logger.warning("PyTorch not importable -> using MLP fallback.")
            model_kind = "mlp"

    logger.info("=" * 60)
    logger.info("Acquiring data")
    runs, data_label, is_syn = load_runs(args, logger)
    logger.info("Total runs to benchmark: %d", len(runs))

    all_records = []      # flat rows for unified table
    per_run_payload = []  # for dashboard
    for r in runs:
        logger.info("=" * 60)
        logger.info("BENCHMARK: %s", r["label"])
        Xp = preprocess(r["X"], args)
        try:
            res = benchmark(Xp, r["y"], args, logger, model_kind)
        except Exception as e:
            logger.exception("run %s failed: %s", r["label"], e); continue

        safe = r["label"].replace("/", "__").replace(" ", "_")
        run_dir = OUT_DIR / safe; run_dir.mkdir(exist_ok=True)
        _plots_for_run(run_dir, r, res, data_label, is_syn)

        for mdl, m in res["models"].items():
            all_records.append({"run": r["label"], "target": r["target"],
                                "model": m["label"], **m})
        per_run_payload.append({
            "label": r["label"], "target": r["target"],
            "test_y": [float(v) for v in res["test_y"]],
            "preds": {k: [float(v) for v in vv]
                      for k, vv in res["predictions"].items()},
            "models": res["models"]})

    if not all_records:
        sys.exit("[fatal] no successful runs; see results/run_log.txt")

    # ---- unified metrics table ----
    import pandas as pd
    dfm = pd.DataFrame(all_records)
    dfm.to_csv(OUT_DIR / "unified_metrics.csv", index=False)
    meta = {"timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "data_source": data_label, "is_synthetic": is_syn,
            "kfold": args.kfold, "model_kind": model_kind,
            "preprocessing": {"savgol": args.smooth, "snv": args.snv},
            "records": all_records}
    (OUT_DIR / "unified_metrics.json").write_text(json.dumps(meta, indent=2))

    _unified_matplotlib(dfm, data_label, is_syn)
    _unified_dashboard(per_run_payload, dfm, data_label, is_syn, args)
    _report_md(dfm, data_label, is_syn, args)
    summary = append_batch_summary(dfm, meta)

    logger.info("=" * 60)
    logger.info("DONE. Batch outputs in %s", OUT_DIR)
    logger.info("Cross-batch summary updated: %s", summary)
    if is_syn:
        logger.warning("Run was SYNTHETIC - do not publish these numbers.")


# ============================================================================
# 9. PER-RUN PLOTS
# ============================================================================
def _plots_for_run(run_dir, r, res, stamp, is_syn):
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warn = "#b00020" if is_syn else "#333"
    yte = res["test_y"]; preds = res["predictions"]

    fig, axes = plt.subplots(1, len(preds), figsize=(5.2 * len(preds), 4.6),
                             squeeze=False)
    for ax, (k, yp) in zip(axes[0], preds.items()):
        ax.scatter(yte, yp, s=16, alpha=.7, edgecolor="none")
        lo, hi = min(yte.min(), yp.min()), max(yte.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "--", color="grey", lw=1)
        m = res["models"][k]
        ax.set_xlabel(f"Measured {r['target']}"); ax.set_ylabel("Predicted")
        ax.set_title(f"{m['label']}\nR2={m['R2']:.3f} RMSE={m['RMSE']:.2f} "
                     f"RPD={m['RPD']:.2f}")
    fig.suptitle(f"{r['label']}  |  {stamp}", color=warn, fontsize=9, y=1.03)
    fig.tight_layout(); fig.savefig(run_dir / "pred_vs_measured.png", dpi=300,
                                    bbox_inches="tight"); plt.close(fig)


# ============================================================================
# 10. UNIFIED MATPLOTLIB (cross-run benchmark)
# ============================================================================
def _unified_matplotlib(dfm, stamp, is_syn):
    import numpy as np, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warn = "#b00020" if is_syn else "#333"
    runs = list(dict.fromkeys(dfm["run"]))
    models = list(dict.fromkeys(dfm["model"]))
    x = np.arange(len(runs)); w = 0.8 / max(1, len(models))

    for metric in ("R2", "RMSE"):
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(runs)), 4.6))
        for i, mdl in enumerate(models):
            vals = [dfm[(dfm["run"] == rr) & (dfm["model"] == mdl)][metric].mean()
                    for rr in runs]
            ax.bar(x + i * w, vals, w, label=mdl)
        ax.set_xticks(x + w * (len(models) - 1) / 2)
        ax.set_xticklabels(runs, rotation=20, ha="right", fontsize=8)
        ax.set_ylabel(metric); ax.legend(fontsize=8)
        ax.set_title(f"Unified benchmark - {metric} by run/model")
        fig.suptitle(stamp, color=warn, fontsize=9, y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"unified_{metric}.png", dpi=300,
                    bbox_inches="tight")
        plt.close(fig)


# ============================================================================
# 11. UNIFIED ECHARTS DASHBOARD
# ============================================================================
def _unified_dashboard(payload, dfm, stamp, is_syn, args):
    import json
    runs = list(dict.fromkeys(dfm["run"]))
    models = list(dict.fromkeys(dfm["model"]))
    overview = {"runs": runs, "models": models,
                "R2": {m: [float(dfm[(dfm.run == rr) & (dfm.model == m)]["R2"].mean())
                           for rr in runs] for m in models},
                "RMSE": {m: [float(dfm[(dfm.run == rr) & (dfm.model == m)]["RMSE"].mean())
                             for rr in runs] for m in models}}
    data = {"stamp": stamp, "synthetic": is_syn, "kfold": args.kfold,
            "overview": overview, "runs": payload}
    html = _DASH.replace("__PAYLOAD__", json.dumps(data))
    (OUT_DIR / "dashboard.html").write_text(html, encoding="utf-8")


# ============================================================================
# 12. REPORT.MD
# ============================================================================
def _report_md(dfm, stamp, is_syn, args):
    lines = ["# Unified benchmark summary\n",
             f"**Data source:** {stamp}\n"]
    if is_syn:
        lines.append("> WARNING: SYNTHETIC DATA - do NOT use in the manuscript.\n")
    lines.append(f"- k-fold: {args.kfold or 'off'}  |  preprocessing: "
                 f"savgol={args.smooth}, snv={args.snv}\n")
    lines.append("\n## Held-out test metrics\n")
    cols = ["run", "model", "R2", "RMSE", "MAE", "RPD", "n"]
    have = [c for c in cols if c in dfm.columns]
    lines.append("| " + " | ".join(have) + " |")
    lines.append("|" + "---|" * len(have))
    for _, row in dfm.iterrows():
        lines.append("| " + " | ".join(
            (f"{row[c]:.3f}" if isinstance(row[c], float) else str(row[c]))
            for c in have) + " |")
    if args.kfold and "cv_R2_mean" in dfm.columns:
        lines.append("\n## Cross-validated (mean +/- sd)\n")
        lines.append("| run | model | R2 | RMSE |")
        lines.append("|---|---|---|---|")
        for _, row in dfm.iterrows():
            if "cv_R2_mean" in row and row.get("cv_R2_mean") == row.get("cv_R2_mean"):
                lines.append(f"| {row['run']} | {row['model']} | "
                             f"{row['cv_R2_mean']:.3f} +/- {row['cv_R2_sd']:.3f} | "
                             f"{row['cv_RMSE_mean']:.3f} +/- {row['cv_RMSE_sd']:.3f} |")
    lines.append("\n_Generated by run_experiment.py v2_\n")
    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


# ============================================================================
# 13. DASHBOARD TEMPLATE (Apache ECharts)
# ============================================================================
_DASH = r"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UV-Vis unified benchmark</title>
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#f5f6f8;color:#1e2430}
 header{background:#0f2540;color:#fff;padding:16px 22px}
 header h1{margin:0;font-size:18px}.sub{font-size:12px;margin-top:5px}
 .warn{background:#b00020;color:#fff;padding:8px 22px;font-weight:600}
 .bar{padding:10px 22px}.bar select{padding:6px;font-size:14px}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px;padding:0 22px 22px}
 .full{grid-column:1/-1}
 .card{background:#fff;border-radius:10px;box-shadow:0 1px 4px rgba(0,0,0,.08);padding:10px}
 .card h2{font-size:14px;margin:4px 8px}.chart{width:100%;height:340px}
 table{border-collapse:collapse;width:100%;font-size:13px}
 th,td{border:1px solid #e3e6ea;padding:6px 8px;text-align:center}th{background:#eef1f5}
 @media(max-width:820px){.grid{grid-template-columns:1fr}}
</style></head><body>
<header><h1>UV-Vis wastewater - unified ML benchmark</h1>
<div class="sub" id="sub"></div></header>
<div id="warn"></div>
<div class="bar">Run: <select id="sel"></select></div>
<div class="grid">
 <div class="card full"><h2>Overview: R2 by run &amp; model</h2><div id="ov" class="chart"></div></div>
 <div class="card"><h2>Predicted vs measured</h2><div id="sc" class="chart"></div></div>
 <div class="card"><h2>Residuals</h2><div id="re" class="chart"></div></div>
 <div class="card full"><h2>Selected run metrics</h2><div id="tb" style="padding:8px"></div></div>
</div>
<script>
const D=__PAYLOAD__;
document.getElementById('sub').textContent='Data source: '+D.stamp+'  |  k-fold: '+(D.kfold||'off');
if(D.synthetic){const w=document.getElementById('warn');w.className='warn';
 w.textContent='SYNTHETIC DATA - NOT FOR PUBLICATION';}
// overview
const ov=echarts.init(document.getElementById('ov'));
ov.setOption({tooltip:{trigger:'axis'},legend:{top:0},
 xAxis:{type:'category',data:D.overview.runs,axisLabel:{interval:0,rotate:18,fontSize:10}},
 yAxis:{type:'value',name:'R2'},
 series:D.overview.models.map(m=>({name:m,type:'bar',data:D.overview.R2[m]}))});
// selector
const sel=document.getElementById('sel');
D.runs.forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.text=r.label;sel.add(o);});
const sc=echarts.init(document.getElementById('sc'));
const re=echarts.init(document.getElementById('re'));
function draw(i){const r=D.runs[i];const ks=Object.keys(r.preds);
 let lo=Math.min(...r.test_y),hi=Math.max(...r.test_y);
 ks.forEach(k=>{lo=Math.min(lo,...r.preds[k]);hi=Math.max(hi,...r.preds[k]);});
 sc.setOption({tooltip:{},legend:{top:0},
  xAxis:{name:'Measured',type:'value',min:lo,max:hi},
  yAxis:{name:'Predicted',type:'value',min:lo,max:hi},
  series:ks.map(k=>({name:(r.models[k]||{}).label||k,type:'scatter',symbolSize:7,
    data:r.test_y.map((v,j)=>[v,r.preds[k][j]])})).concat([
    {name:'1:1',type:'line',showSymbol:false,data:[[lo,lo],[hi,hi]],
     lineStyle:{type:'dashed',color:'#888'}}])},true);
 re.setOption({tooltip:{},legend:{top:0},
  xAxis:{name:'Predicted',type:'value'},yAxis:{name:'Residual',type:'value'},
  series:ks.map(k=>({name:(r.models[k]||{}).label||k,type:'scatter',symbolSize:7,
    data:r.test_y.map((v,j)=>[r.preds[k][j],v-r.preds[k][j]])}))},true);
 let h='<table><tr><th>Model</th><th>R2</th><th>RMSE</th><th>MAE</th><th>RPD</th></tr>';
 for(const k in r.models){const m=r.models[k];
  h+=`<tr><td>${m.label||k}</td><td>${m.R2.toFixed(3)}</td><td>${m.RMSE.toFixed(3)}</td>`+
     `<td>${m.MAE.toFixed(3)}</td><td>${m.RPD.toFixed(2)}</td></tr>`;}
 document.getElementById('tb').innerHTML=h+'</table>';}
sel.onchange=e=>draw(+e.target.value);draw(0);
window.addEventListener('resize',()=>{ov.resize();sc.resize();re.resize();});
</script></body></html>
"""


# ============================================================================
# 14. CLI
# ============================================================================
def build_parser():
    p = argparse.ArgumentParser(
        description="UV-Vis wastewater ML benchmark (1D-CNN vs PLS).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    src = p.add_argument_group("data source (choose one)")
    src.add_argument("--data", help="single CSV/XLSX with spectra + target")
    src.add_argument("--data-dir", help="folder of CSV/XLSX to process")
    src.add_argument("--url", help="download URL (a .zip is auto-extracted)")
    src.add_argument("--spectra-file", help="join mode: spectra file")
    src.add_argument("--labels-file", help="join mode: grab-sample/labels file")
    p.add_argument("--target", help="target column, e.g. ammonium")
    p.add_argument("--targets", help="comma list, e.g. 'ammonium,DOC,phosphate'")
    p.add_argument("--time-col", help="datetime column for join mode")
    p.add_argument("--join-tol", default="5min", help="join tolerance (pandas)")
    p.add_argument("--model", choices=["cnn1d", "mlp"], default="cnn1d")
    p.add_argument("--kfold", type=int, default=0, help="k folds (>=2 to enable)")
    p.add_argument("--cv", choices=["random", "blocked"], default="random",
                   help="random = shuffled KFold (original); blocked = contiguous "
                        "time periods (leave-one-period-out)")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smooth", action="store_true", help="Savitzky-Golay smoothing")
    p.add_argument("--snv", action="store_true", help="SNV normalisation")
    p.add_argument("--tag", help="optional label appended to batch folder name")
    p.add_argument("--no-bootstrap", action="store_true")
    p.add_argument("--spectral-only", action="store_true",
                   help="use only scan_absorbance channels (drops the validity-test column that "
                        "v1.0.1 included; recommended, default off to keep v1.0.1 reproducible)")
    return p


def main():
    args = build_parser().parse_args()
    if os.environ.get(VENV_FLAG) != "1" and not args.no_bootstrap:
        bootstrap_and_reexec(sys.argv[1:]); return
    run_experiment(args)


if __name__ == "__main__":
    main()
