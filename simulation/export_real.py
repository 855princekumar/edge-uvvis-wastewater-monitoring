#!/usr/bin/env python3
"""Exports the 510 time-ordered real Eawag spectra matched to grab samples (raw abs/m, valid
rows only) for robust_realmatrix.py. The output is derived from the CC-BY dataset and is
NOT committed. Usage (repo root): python simulation/export_real.py data/raw/extracted out.npy"""
import argparse, logging, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_experiment as RE
_orig = RE._abs_wavelength
RE._abs_wavelength = lambda c: None if 'validity' in str(c).lower() else _orig(c)   # spectral columns only

args = argparse.Namespace(targets="nh4", target=None, join_tol="5min", smooth=False, snv=False)
r = RE.load_eawag(Path(sys.argv[1]), args, logging.getLogger("export"))[0]
t = r["t"]; assert np.all(np.diff(t).astype(float) >= 0)
np.save(sys.argv[2], np.asarray(r["X"], dtype=float)); np.save(sys.argv[2].replace('.npy', '_wl.npy'), np.asarray(r["wl"], float))
print("saved", np.asarray(r["X"]).shape)
