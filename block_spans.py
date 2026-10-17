import argparse, logging, sys, csv
from pathlib import Path
import numpy as np
sys.path.insert(0, ".")
import run_experiment as RE
from sklearn.model_selection import KFold
args = argparse.Namespace(targets="nh4,po4,doc", target=None, join_tol="5min", smooth=True, snv=True)
log = logging.getLogger("x"); logging.basicConfig(level=logging.WARNING)
runs = RE.load_eawag(Path("data/raw/extracted"), args, log)
out = []
for r in runs:
    t = r["t"]; assert np.all(np.diff(t).astype(float) >= 0), "not time-sorted"
    for i, (_, va) in enumerate(KFold(5, shuffle=False).split(r["X"])):
        out.append([r["target"], i + 1, len(va), str(t[va[0]])[:10], str(t[va[-1]])[:10]])
w = csv.writer(open("results/batch-4_smoothsnv_blocked/block_date_spans.csv", "w", newline=""))
w.writerow(["target", "block", "n_val", "start", "end"]); w.writerows(out)
for o in out: print(o)
