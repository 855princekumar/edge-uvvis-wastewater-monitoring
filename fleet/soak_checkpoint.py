#!/usr/bin/env python3
"""Interim/final summary of a soak run (works on a live out/ folder or an archived run).
Per node: received, expected (from elapsed replay time), missing sequence numbers, e2e latency
median/p95/p99/max, PLS ms; broker restarts and per-restart recovery (longest result gap around it).
Usage: python soak_checkpoint.py <run_dir> [rate_hz]"""
import gzip, json, os, sys
import numpy as np, pandas as pd
run = sys.argv[1]; rate = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
f = os.path.join(run, "results.jsonl"); f = f if os.path.exists(f) else f + ".gz"
op = gzip.open if f.endswith(".gz") else open
R = pd.read_json(op(f, "rt"), lines=True)
R = R.drop_duplicates(["node", "seq"]); R["e2e_ms"] = (R.t_coll - R.t_send) * 1e3
ev = [l.split() for l in open(os.path.join(run, "events.txt"))] if os.path.exists(os.path.join(run, "events.txt")) else []
restarts = [float(e[0]) for e in ev if len(e) > 1 and e[1] == "broker_restart_begin"]
t0 = float(ev[0][0]) if ev else R.t_send.min()
sent_file = os.path.join(run, "replayer.json")
sent = json.load(open(sent_file))["sent_per_node"] if os.path.exists(sent_file) else None
out = {"elapsed_h": float((R.t_send.max() - R.t_send.min()) / 3600), "broker_restarts": len(restarts), "nodes": {}}
for n, g in R.groupby("node"):
    seqmax = int(g.seq.max()); exp = sent if sent is not None else seqmax + 1
    missing = sorted(set(range(exp)) - set(g.seq.astype(int)))
    gs = g.sort_values("t_coll"); gaps = np.diff(gs.t_coll.to_numpy())
    rec = []
    for r in restarts:
        tr = t0 + r; w = gs[(gs.t_coll > tr - 120) & (gs.t_coll < tr + 600)].t_coll.to_numpy()
        rec.append(float(np.max(np.diff(w))) if len(w) > 1 else None)
    out["nodes"][n] = {"received": int(len(g)), "expected": int(exp), "missing": len(missing), "loss_pct": 100 * len(missing) / max(exp, 1),
                       "missing_first": missing[:10], "e2e_median_ms": float(g.e2e_ms.median()), "e2e_p95_ms": float(g.e2e_ms.quantile(.95)),
                       "e2e_p99_ms": float(g.e2e_ms.quantile(.99)), "e2e_max_ms": float(g.e2e_ms.max()), "pls_ms_median": float(g.inf_ms.median()),
                       "longest_result_gap_s": float(gaps.max()) if len(gaps) else None, "gap_around_each_restart_s": rec}
json.dump(out, open(os.path.join(run, "soak_summary.json"), "w"), indent=1); print(json.dumps(out, indent=1))
