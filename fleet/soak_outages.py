#!/usr/bin/env python3
"""Outage analysis of a soak run: for each node, every interval longer than GAP_S without a result
(start time, duration, overlapping broker restart if any), messages delivered late after each gap,
latency outside outage windows, and the physical Pi's CPU use from its vmstat log.
Usage: python soak_outages.py runs/<soak_run> [gap_s]"""
import gzip, json, os, re, sys
import numpy as np, pandas as pd
run = sys.argv[1]; GAP_S = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
R = pd.read_json(gzip.open(os.path.join(run, "results.jsonl.gz"), "rt"), lines=True).drop_duplicates(["node", "seq"])
R["e2e_s"] = R.t_coll - R.t_send
ev = [l.split() for l in open(os.path.join(run, "events.txt"))]
t0 = float(ev[0][0]); restarts = [t0 + float(e[0]) for e in ev if len(e) > 1 and e[1] == "broker_restart_begin"]
sent = json.load(open(os.path.join(run, "replayer.json")))["sent_per_node"]
out = {"sent_per_node": sent, "duration_h": float((R.t_send.max() - R.t_send.min()) / 3600), "broker_restarts": len(restarts), "nodes": {}}
for n, g in R.groupby("node"):
    g = g.sort_values("t_coll"); tc = g.t_coll.to_numpy(); gaps = np.diff(tc); idx = np.where(gaps > GAP_S)[0]
    outs, mask = [], np.zeros(len(g), bool)
    for i in idx:
        a, b = tc[i], tc[i + 1]
        near = [r for r in restarts if a - 60 <= r <= b + 5]
        late = g[(g.t_coll >= b) & (g.t_coll <= b + 120)]
        outs.append({"start_utc": pd.to_datetime(a, unit="s").strftime("%Y-%m-%d %H:%M:%S"), "duration_s": float(b - a),
                     "cause": "broker restart (scheduled)" if near else "network/VPN (unplanned)",
                     "max_e2e_after_s": float(late.e2e_s.max()) if len(late) else None})
        mask |= (g.t_send.to_numpy() >= a - 5) & (g.t_send.to_numpy() <= b + 5)
    ok = g[~mask] if mask.any() else g
    seqs = set(g.seq.astype(int)); missing = sorted(set(range(sent)) - seqs)
    out["nodes"][n] = {"received": int(len(g)), "missing": len(missing), "loss_pct": 100 * len(missing) / sent, "missing_first": missing[:10],
                       "outages_gt_%ds" % GAP_S: outs, "e2e_ms_all": {"median": 1e3 * float(g.e2e_s.median()), "p95": 1e3 * float(g.e2e_s.quantile(.95)), "max": 1e3 * float(g.e2e_s.max())},
                       "e2e_ms_outside_outages": {"median": 1e3 * float(ok.e2e_s.median()), "p95": 1e3 * float(ok.e2e_s.quantile(.95)), "p99": 1e3 * float(ok.e2e_s.quantile(.99))},
                       "pls_ms_median": float(g.inf_ms.median()), "cnn_ms_median": float(g.cnn_ms.median()) if "cnn_ms" in g and g.cnn_ms.notna().any() else None,
                       "peak_rss_mb": float(g.rss_mb.max())}
vm = os.path.join(run, "pi_vmstat.txt")
if os.path.exists(vm):
    rows = [l.split() for l in open(vm) if re.match(r"\s*\d", l)][1:]
    busy = np.array([int(r[12]) + int(r[13]) for r in rows]); free_mb = np.array([int(r[3]) for r in rows]) / 1024
    out["pi_vmstat"] = {"samples_60s": len(rows), "busy_pct_all_cores_median": float(np.median(busy)), "busy_pct_all_cores_p95": float(np.percentile(busy, 95)),
                        "busy_pct_all_cores_max": int(busy.max()), "one_core_equiv_median_pct": 4 * float(np.median(busy)), "free_mem_mb_min": float(free_mb.min())}
json.dump(out, open(os.path.join(run, "soak_outages.json"), "w"), indent=1); print(json.dumps(out, indent=1))
