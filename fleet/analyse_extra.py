#!/usr/bin/env python3
"""Extra fleet metrics for one archived run (EMULATED):
  - container CPU % (docker stats samples during the replay window) per node: median, max
  - for a broker-restart run: per node, sequence numbers lost, the time between the restart
    and the first result after it, and whether results resumed.
Usage: python analyse_extra.py runs/<run>   -> runs/<run>/extra_summary.csv"""
import json, sys
import numpy as np, pandas as pd
run = sys.argv[1]
R = pd.read_json(f"{run}/results.jsonl", lines=True).drop_duplicates(["node", "seq"])
sent = json.load(open(f"{run}/replayer.json"))["sent_per_node"]
S = pd.read_csv(f"{run}/docker_stats.csv")
S["cpu"] = S.cpu_pct.str.rstrip("%").astype(float)
S["node"] = S.container.str.replace(r"^fleet-(.*)-1$", r"\1", regex=True)
# replay window = samples after the replayer warm-up; skip the first 120 s of samples (start-up/imports)
S = S[S.ts >= 120]
restart = None
try:
    open(f"{run}/broker_restart.txt").read()
    restart = True
except OSError:
    pass
rows = []
for n, g in R.groupby("node"):
    c = S[S.node == n].cpu
    row = dict(node=n, sent=sent, received=len(g), lost=sent - len(g), loss_pct=100 * (1 - len(g) / sent),
               cpu_median_pct=float(c.median()) if len(c) else None, cpu_max_pct=float(c.max()) if len(c) else None)
    if restart:
        g = g.sort_values("t_send")
        gaps = np.diff(g.t_coll.to_numpy())
        i = int(np.argmax(gaps))
        row.update(max_result_gap_s=float(gaps[i]),
                   seq_before_gap=int(g.seq.iloc[i]), seq_after_gap=int(g.seq.iloc[i + 1]),
                   resumed=bool(g.seq.max() > g.seq.iloc[i + 1]))
    rows.append(row)
for n in ["broker", "collector", "replayer"]:
    c = S[S.node == n].cpu
    if len(c): rows.append(dict(node=n, cpu_median_pct=float(c.median()), cpu_max_pct=float(c.max())))
T = pd.DataFrame(rows); T.to_csv(f"{run}/extra_summary.csv", index=False); print(T.to_string())
