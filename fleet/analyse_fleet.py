#!/usr/bin/env python3
"""Per-node: messages sent/received, loss %, end-to-end latency (median, p95, max),
inference ms, CNN ms, peak RSS, max |pred - reference| (cross-architecture agreement)."""
import json, numpy as np, pandas as pd
R = pd.read_json("out/results.jsonl", lines=True); sent = json.load(open("out/replayer.json"))["sent_per_node"]
ref = np.load("models/replay_ref_pred.npy"); prof = {k: v for k, v in json.load(open("profiles.json")).items() if not k.startswith("_")}
nodes = json.load(open("out/nodes.json"))
R["e2e_ms"] = (R.t_coll - R.t_send) * 1e3; R["ref"] = ref[R.seq % len(ref)]
rows = []
for n, g in R.groupby("node"):
    g = g.drop_duplicates("seq")
    pr = prof.get(n, {"analogue": "physical node (external)", "platform": "-", "mem": "-", "cpus": "-"})
    rows.append(dict(node=n, analogue=pr["analogue"], platform=pr["platform"], mem=pr["mem"], cpus=pr["cpus"],
        arch=nodes.get(n, {}).get("arch"), sent=sent, received=len(g), loss_pct=100 * (1 - len(g) / sent),
        e2e_median_ms=g.e2e_ms.median(), e2e_p95_ms=g.e2e_ms.quantile(.95), e2e_max_ms=g.e2e_ms.max(),
        pls_inf_median_ms=g.inf_ms.median(), cnn_inf_median_ms=g.cnn_ms.median() if "cnn_ms" in g and g.cnn_ms.notna().any() else None,
        peak_rss_mb=g.rss_mb.max(), max_abs_dev_vs_ref=float((g.pred - g.ref).abs().max())))
T = pd.DataFrame(rows); T.to_csv("out/fleet_summary.csv", index=False); print(T.to_string())
