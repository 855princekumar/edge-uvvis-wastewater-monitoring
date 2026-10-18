#!/usr/bin/env python3
"""Real-data test of fouling/cleaning signatures in the Eawag UV-Vis record (Fig. S9 was simulated).

Uses the 40 'cleaning' events of 4_sensor_logbook/flume_logbook.csv and the 2-min absorbance at
700 nm (valid rows of flume_scan_absorbance.csv). Analysis fixed in advance:
 (1) Step test: per event, median A700 in the 6 h before the cleaning start minus the 6 h after its end.
     Control: the same windows shifted 24 h earlier (same time of day, no cleaning). Wilcoxon tests.
     Repeated on A700 residuals after regressing out turbidity from the independent Turbimax sensor.
 (2) Drift: slope of daily-median A700 against days since the previous cleaning, per interval >= 2 d.
 (3) Detector rule of the paper applied to real data: alarm when the 24 h trailing median exceeds the
     median of the first 24 h after the previous cleaning by k robust SD (k = 3) of the day-to-day
     changes of the 24 h median over the whole record; per cycle: alarmed?, lead time before cleaning.
Usage (repo root): python validation/drift_cleaning_real.py data/raw/extracted/2_data results/validation_v130
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

root, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
WL = "scan_absorbance7000__m"; H6 = pd.Timedelta(hours=6); K = 3.0
S = pd.read_csv(root / "3_sensor_data/flume_scan_absorbance.csv", usecols=["timestamp_iso", WL, "valid_data"], parse_dates=["timestamp_iso"])
S = S[S.valid_data == 1].dropna().set_index("timestamp_iso").sort_index()[WL]
T = pd.read_csv(root / "3_sensor_data/flume_turbimax_turbidity.csv", usecols=["timestamp_iso", "turbimax_turbidity_fnu", "valid_data"], parse_dates=["timestamp_iso"])
T = T[T.valid_data == 1].dropna().set_index("timestamp_iso").sort_index()["turbimax_turbidity_fnu"]
L = pd.read_csv(root / "4_sensor_logbook/flume_logbook.csv", skiprows=[1])
L["start"] = pd.to_datetime(L["start"]); L["end"] = pd.to_datetime(L["end"].str.strip())
C = L[L.comment.str.strip("' ").str.lower() == "cleaning"].reset_index(drop=True)
# turbidity-corrected residual (log-linear fit on matched timestamps)
J = pd.merge_asof(S.rename("a").to_frame(), T.rename("t").to_frame(), left_index=True, right_index=True, tolerance=pd.Timedelta("5min"), direction="nearest").dropna()
J = J[J.t > 0]; b = np.polyfit(np.log(J.t), J.a, 1); R = (J.a - np.polyval(b, np.log(J.t))).rename("resid")


def step(series, t0, t1):
    pre = series[(series.index >= t0 - H6) & (series.index < t0)]; post = series[(series.index > t1) & (series.index <= t1 + H6)]
    return (pre.median() - post.median()) if len(pre) > 30 and len(post) > 30 else np.nan


res = {"n_cleaning_events": int(len(C)), "wavelength_nm": 700, "turbidity_fit": {"slope_per_log_fnu": float(b[0]), "r": float(np.corrcoef(np.log(J.t), J.a)[0, 1])}}
rows = []
for _, e in C.iterrows():
    rows.append(dict(event=int(e.event), start=str(e.start), step=step(S, e.start, e.end), step_ctrl=step(S, e.start - pd.Timedelta(hours=24), e.end - pd.Timedelta(hours=24)),
                     step_resid=step(R, e.start, e.end), step_resid_ctrl=step(R, e.start - pd.Timedelta(hours=24), e.end - pd.Timedelta(hours=24))))
E = pd.DataFrame(rows)
for col, ctrl in (("step", "step_ctrl"), ("step_resid", "step_resid_ctrl")):
    d = E[[col, ctrl]].dropna()
    res[col] = {"n": int(len(d)), "median": float(d[col].median()), "iqr": [float(d[col].quantile(.25)), float(d[col].quantile(.75))],
                "frac_positive": float((d[col] > 0).mean()), "wilcoxon_vs_zero_p": float(stats.wilcoxon(d[col]).pvalue),
                "control_median": float(d[ctrl].median()), "control_frac_positive": float((d[ctrl] > 0).mean()),
                "wilcoxon_vs_control_p": float(stats.wilcoxon(d[col], d[ctrl]).pvalue)}
# drift between consecutive cleanings
dm = S.resample("1D").median().dropna(); slopes = []
ends = list(C.start) ; starts = list(C.end)
for a, z in zip(starts[:-1], ends[1:]):
    seg = dm[(dm.index > a.normalize()) & (dm.index < z.normalize())]
    if len(seg) >= 2:
        x = (seg.index - a).total_seconds() / 86400
        slopes.append(dict(start=str(a), days=float(x.max()), slope_per_day=float(np.polyfit(x, seg.values, 1)[0]), spearman=float(stats.spearmanr(x, seg.values)[0])))
D = pd.DataFrame(slopes)
res["drift"] = {"n_intervals": int(len(D)), "median_slope_abs_m_per_day": float(D.slope_per_day.median()), "frac_positive_slope": float((D.slope_per_day > 0).mean()),
                "wilcoxon_slope_vs_zero_p": float(stats.wilcoxon(D.slope_per_day).pvalue)}
# detector rule on real data
m24 = S.rolling("24h").median(); dd = m24.resample("1D").last().diff().dropna(); sigma = 1.4826 * np.median(np.abs(dd - dd.median()))
cyc = []
for a, z in zip(starts[:-1], ends[1:]):
    ref = S[(S.index > a) & (S.index <= a + pd.Timedelta(hours=24))].median()
    win = m24[(m24.index > a + pd.Timedelta(hours=24)) & (m24.index < z)]
    if np.isnan(ref) or win.empty: continue
    hit = win[win > ref + K * sigma]
    cyc.append(dict(cycle_start=str(a), cycle_days=float((z - a).total_seconds() / 86400), alarmed=bool(len(hit)),
                    lead_days=float((z - hit.index[0]).total_seconds() / 86400) if len(hit) else None))
Y = pd.DataFrame(cyc)
res["detector"] = {"k_sigma": K, "sigma_abs_m": float(sigma), "n_cycles": int(len(Y)), "frac_alarmed": float(Y.alarmed.mean()),
                   "median_lead_days": float(Y.lead_days.dropna().median()) if Y.alarmed.any() else None,
                   "median_cycle_days": float(Y.cycle_days.median())}
E.to_csv(out / "drift_cleaning_events.csv", index=False); D.to_csv(out / "drift_cleaning_intervals.csv", index=False); Y.to_csv(out / "drift_cleaning_detector.csv", index=False)
json.dump(res, open(out / "drift_cleaning_real.json", "w"), indent=1); print(json.dumps(res, indent=1))
