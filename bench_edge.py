#!/usr/bin/env python3
"""
bench_edge.py - inference cost of the ACTUAL models used in the manuscript
(PLS control, 12 latent variables; compact 1D-CNN, identical to build_cnn in
run_experiment.py) on the device it runs on (Raspberry Pi 4/5, Jetson, laptop).

Latency, memory and model size do not depend on the spectral values, so the
models are fitted on random data of the real input shape (215 channels).
Accuracy is NOT measured here; it comes from run_experiment.py.

Usage:  python3 bench_edge.py --repeats 2000 --out bench_<device>.json
Optional: read average power from a USB-C meter during the "sustained" phase
and pass it with --watts to get energy per inference.
"""
import argparse, json, os, platform, resource, time, statistics, io, pickle

import numpy as np

N_WL = 215   # scan_absorbance channels 200-735 nm (v1.1.0 used 216 incl. one validity column)


def device_info():
    info = {"machine": platform.machine(), "python": platform.python_version(),
            "system": platform.platform()}
    try:
        with open("/proc/device-tree/model") as f:
            info["model"] = f.read().strip("\x00\n ")
    except OSError:
        info["model"] = platform.processor() or "unknown"
    info["cpu_count"] = os.cpu_count()
    return info


def rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 1024.0 if platform.system() == "Linux" else r / 1024.0 / 1024.0


def time_it(fn, x, repeats, warmup=50):
    for _ in range(warmup):
        fn(x)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(x)
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {"median_ms": statistics.median(ts),
            "p95_ms": ts[int(0.95 * len(ts)) - 1],
            "mean_ms": statistics.fmean(ts)}


def cpu_times():
    t = os.times()
    return t.user + t.system


def bench_pls(repeats):
    from sklearn.cross_decomposition import PLSRegression
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(408, N_WL)), rng.normal(size=408)
    m = PLSRegression(n_components=12).fit(X, y)
    x1 = X[:1]
    res = time_it(lambda a: m.predict(a), x1, repeats)
    # deployable form: centred input x coef (a single dot product)
    coef = np.asarray(m.coef_).ravel()
    res["n_parameters"] = int(coef.size + 2 * N_WL + 1)  # coef + x mean/scale + intercept
    res["serialized_kb"] = len(pickle.dumps(m)) / 1024.0
    res["flops_per_spectrum"] = 2 * N_WL * 2  # centring/scaling + dot product (approx.)
    return res


def bench_cnn(repeats):
    try:
        import torch, torch.nn as nn
    except ImportError:
        return {"skipped": "torch not installed (pip install torch)"}
    torch.set_num_threads(1)
    torch.manual_seed(42)
    net = nn.Sequential(
        nn.Conv1d(1, 16, 7, padding=3), nn.ReLU(), nn.BatchNorm1d(16), nn.MaxPool1d(2),
        nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.BatchNorm1d(32), nn.MaxPool1d(2),
        nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        nn.Flatten(), nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)).eval()
    x1 = torch.randn(1, 1, N_WL)
    with torch.no_grad():
        res = time_it(lambda a: net(a), x1, repeats)
    buf = io.BytesIO(); torch.save(net.state_dict(), buf)
    res["n_parameters"] = int(sum(p.numel() for p in net.parameters()))
    res["serialized_kb"] = buf.tell() / 1024.0
    # multiply-accumulates x2, conv + linear layers
    L1, L2, L3 = N_WL, N_WL // 2, N_WL // 4
    macs = L1 * 16 * 7 + L2 * 32 * 16 * 5 + L3 * 64 * 32 * 3 + 64 * 32 + 32
    res["flops_per_spectrum"] = int(2 * macs)
    res["torch_threads"] = 1
    return res


def numpy_cnn():
    """Same architecture/weights shape as build_cnn (Table 5), inference only, pure NumPy.
    Runs on any CPU architecture (incl. 32-bit ARM, where PyTorch wheels do not exist)."""
    rng = np.random.default_rng(42)
    W = {"c1": rng.normal(0, .1, (16, 1, 7)), "b1": np.zeros(16), "g1": np.ones(16), "s1": np.zeros(16),
         "c2": rng.normal(0, .1, (32, 16, 5)), "b2": np.zeros(32), "g2": np.ones(32), "s2": np.zeros(32),
         "c3": rng.normal(0, .1, (64, 32, 3)), "b3": np.zeros(64),
         "l1": rng.normal(0, .1, (32, 64)), "lb1": np.zeros(32), "l2": rng.normal(0, .1, (1, 32)), "lb2": np.zeros(1)}
    W = {k: v.astype(np.float32) for k, v in W.items()}

    def conv(x, w, b):                       # x: (C_in, L), same padding
        k = w.shape[2]; p = k // 2
        xp = np.pad(x, ((0, 0), (p, p)))
        cols = np.lib.stride_tricks.sliding_window_view(xp, k, axis=1)   # (C_in, L, k)
        return np.einsum("clk,ock->ol", cols, w, optimize=True) + b[:, None]

    def fwd(x):
        h = np.maximum(conv(x, W["c1"], W["b1"]), 0) * W["g1"][:, None] + W["s1"][:, None]
        h = h[:, :h.shape[1] // 2 * 2].reshape(16, -1, 2).max(2)
        h = np.maximum(conv(h, W["c2"], W["b2"]), 0) * W["g2"][:, None] + W["s2"][:, None]
        h = h[:, :h.shape[1] // 2 * 2].reshape(32, -1, 2).max(2)
        h = np.maximum(conv(h, W["c3"], W["b3"]), 0).mean(1)
        h = np.maximum(W["l1"] @ h + W["lb1"], 0)
        return W["l2"] @ h + W["lb2"]
    n = sum(v.size for k, v in W.items() if k not in ("s1", "s2")) - 0
    fwd.weights = W
    return fwd, n


def bench_cnn_numpy(repeats):
    fwd, _ = numpy_cnn()
    x = np.random.default_rng(0).normal(size=(1, N_WL)).astype(np.float32)
    res = time_it(fwd, x, repeats)
    res["n_parameters"] = 11137
    res["serialized_kb"] = len(pickle.dumps(fwd.weights)) / 1024.0  # float32 weight arrays
    res["note"] = "pure NumPy forward pass, architecture of Table 5, float32"
    return res


def sustained(seconds, fn):
    n, t0, c0 = 0, time.perf_counter(), cpu_times()
    while time.perf_counter() - t0 < seconds:
        fn(); n += 1
    wall = time.perf_counter() - t0
    return {"inferences": n, "per_s": n / wall,
            "cpu_util_pct_one_core": 100.0 * (cpu_times() - c0) / wall}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2000)
    ap.add_argument("--sustain", type=float, default=30.0)
    ap.add_argument("--watts", type=float, help="avg power from external meter (W)")
    ap.add_argument("--out", default="bench_edge.json")
    ap.add_argument("--label", default="", help="e.g. 'arm64-emulated-1g-2cpu' or 'pi4-real'")
    a = ap.parse_args()

    base = rss_mb()
    out = {"device": device_info(), "label": a.label, "input_channels": N_WL, "baseline_rss_mb": base,
           "numpy": np.__version__}
    try:
        with open("/sys/fs/cgroup/memory.max") as f: out["cgroup_memory_max"] = f.read().strip()
        with open("/sys/fs/cgroup/cpu.max") as f: out["cgroup_cpu_max"] = f.read().strip()
    except OSError:
        pass
    out["PLS"] = bench_pls(a.repeats)
    out["CNN1D_torch"] = bench_cnn(a.repeats)
    out["CNN1D_numpy"] = bench_cnn_numpy(a.repeats)
    out["peak_rss_mb"] = rss_mb()

    from sklearn.cross_decomposition import PLSRegression
    rng = np.random.default_rng(1)
    m = PLSRegression(n_components=12).fit(rng.normal(size=(408, N_WL)), rng.normal(size=408))
    x = rng.normal(size=(1, N_WL))
    out["PLS"]["sustained"] = sustained(a.sustain, lambda: m.predict(x))
    fwd, _ = numpy_cnn(); xc = rng.normal(size=(1, N_WL)).astype(np.float32)
    out["CNN1D_numpy"]["sustained"] = sustained(a.sustain, lambda: fwd(xc))
    if a.watts:
        out["PLS"]["sustained"]["avg_watts"] = a.watts
        out["PLS"]["sustained"]["mJ_per_inference"] = 1e3 * a.watts / out["PLS"]["sustained"]["per_s"]
    print(json.dumps(out, indent=2))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
