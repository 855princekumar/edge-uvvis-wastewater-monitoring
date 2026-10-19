#!/usr/bin/env python3
"""Emulated edge node: receives spectra over MQTT, runs the exported PLS model (and,
optionally, the NumPy 1D-CNN), publishes the prediction with timestamps.
Env: NODE_ID, BROKER, MODEL=/models/pls_nh4.json, RUN_CNN=0/1, STRESS_THREADS=0"""
import json, os, time, threading, platform, resource
import numpy as np
import paho.mqtt.client as mqtt
from scipy.signal import savgol_filter

NODE = os.environ.get("NODE_ID", "node0"); BROKER = os.environ.get("BROKER", "broker")
M = json.load(open(os.environ.get("MODEL", "/models/pls_nh4.json")))
mean, scale, coef = (np.array(M[k], dtype=np.float64) for k in ("x_mean", "x_scale", "coef"))
icpt = float(M["intercept"])
RUN_CNN = os.environ.get("RUN_CNN", "0") == "1"
if RUN_CNN:
    import bench_edge; cnn, _ = bench_edge.numpy_cnn()

def burn():                                   # optional competing CPU load (e.g. camera pipeline)
    x = 0.0
    while True: x = (x * 1.000001 + 1.0) % 1e6
for _ in range(int(os.environ.get("STRESS_THREADS", "0"))):
    threading.Thread(target=burn, daemon=True).start()

def predict(a):
    x = np.asarray(a, dtype=np.float64)[None]
    if M.get("smooth"): x = savgol_filter(x, 11, 2, axis=1)
    if M.get("snv"): x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-8)
    return (((x - mean) / scale) @ coef + icpt).item()   # .item(): NumPy 2 compatible

def on_msg(cl, _, msg):
    t_recv = time.time(); d = json.loads(msg.payload)
    t0 = time.perf_counter(); y = predict(d["spectrum"]); t_inf = (time.perf_counter() - t0) * 1e3
    out = {"node": NODE, "seq": d["seq"], "t_send": d["t_send"], "t_recv": t_recv,
           "t_done": time.time(), "pred": y, "inf_ms": t_inf,
           "rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0}
    if RUN_CNN:
        t1 = time.perf_counter(); cnn(np.asarray(d["spectrum"], np.float32)[None]); out["cnn_ms"] = (time.perf_counter() - t1) * 1e3
    cl.publish(f"results/{NODE}", json.dumps(out), qos=1)

# persistent session + (re)subscribe in on_connect: survives broker restarts and network drops
cl = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=NODE, clean_session=False) if hasattr(mqtt, "CallbackAPIVersion") else mqtt.Client(client_id=NODE, clean_session=False)
cl.on_message = lambda c, u, m, *a: on_msg(c, u, m)
cl.on_connect = lambda c, *a: (c.subscribe(f"spectra/{NODE}", qos=1),
    c.publish(f"hello/{NODE}", json.dumps({"node": NODE, "arch": platform.machine(), "py": platform.python_version(), "numpy": np.__version__}), qos=1, retain=True))
cl.reconnect_delay_set(1, 10)
for i in range(60):
    try: cl.connect(BROKER, 1883, 60); break
    except Exception: time.sleep(1)
cl.loop_forever()
