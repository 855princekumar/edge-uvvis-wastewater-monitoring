#!/usr/bin/env python3
"""Streams REAL Eawag spectra (replay.npy, exported by export_model.py) to every node at a
fixed rate. Real acquisition is one spectrum per 2 min; RATE_HZ=1 is a 120x stress factor.
Env: NODES=node0,node1,..  RATE_HZ=1  DURATION_S=1800"""
import json, os, time
import numpy as np
import paho.mqtt.client as mqtt
S = np.load("/models/replay.npy"); nodes = os.environ["NODES"].split(",")
rate = float(os.environ.get("RATE_HZ", "1")); dur = float(os.environ.get("DURATION_S", "1800"))
cl = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="replayer") if hasattr(mqtt, "CallbackAPIVersion") else mqtt.Client(client_id="replayer")
for i in range(60):
    try: cl.connect(os.environ.get("BROKER", "broker"), 1883, 60); break
    except Exception: time.sleep(1)
cl.loop_start(); time.sleep(float(os.environ.get("WARMUP_S", "20")))
t_end = time.time() + dur; seq = 0; t_next = time.time()
while time.time() < t_end:
    row = S[seq % len(S)].tolist()
    for n in nodes:
        cl.publish(f"spectra/{n}", json.dumps({"seq": seq, "row": int(seq % len(S)), "t_send": time.time(), "spectrum": row}), qos=1)
    seq += 1; t_next += 1.0 / rate
    time.sleep(max(0.0, t_next - time.time()))
json.dump({"sent_per_node": seq, "rate_hz": rate, "duration_s": dur}, open("/out/replayer.json", "w"))
time.sleep(30); cl.loop_stop()
