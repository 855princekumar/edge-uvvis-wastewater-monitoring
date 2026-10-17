#!/usr/bin/env python3
"""Records every result to /out/results.jsonl and node metadata to /out/nodes.json."""
import json, os, time
import paho.mqtt.client as mqtt
f = open("/out/results.jsonl", "a"); meta = {}
def on_msg(c, u, m, *a):
    if m.topic.startswith("hello/"):
        meta[m.topic[6:]] = json.loads(m.payload); json.dump(meta, open("/out/nodes.json", "w"), indent=1)
    else:
        d = json.loads(m.payload); d["t_coll"] = time.time(); f.write(json.dumps(d) + "\n"); f.flush()
cl = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="collector") if hasattr(mqtt, "CallbackAPIVersion") else mqtt.Client(client_id="collector")
cl.on_message = on_msg
for i in range(60):
    try: cl.connect(os.environ.get("BROKER", "broker"), 1883, 60); break
    except Exception: time.sleep(1)
cl.subscribe([("results/#", 1), ("hello/#", 1)]); cl.loop_forever()
