#!/usr/bin/env python3
import json, os, sys
P = {k: v for k, v in json.load(open("profiles.json")).items() if not k.startswith("_")}
if os.environ.get("ONLY_NODES"):   # optional subset of emulated profiles, e.g. for a long soak test
    P = {k: v for k, v in P.items() if k in os.environ["ONLY_NODES"].split(",")}
rate = sys.argv[1] if len(sys.argv) > 1 else "1"; dur = sys.argv[2] if len(sys.argv) > 2 else "1800"
# optional: EXTRA_NODES=a,b (external/physical nodes that subscribe over the network), PUBLISH_BROKER=1
extra = [n for n in os.environ.get("EXTRA_NODES", "").split(",") if n]
L = ["services:", "  broker:", "    image: eclipse-mosquitto:2", "    volumes: [./mosquitto.conf:/mosquitto/config/mosquitto.conf:ro]"]
if os.environ.get("PUBLISH_BROKER") == "1": L += ["    ports: ['1883:1883']"]
L += [
     "  collector:", "    build: {context: ., dockerfile: Dockerfile.node}", "    platform: linux/amd64",
     "    command: python3 -u collector.py", "    volumes: [./collector.py:/app/collector.py:ro, ./out:/out]", "    depends_on: [broker]"]
for n, p in P.items():
    L += [f"  {n}:", f"    build: {{context: ., dockerfile: Dockerfile.node}}", f"    image: edgefleet-node:{p['platform'].replace('/', '-')}",
          f"    platform: {p['platform']}", f"    mem_limit: {p['mem']}", f"    memswap_limit: {p['mem']}", f"    cpus: {p['cpus']}",
          f"    environment: ['NODE_ID={n}', 'BROKER=broker', 'MODEL=/models/pls_nh4.json', 'RUN_CNN={p['cnn']}', 'STRESS_THREADS={p['stress']}']",
          "    volumes: [./models:/models:ro]", "    depends_on: [broker]"]
L += ["  replayer:", "    build: {context: ., dockerfile: Dockerfile.node}", "    platform: linux/amd64",
      "    command: python3 -u replayer.py",
      f"    environment: ['BROKER=broker', 'NODES={','.join(list(P) + extra)}', 'RATE_HZ={rate}', 'DURATION_S={dur}', 'WARMUP_S=90']",
      "    volumes: [./replayer.py:/app/replayer.py:ro, ./models:/models:ro, ./out:/out]", f"    depends_on: [broker, collector, {', '.join(P)}]"]
open("docker-compose.yml", "w").write("\n".join(L) + "\n"); print("docker-compose.yml written for", len(P), "nodes")
