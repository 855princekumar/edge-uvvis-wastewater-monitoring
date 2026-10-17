#!/usr/bin/env bash
# Microbenchmark (bench_edge.py) inside each emulated profile. Results -> out/bench_*.json
set -euo pipefail; mkdir -p out
python3 - <<'PY' > /tmp/profiles.tsv
import json
for k, v in json.load(open("profiles.json")).items():
    if not k.startswith("_"): print(k, v["platform"], v["mem"], v["cpus"], sep="\t")
PY
while IFS=$'\t' read -r name plat mem cpus; do
  img="edgefleet-node:${plat//\//-}"
  docker build --platform "$plat" -f Dockerfile.node -t "$img" . >/dev/null
  echo ">> $name ($plat, $mem, $cpus cpu)"
  docker run --rm --platform "$plat" --memory "$mem" --memory-swap "$mem" --cpus "$cpus" \
    -v "$PWD/out:/out" "$img" python3 bench_edge.py --repeats 1000 --sustain 20 \
    --label "${name}-emulated" --out "/out/bench_${name}.json" > /dev/null || echo "!! $name failed (record as a result: e.g. OOM)"
done < /tmp/profiles.tsv
