#!/usr/bin/env bash
# Same as run_bench_matrix.sh, but Git-Bash/Windows-safe (no path mangling), records each
# profile's exit code in out/bench_status.tsv (a failure/OOM is a result), and for profiles
# with "stress" > 0 runs a co-located CPU-burning process inside the same cgroup.
set -uo pipefail; mkdir -p out; export MSYS_NO_PATHCONV=1
HOSTOUT="$(pwd -W 2>/dev/null || pwd)/out"
REPEATS="${REPEATS:-1000}"; SUSTAIN="${SUSTAIN:-20}"
python - <<'PY' > out/profiles.tsv
import json
for k, v in json.load(open("profiles.json")).items():
    if not k.startswith("_"): print(k, v["platform"], v["mem"], v["cpus"], v.get("stress", 0), sep="\t")
PY
[ -f out/bench_status.tsv ] || printf "profile\tplatform\tmem\tcpus\tstress\texit_code\toom_killed\tseconds\trepeats\n" > out/bench_status.tsv
while IFS=$'\t' read -r name plat mem cpus stress; do
  name="${name%$'\r'}"; stress="${stress%$'\r'}"
  case " ${SKIP:-} " in *" $name "*) continue;; esac
  img="edgefleet-node:${plat//\//-}"
  burn=""; [ "$stress" != "0" ] && burn="python3 -c 'x=0.0
while True: x=(x*1.000001+1.0)%1e6' & "
  echo ">> $name ($plat, $mem, $cpus cpu, stress=$stress)"; t0=$(date +%s)
  docker run --name "bench_$name" --platform "$plat" --memory "$mem" --memory-swap "$mem" --cpus "$cpus" \
    -v "$HOSTOUT:/out" "$img" sh -c "${burn}python3 bench_edge.py --repeats $REPEATS --sustain $SUSTAIN --label ${name}-emulated-repeats${REPEATS} --out /out/bench_${name}.json" > "out/bench_${name}.log" 2>&1
  rc=$?; oom=$(docker inspect -f '{{.State.OOMKilled}}' "bench_$name" 2>/dev/null); docker rm "bench_$name" >/dev/null 2>&1
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$name" "$plat" "$mem" "$cpus" "$stress" "$rc" "$oom" "$(( $(date +%s) - t0 ))" "$REPEATS" >> out/bench_status.tsv
  [ $rc -ne 0 ] && echo "!! $name exit $rc oom=$oom (recorded as a result)"
done < out/profiles.tsv
cat out/bench_status.tsv
