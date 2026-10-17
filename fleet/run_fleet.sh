#!/usr/bin/env bash
# Usage: ./run_fleet.sh <rate_hz> <duration_s> <outdir_name> [broker_restart_at_s]
# Runs the emulated fleet end to end and archives everything to runs/<outdir_name>/.
set -uo pipefail; export MSYS_NO_PATHCONV=1
RATE=$1; DUR=$2; NAME=$3; RESTART_AT=${4:-}
rm -rf out; mkdir -p out runs
python gen_compose.py "$RATE" "$DUR"
docker compose up -d ${COMPOSE_BUILD:-} > out/compose_up.log 2>&1 || { echo "compose up failed"; cat out/compose_up.log; }
t0=$(date +%s); echo "ts,container,cpu_pct,mem_usage" > out/docker_stats.csv; restarted=0
while [ ! -f out/replayer.json ]; do
  now=$(( $(date +%s) - t0 ))
  docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' | sed "s/^/$now,/" >> out/docker_stats.csv
  if [ -n "$RESTART_AT" ] && [ $restarted -eq 0 ] && [ $now -ge "$RESTART_AT" ]; then
    echo "$now broker restart" > out/broker_restart.txt; docker compose restart broker >> out/broker_restart.txt 2>&1
    echo "$(( $(date +%s) - t0 )) broker back" >> out/broker_restart.txt; restarted=1
  fi
  [ $now -gt $(( DUR + 600 )) ] && { echo "timeout waiting for replayer.json"; break; }
  sleep 15
done
sleep 30
docker compose ps -a --format '{{.Name}},{{.State}},{{.ExitCode}}' > out/container_states.csv
for c in $(docker compose ps -a -q); do echo "$(docker inspect -f '{{.Name}},{{.State.OOMKilled}},{{.RestartCount}}' $c)"; done > out/oom.csv
docker compose logs --no-color > out/compose_logs.txt 2>&1
docker compose down > /dev/null 2>&1
"${PY:-python}" analyse_fleet.py > out/analyse.txt 2>&1; cat out/analyse.txt
rm -rf "runs/$NAME"; cp -r out "runs/$NAME"; echo "archived to runs/$NAME"
