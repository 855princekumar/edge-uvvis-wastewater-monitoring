#!/usr/bin/env bash
# Long soak test with one PHYSICAL Raspberry Pi over a VPN (+ the native x86-64 node as a control).
# Real Eawag spectra at RATE Hz for DUR s; the broker is restarted every RESTART_EVERY s to test
# recovery; the Pi runs edge_node.py from a throwaway directory (PI_DIR) and logs vmstat every 60 s.
# Usage: BROKER_IP=<host VPN IP> PI_SSH="<cmd that runs a remote shell command>" ./run_soak.sh 1 259200 28800 soak_72h
set -uo pipefail; export MSYS_NO_PATHCONV=1
RATE=$1; DUR=$2; RESTART_EVERY=$3; NAME=$4; NODE=pi4_real_vpn; PI_DIR=${PI_DIR:-/tmp/edgebench}; BROKER_IP=${BROKER_IP:?}
rm -rf out; mkdir -p out runs
ONLY_NODES=node_ref_amd64 EXTRA_NODES=$NODE PUBLISH_BROKER=1 python gen_compose.py "$RATE" "$DUR"
docker compose up -d > out/compose_up.log 2>&1 || { echo "compose up failed"; cat out/compose_up.log; exit 1; }
for i in $(seq 1 60); do docker compose ps --format '{{.Name}} {{.State}}' | grep -q "broker-1 running" && break; sleep 2; done
$PI_SSH "cd $PI_DIR && (setsid nohup vmstat 60 > vmstat_soak.txt 2>&1 &) && (NODE_ID=$NODE BROKER=$BROKER_IP MODEL=$PI_DIR/pls_nh4.json RUN_CNN=1 setsid nohup venv/bin/python -u edge_node.py > node.log 2>&1 &) && echo pi node started" > out/pi_start.txt 2>&1
t0=$(date +%s); echo "$t0 start" > out/events.txt; next_restart=$((t0 + 90 + RESTART_EVERY))
echo "ts,container,cpu_pct,mem_usage" > out/docker_stats.csv
while [ ! -f out/replayer.json ]; do
  now=$(date +%s)
  docker stats --no-stream --format '{{.Name}},{{.CPUPerc}},{{.MemUsage}}' 2>/dev/null | sed "s/^/$((now - t0)),/" >> out/docker_stats.csv
  if [ "$now" -ge "$next_restart" ] && [ "$now" -lt $((t0 + 90 + DUR - 600)) ]; then
    echo "$((now - t0)) broker_restart_begin $(date -u +%FT%TZ)" >> out/events.txt
    docker compose restart broker >> out/events.txt 2>&1
    echo "$(( $(date +%s) - t0 )) broker_restart_end" >> out/events.txt; next_restart=$((next_restart + RESTART_EVERY))
  fi
  [ $((now - t0)) -gt $((DUR + 1800)) ] && { echo "timeout" >> out/events.txt; break; }
  sleep 60
done
sleep 60
docker compose ps -a --format '{{.Name}},{{.State}},{{.ExitCode}}' > out/container_states.csv
for c in $(docker compose ps -a -q); do echo "$(docker inspect -f '{{.Name}},{{.State.OOMKilled}},{{.RestartCount}}' $c)"; done > out/oom.csv
docker compose logs --no-color > out/compose_logs.txt 2>&1; docker compose down > /dev/null 2>&1
$PI_SSH "pkill -f '^venv/bin/python -u edge_node'; pkill -x vmstat; grep -v Warning $PI_DIR/node.log | tail -5; echo stopped" > out/pi_stop.txt 2>&1
$PI_SSH "cat $PI_DIR/vmstat_soak.txt" > out/pi_vmstat.txt 2>&1
"${PY:-python}" analyse_fleet.py > out/analyse.txt 2>&1; "${PY:-python}" analyse_extra.py out > out/analyse_extra.txt 2>&1
rm -rf "runs/$NAME"; cp -r out "runs/$NAME"; gzip -f "runs/$NAME/results.jsonl"; echo "archived to runs/$NAME"
