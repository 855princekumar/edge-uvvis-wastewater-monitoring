#!/usr/bin/env bash
# Fleet run with one PHYSICAL Raspberry Pi as an extra node, reached over a VPN. The broker port is
# published on the host; the Pi runs edge_node.py from a throwaway directory (PI_DIR) via SSH
# (PI_SSH = command prefix that runs a remote shell command). Usage: ./run_fleet_with_real_pi.sh <rate> <dur> <name>
set -uo pipefail; export MSYS_NO_PATHCONV=1
RATE=$1; DUR=$2; NAME=$3; NODE=pi4_real_vpn; BROKER_IP=${BROKER_IP:?}; PI_DIR=${PI_DIR:-/tmp/edgebench}
( for i in $(seq 1 60); do docker compose ps --format '{{.Name}} {{.State}}' 2>/dev/null | grep -q "broker-1 running" && break; sleep 2; done
  $PI_SSH "cd $PI_DIR && (setsid nohup vmstat 5 > vmstat_fleet.txt 2>&1 &) && (NODE_ID=$NODE BROKER=$BROKER_IP MODEL=$PI_DIR/pls_nh4.json RUN_CNN=1 setsid nohup venv/bin/python -u edge_node.py > node.log 2>&1 &) && echo pi node started" ) &
EXTRA_NODES=$NODE PUBLISH_BROKER=1 ./run_fleet.sh "$RATE" "$DUR" "$NAME"
wait
$PI_SSH "pkill -f '^venv/bin/python -u edge_node'; pkill -x vmstat; cat $PI_DIR/node.log | grep -v Warning | tail -3; echo stopped"
$PI_SSH "cat $PI_DIR/vmstat_fleet.txt" > "runs/$NAME/pi_vmstat.txt"
