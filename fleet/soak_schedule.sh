#!/usr/bin/env bash
# Side scheduler for a running soak (run_soak.sh): adds broker restarts at the given offsets (hours from
# the soak start in out/events.txt) and ends the soak cleanly at STOP_H: the replayer is stopped, in-flight
# messages drain for 90 s, and out/replayer.json is written with sent_per_node = highest sequence number
# published + 1 (read from the collector's results), which lets run_soak.sh continue its normal teardown.
# Usage: ./soak_schedule.sh "4 12 20" 23.9
set -uo pipefail; export MSYS_NO_PATHCONV=1
EXTRA_H=$1; STOP_H=$2
t0=$(head -1 out/events.txt | cut -d' ' -f1)
for h in $EXTRA_H; do
  at=$(python -c "print(int($t0 + $h * 3600))")
  while [ "$(date +%s)" -lt "$at" ]; do sleep 30; done
  now=$(date +%s); echo "$((now - t0)) broker_restart_begin $(date -u +%FT%TZ) (scheduled_extra)" >> out/events.txt
  docker compose restart broker >> out/events.txt 2>&1
  echo "$(( $(date +%s) - t0 )) broker_restart_end" >> out/events.txt
done
stop=$(python -c "print(int($t0 + $STOP_H * 3600))")
while [ "$(date +%s)" -lt "$stop" ]; do sleep 30; done
echo "$(( $(date +%s) - t0 )) replay_stop (soak shortened to ${STOP_H} h)" >> out/events.txt
docker compose stop replayer >> out/events.txt 2>&1
sleep 90
python - <<'PY'
import json
seq = -1
for line in open("out/results.jsonl"):
    try: seq = max(seq, int(json.loads(line)["seq"]))
    except Exception: pass
json.dump({"sent_per_node": seq + 1, "rate_hz": 1.0, "note": "soak stopped early by soak_schedule.sh; sent = highest sequence number received + 1"}, open("out/replayer.json", "w"))
print("replayer.json written", seq + 1)
PY
