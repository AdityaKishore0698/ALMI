#!/usr/bin/env bash
# Attach to a running ALMI_trans training job and stop it at a loss plateau or a
# wall-clock deadline, right after its next end-of-epoch save.
# Plateau: at least MIN steps, and the mean of the last 60 printed losses (100 steps
# each) is less than 2 % below the mean of the 60 before them.
#
# Usage: bash supervise_run.sh <pid> <log file> <min steps> <deadline as "YYYY-MM-DD HH:MM">
set -uo pipefail
PID=$1; LOG=$2; MIN=$3; DEADLINE=$(date -d "$4" +%s)
say() { echo "[$(date '+%F %T')] $*"; }
plateau() {
  tr '\r' '\n' < "$LOG" | awk -v min=$MIN -v win=60 -v gain=2 '
    /Train. Iter [0-9]+ : Loss\. / { it = $0; sub(/.*Train. Iter /, "", it); it += 0
      v = $0; sub(/.*Loss\. /, "", v); l[n++] = v + 0 }
    END {
      if (it < min || n < 2 * win) exit
      for (i = n - win; i < n; i++) now += l[i]
      for (i = n - 2 * win; i < n - win; i++) before += l[i]
      if (now > before * (1 - gain / 100)) printf "plateau at step %d (%.5f -> %.5f)\n", it, before / win, now / win }'
}
say "supervising pid $PID ($LOG), min $MIN steps, deadline $4"
decided=""
while kill -0 $PID 2>/dev/null; do
  sleep 120
  if [ -z "$decided" ]; then
    res=$(plateau)
    [ -z "$res" ] && [ "$(date +%s)" -ge "$DEADLINE" ] && res="deadline $4"
    [ -n "$res" ] && { decided=$(grep -ac "model last saved" "$LOG"); say "stop ($res); waiting for the next epoch save"; }
  elif [ "$(grep -ac "model last saved" "$LOG")" -gt "$decided" ]; then
    sleep 30; kill $PID; say "stopped after its save"; break
  fi
done
say "done"
