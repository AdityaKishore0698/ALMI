#!/usr/bin/env bash
# Thermal watchdog for training on cir (the GPU cooling is unreliable).
#
# Every INTERVAL seconds it logs GPU temperature/power/utilisation/memory, CPU
# package temperatures and RAM (with sync, so the last readings survive a hard
# crash). If the GPU reaches PAUSE_AT degrees C, it pauses the training process
# (SIGSTOP) until the GPU has cooled to RESUME_AT, then continues it (SIGCONT).
# Pausing loses no training progress. It exits when the training process exits.
#
# Usage: bash scripts/gpu_watchdog.sh <train-pid> <log-file> [PAUSE_AT] [RESUME_AT]
set -u
PID=$1; LOG=$2
PAUSE_AT=${3:-82}      # GPU target temperature is 84 C, slowdown 91 C, shutdown 94 C
RESUME_AT=${4:-70}
INTERVAL=${INTERVAL:-10}
paused=0

log() { echo "$(date '+%F %T') $*" >> "$LOG"; sync "$LOG" 2>/dev/null; }

log "watchdog start: pid=$PID pause_at=${PAUSE_AT}C resume_at=${RESUME_AT}C"
while kill -0 "$PID" 2>/dev/null; do
  read -r temp power util mem < <(nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,memory.used \
                                    --format=csv,noheader,nounits | tr -d ',')
  cpu=$(sensors 2>/dev/null | awk '/^Package id/{printf "%s ", $4}')
  ram=$(free -g | awk '/^Mem/{print $3"G"}')
  state=$([ $paused -eq 1 ] && echo PAUSED || echo running)
  log "gpu ${temp}C ${power}W ${util}% ${mem}MiB | cpu ${cpu}| ram ${ram} | ${state}"

  if [ $paused -eq 0 ] && [ "${temp:-0}" -ge "$PAUSE_AT" ]; then
    kill -STOP "$PID" && paused=1 && log "PAUSE: GPU ${temp}C >= ${PAUSE_AT}C"
  elif [ $paused -eq 1 ] && [ "${temp:-99}" -le "$RESUME_AT" ]; then
    kill -CONT "$PID" && paused=0 && log "RESUME: GPU ${temp}C <= ${RESUME_AT}C"
  fi
  sleep "$INTERVAL"
done
# never leave the training process stopped
kill -CONT "$PID" 2>/dev/null
log "watchdog exit: training process $PID ended"
