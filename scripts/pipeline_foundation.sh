#!/usr/bin/env bash
# Foundation models at the paper's batch sizes, back to back on cir:
#   1. attach to a running CL-400sl job (effective batch 64 via gradient accumulation)
#      and stop it at a loss plateau or a wall-clock cap, right after an epoch save
#   2. retrain OL at the paper's batch 128 (64 x 2 accumulated) on the trained VQ-VAE
# A stage stops when it has at least MIN optimizer steps and the mean of the last 60
# printed losses (100 steps each) is less than 2 % below the 60 before them, or at the cap.
#
# Usage: bash pipeline_foundation.sh <cl400sl-pid> <cl400sl-run> <vq-run> <tag>
set -uo pipefail
C_PID=$1; C_RUN=$2; VQ_RUN=$3; TAG=$4
C_MIN=${C_MIN:-5000};  C_MAXSEC=${C_MAXSEC:-$((5*3600))}
O_MIN=${O_MIN:-20000}; O_MAXSEC=${O_MAXSEC:-$((6*3600))}
LOGS=$HOME/ALMI/logs
TR=$HOME/ALMI/ALMI-Open/ALMI_trans
WATCHDOG=${WATCHDOG:-$HOME/ALMI/bin/gpu_watchdog.sh}
say() { echo "[$(date '+%F %T')] $*"; }
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate almi-trans
cd "$TR"

plateau() {  # $1 log  $2 min steps -> prints a reason when the loss has flattened
  tr '\r' '\n' < "$1" | awk -v min=$2 -v win=60 -v gain=2 '
    /Train. Iter [0-9]+ : Loss\. / { it = $0; sub(/.*Train. Iter /, "", it); it += 0
      v = $0; sub(/.*Loss\. /, "", v); l[n++] = v + 0 }
    END {
      if (it < min || n < 2 * win) exit
      for (i = n - win; i < n; i++) now += l[i]
      for (i = n - 2 * win; i < n - win; i++) before += l[i]
      if (now > before * (1 - gain / 100)) printf "plateau at step %d (%.5f -> %.5f)\n", it, before / win, now / win }'
}

supervise() {  # $1 pid  $2 run  $3 min steps  $4 max seconds  $5 start time (epoch s)
  local pid=$1 run=$2 min=$3 maxsec=$4 t0=$5 decided="" res log=$LOGS/$2.log
  while kill -0 $pid 2>/dev/null; do
    sleep 120
    if [ -z "$decided" ]; then
      res=$(plateau "$log" $min)
      [ -z "$res" ] && [ $(( $(date +%s) - t0 )) -ge $maxsec ] && res="wall-clock cap"
      [ -n "$res" ] && { decided=$(grep -ac "model last saved" "$log"); say "$run: stop ($res); waiting for the next epoch save"; }
    elif [ "$(grep -ac "model last saved" "$log")" -gt "$decided" ]; then
      sleep 30; kill $pid; say "$run: stopped after its save"; break
    fi
  done
  while kill -0 $pid 2>/dev/null; do sleep 5; done
}

# ---- 1. CL-400sl (already running) ----
say "attached to $C_RUN (pid $C_PID)"
supervise $C_PID $C_RUN $C_MIN $C_MAXSEC $(stat -c %Y "$LOGS/$C_RUN.pid")
say "CL-400sl model: $TR/output/$C_RUN/almi_trans_cl_400sl_last.pth"

# ---- 2. OL at the paper's batch 128 = 64 x 2 ----
R=ol_${TAG}_b128acc
export ALMI_SAVE_EVERY_EPOCHS=1 ALMI_ACCUM_STEPS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_almi_ol.py --num-obs 71 --exp-name $R --batch-size 64 --num-layers 9 --embed-dim-gpt 256 \
  --n-head-gpt 16 --seq-length 20 --ff-rate 4 --drop-out-rate 0.1 --out-dir output --total-epoch 10000 \
  --lr-scheduler 150000 --lr 0.0001 --dataname almi --dilation-growth-rate 3 --device cuda:0 \
  --vq-name $VQ_RUN --resume-pth output/$VQ_RUN/vq_net_last.pth --nb-code 1024 --vq-act relu --down-t 2 \
  --block-size 176 --max-motion-len 700 > "$LOGS/$R.log" 2>&1 < /dev/null &
PID=$!; echo $PID > "$LOGS/$R.pid"
bash "$WATCHDOG" $PID "$LOGS/watchdog_$R.log" 88 82 &
say "launched $R pid=$PID (effective batch 128)"
supervise $PID $R $O_MIN $O_MAXSEC $(date +%s)
say "OL model: $TR/output/$R/almi_trans_ol_last.pth"
say "ALL DONE"
