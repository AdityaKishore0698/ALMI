#!/usr/bin/env bash
# Unattended GPU queue on cir, one stage straight after another:
#   B  continue the wide-command lower-3 for B_ITERS more iterations (the plateau rule
#      stopped it early; the aim is to recover easy-level performance), then evaluate it
#   A1 foundation model CL-400sl
#   A2 VQ-VAE tokenizer for the open-loop model
#   A3 open-loop model OL on the VQ-VAE tokens
# A stages stop at a loss plateau (at least MIN iterations, < GAIN % improvement between
# two windows of WIN printed values), an iteration cap, or a wall-clock cap, always
# right after an end-of-epoch save (ALMI_SAVE_EVERY_EPOCHS=1).
#
# Usage: bash pipeline_b_then_a.sh <sha>
#        ATTACH_B_PID=<pid> ATTACH_B_RUN=<run> bash pipeline_b_then_a.sh <sha>   (take over a running B)
set -uo pipefail
SHA=$1
B_ITERS=${B_ITERS:-2500}
LOGS=$HOME/ALMI/logs
REPO=$HOME/ALMI/ALMI-Open
RL=$REPO/ALMI_RL
TR=$REPO/ALMI_trans
EVAL=$HOME/ALMI/data/eval
RESULTS=${RESULTS:-$HOME/ALMI/results/20260929}
WATCHDOG=${WATCHDOG:-$HOME/ALMI/bin/gpu_watchdog.sh}
PAUSE_AT=${PAUSE_AT:-88}; RESUME_AT=${RESUME_AT:-82}
DAY=$(date +%Y%m%d)
say() { echo "[$(date '+%F %T')] $*"; }
source "$HOME/anaconda3/etc/profile.d/conda.sh"
latest_ckpt() { local f; f=$(ls "$1" 2>/dev/null | grep "^model_[0-9]*\.pt$" | sort -t_ -k2 -n | tail -1); [ -n "$f" ] && echo "$1/$f"; }

# ================= B: longer wide-command lower-3 =================
conda activate almi-rl
export WANDB_MODE=offline
cd "$RL"
source "$LOGS/almi_final_policies.txt"   # upper2, lower3wide, ...
if [ -n "${ATTACH_B_PID:-}" ]; then   # take over a B run started by an earlier copy of this script
  PID=$ATTACH_B_PID; R=$ATTACH_B_RUN
  say "B: attached to $R pid=$PID"
  while kill -0 $PID 2>/dev/null; do sleep 30; done
else
  R=lower3wide_long_${DAY}_${SHA}
  f=$(basename "$lower3wide")
  ALMI_UPPER_POLICY=$upper2 ALMI_INIT_ARM_WEIGHT=1.0 ALMI_MAX_CMD_X=1.3 ALMI_CMD_Y=0.6 ALMI_CMD_YAW=0.6 \
    python legged_gym/scripts/train.py --task=h1_2_lower --run_name=$R --headless \
    --resume --load_run="$(dirname "$lower3wide")" --checkpoint="${f//[!0-9]/}" --max_iterations $B_ITERS \
    > "$LOGS/$R.log" 2>&1 < /dev/null &
  PID=$!; echo $PID > "$LOGS/$R.pid"
  bash "$WATCHDOG" $PID "$LOGS/watchdog_$R.log" $PAUSE_AT $RESUME_AT &
  say "B: launched $R pid=$PID ($B_ITERS iterations from $lower3wide)"
  wait $PID
fi
dir=$(ls -d logs/h1_2_lower/*_"$R" | tail -1)
LOWER3WL=$RL/$(latest_ckpt "$dir")
say "B: finished, final policy $LOWER3WL"
echo "lower3wide_long=$LOWER3WL" >> "$LOGS/almi_final_policies.txt"
python legged_gym/scripts/eval_almi.py --lower_policy "$LOWER3WL" --upper_policy "$upper2" \
  --motions "$EVAL/eval_motions.pkl" --motion_csv "$EVAL/eval_mean_episode_length.csv" \
  --label "lower-3 wide cmds long + upper-2" --out "$RESULTS/lower3widelong_upper2.json" > "$RESULTS/lower3widelong_upper2.log" 2>&1
say "B: evaluation exit $?"; grep -a "^\[eval\]" "$RESULTS/lower3widelong_upper2.log"
python "$REPO/scripts/results_table.py" "$RESULTS"/lower*_upper2.json > "$RESULTS/RESULTS.md"
conda deactivate

# ================= A: foundation models =================
conda activate almi-trans
cd "$TR"
export ALMI_SAVE_EVERY_EPOCHS=1

# $1 run  $2 log  $3 regex before the metric  $4 min iters  $5 max iters  $6 max seconds
# $7 save marker  then: the training command
run_trans() {
  local run=$1 log=$2 key=$3 min=$4 max=$5 maxsec=$6 marker=$7; shift 7
  "$@" > "$log" 2>&1 < /dev/null &
  local pid=$! t0=$(date +%s) decided=""
  echo $pid > "$LOGS/$run.pid"
  bash "$WATCHDOG" $pid "$LOGS/watchdog_$run.log" $PAUSE_AT $RESUME_AT &
  say "A: launched $run pid=$pid"
  while kill -0 $pid 2>/dev/null; do
    sleep 120
    if [ -z "$decided" ]; then
      res=$(tr '\r' '\n' < "$log" | awk -v key="$key" -v min=$min -v max=$max -v win=60 -v gain=2 '
        /Train. Iter [0-9]+/ && match($0, key "[ \t]*[0-9.eE+-]+") {
          it = $0; sub(/.*Train. Iter /, "", it); it += 0
          v = substr($0, RSTART, RLENGTH); sub(key "[ \t]*", "", v); l[n++] = v + 0 }
        END {
          if (it >= max) { printf "cap %d\n", it; exit }
          if (it < min || n < 2 * win) exit
          for (i = n - win; i < n; i++) now += l[i]
          for (i = n - 2 * win; i < n - win; i++) before += l[i]
          if (now > before * (1 - gain / 100)) printf "plateau %d %.5f %.5f\n", it, before / win, now / win }')
      [ -z "$res" ] && [ $(( $(date +%s) - t0 )) -ge $maxsec ] && res="wall-clock cap"
      [ -n "$res" ] && { decided=$(grep -ac "$marker" "$log"); say "A: $run stop rule met ($res); waiting for the next save"; }
    elif [ "$(grep -ac "$marker" "$log")" -gt "$decided" ]; then
      sleep 30; kill $pid; say "A: $run stopped after its save"; break
    fi
  done
  while kill -0 $pid 2>/dev/null; do sleep 5; done
}

# A1: CL-400sl (paper hyper-parameters from train_almi_cl_400sl.sh)
# If the GPU is shared and the paper's batch does not fit, retry once with half the batch.
a1() {  # $1 run name  $2 batch size
  run_trans $1 "$LOGS/$1.log" "Loss\\." 20000 150000 $((3*3600)) "model last saved" \
    python train_almi_cl_400sl.py --num-obs 71 --exp-name $1 --batch-size $2 --num-layers 9 --embed-dim-gpt 256 \
    --n-head-gpt 16 --seq-length 400 --ff-rate 4 --drop-out-rate 0.1 --out-dir output --total-epoch 10000 \
    --lr-scheduler 100000 --lr 0.0001 --dataname almi --dilation-growth-rate 3 --device cuda:0 --pred-action
}
R4=cl400sl_${DAY}_${SHA}
a1 $R4 64
if grep -aq "OutOfMemoryError" "$LOGS/$R4.log"; then
  say "A1: out of GPU memory at batch 64 (shared GPU); retrying with batch 32"
  R4=${R4}_bs32; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True a1 $R4 32
fi
say "A1 model: $TR/output/$R4/almi_trans_cl_400sl_last.pth"

# A2: VQ-VAE (paper hyper-parameters from train_almi_vq.sh)
RV=vq_${DAY}_${SHA}
run_trans $RV "$LOGS/$RV.log" "Recons\\." 20000 200000 $((90*60)) "model last saved" \
  python train_almi_vq.py --batch-size 128 --lr 2e-4 --lr-scheduler 200000 --nb-code 1024 --down-t 2 --depth 3 \
  --dilation-growth-rate 3 --out-dir output --dataname almi --vq-act relu --quantizer ema_reset --loss-vel 0.5 \
  --recons-loss l1_smooth --exp-name $RV --device cuda:0 --window-size 64 --total-epoch 10000
say "A2 model: $TR/output/$RV/vq_net_last.pth"

# A3: OL on the trained VQ-VAE (upstream script points at vq_net_0.pth, the untrained epoch-0 weights)
a3() {  # $1 run name  $2 batch size
  run_trans $1 "$LOGS/$1.log" "Loss\\." 20000 150000 $((3*3600)) "model last saved" \
    python train_almi_ol.py --num-obs 71 --exp-name $1 --batch-size $2 --num-layers 9 --embed-dim-gpt 256 \
    --n-head-gpt 16 --seq-length 20 --ff-rate 4 --drop-out-rate 0.1 --out-dir output --total-epoch 10000 \
    --lr-scheduler 150000 --lr 0.0001 --dataname almi --dilation-growth-rate 3 --device cuda:0 \
    --vq-name $RV --resume-pth output/$RV/vq_net_last.pth --nb-code 1024 --vq-act relu --down-t 2 \
    --block-size 176 --max-motion-len 700
}
RO=ol_${DAY}_${SHA}
a3 $RO 128
if grep -aq "OutOfMemoryError" "$LOGS/$RO.log"; then
  say "A3: out of GPU memory at batch 128 (shared GPU); retrying with batch 64"
  RO=${RO}_bs64; PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True a3 $RO 64
fi
say "A3 model: $TR/output/$RO/almi_trans_ol_last.pth"
say "ALL DONE"
