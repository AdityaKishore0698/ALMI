#!/usr/bin/env bash
# Upper-body experiments on cir, back to back on the GPU. Each run is CL-20sl (the paper's
# hyper-parameters) for the same number of optimizer steps as the reproduced baseline, with:
#   A. text-feature standardisation (--text-norm), one action per step
#   B. text-feature standardisation + action chunks of 10 (--chunk 10)
# After each run: TorchScript export and the text-sensitivity check (eval_text_sensitivity.py).
#
# Usage: bash pipeline_upper_body.sh <tag>      e.g. tag = 20261001_f313925wip
set -uo pipefail
TAG=$1
ITERS=${ITERS:-126000}   # the baseline CL-20sl ran about 126k steps
RUNS=${RUNS:-"tn tn_k10"}
LOGS=$HOME/ALMI/logs
TR=$HOME/ALMI/ALMI-Open/ALMI_trans
WATCHDOG=${WATCHDOG:-$HOME/ALMI/bin/gpu_watchdog.sh}
say() { echo "[$(date '+%F %T')] $*"; }
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate almi-trans
cd "$TR"

for variant in $RUNS; do
  case $variant in
    tn)     extra="--text-norm" ;;
    tn_k10) extra="--text-norm --chunk 10" ;;
    k10)    extra="--chunk 10" ;;
    *)      say "unknown variant $variant"; continue ;;
  esac
  R=cl20sl_${variant}_${TAG}
  python train_almi_cl_20sl.py --num-obs 71 --exp-name $R --batch-size 128 --num-layers 9 --embed-dim-gpt 256 \
    --n-head-gpt 16 --seq-length 20 --ff-rate 4 --drop-out-rate 0.1 --out-dir output --total-epoch 10000 \
    --lr-scheduler 150000 --lr 0.0001 --dataname almi --dilation-growth-rate 3 --device cuda:0 --pred-action \
    --max-iter $ITERS $extra > "$LOGS/$R.log" 2>&1 < /dev/null &
  PID=$!; echo $PID > "$LOGS/$R.pid"
  bash "$WATCHDOG" $PID "$LOGS/watchdog_$R.log" 88 82 &
  say "launched $R pid=$PID ($extra, $ITERS steps)"
  wait $PID
  say "$R finished: $(tr '\r' '\n' < "$LOGS/$R.log" | grep -a 'Train. Iter' | tail -1)"
  python export.py --seq_len 20 --model_path output/$R/almi_trans_cl_20sl_last.pth \
    --export_path output/export --export_name $R >> "$LOGS/$R.log" 2>&1
  python eval_text_sensitivity.py output/$R/almi_trans_cl_20sl_last.pth 2>&1 \
    | grep -v "Warning\|torch.load\|wrong name\|flag wrong\|it/s" > "$LOGS/${R}_text_sensitivity.txt"
  say "$R exported to output/export/policy_trans_$R.pt; text sensitivity in $LOGS/${R}_text_sensitivity.txt"
done
say "ALL DONE"
