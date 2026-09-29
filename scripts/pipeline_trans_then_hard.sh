#!/usr/bin/env bash
# Unattended follow-up on cir:
#   stage A: let the running CL-20sl foundation-model training reach a loss plateau,
#            then stop it right after its next end-of-epoch save
#   stage B: retrain lower-3 with the command ranges the paper's hard level needs
#            (vx up to 1.3, vy and yaw rate +-0.6; gap G14), fine-tuned from lower-2
#            against upper-2, then evaluate it with eval_almi.py
#
# A stops when it has at least A_MIN_ITERS iterations and the mean loss of the last
# A_WIN printed values (100 iterations each) is less than A_GAIN % below the mean of
# the A_WIN before them, or at A_MAX_ITERS.
# B stops like pipeline_rounds.sh, but only after max_command_x has reached 1.3.
#
# Usage: bash pipeline_trans_then_hard.sh <cl20sl-pid> <cl20sl-run> <sha>
set -uo pipefail
A_PID=$1; A_RUN=$2; SHA=$3
A_MIN_ITERS=${A_MIN_ITERS:-40000}; A_MAX_ITERS=${A_MAX_ITERS:-120000}
A_WIN=${A_WIN:-60}; A_GAIN=${A_GAIN:-2}
MIN_ITERS=${MIN_ITERS:-1000}; MAX_ITERS=${MAX_ITERS:-6000}
WIN=${WIN:-200}; LAG=${LAG:-500}; GAIN=${GAIN:-2}
MAX_CMD_X=1.3; CMD_Y=0.6; CMD_YAW=0.6
LOGS=$HOME/ALMI/logs
REPO=$HOME/ALMI/ALMI-Open
RL=$REPO/ALMI_RL
EVAL=$HOME/ALMI/data/eval
RESULTS=${RESULTS:-$HOME/ALMI/results/20260929}
WATCHDOG=${WATCHDOG:-$HOME/ALMI/bin/gpu_watchdog.sh}
PAUSE_AT=${PAUSE_AT:-88}; RESUME_AT=${RESUME_AT:-82}
say() { echo "[$(date '+%F %T')] $*"; }
source "$HOME/anaconda3/etc/profile.d/conda.sh"

# ---------- stage A: foundation model to a loss plateau ----------
A_LOG=$LOGS/$A_RUN.log
a_plateaued() {
  tr '\r' '\n' < "$A_LOG" | awk -v win=$A_WIN -v gain=$A_GAIN -v min=$A_MIN_ITERS -v max=$A_MAX_ITERS '
    /Train. Iter [0-9]+ : Loss/ {it = $(NF-3) + 0; l[n++] = $NF + 0}
    END {
      if (it >= max) { printf "yes (cap) %d\n", it; exit }
      if (it < min || n < 2 * win) exit
      for (i = n - win; i < n; i++) now += l[i]
      for (i = n - 2 * win; i < n - win; i++) before += l[i]
      now /= win; before /= win
      if (now > before * (1 - gain / 100)) printf "yes %d %.5f %.5f\n", it, before, now
    }'
}
saves() { grep -ac "model last saved" "$A_LOG"; }
say "stage A: watching $A_RUN (pid $A_PID)"
decided=""
while kill -0 $A_PID 2>/dev/null; do
  sleep 120
  if [ -z "$decided" ]; then
    res=$(a_plateaued)
    [ -n "$res" ] && { decided=$(saves); say "$A_RUN: stop rule met ($res); waiting for the next epoch save"; }
  elif [ "$(saves)" -gt "$decided" ]; then
    sleep 30; kill $A_PID; say "$A_RUN: stopped after the epoch save"; break
  fi
done
while kill -0 $A_PID 2>/dev/null; do sleep 5; done
say "$A_RUN final model: $REPO/ALMI_trans/output/$A_RUN/almi_trans_cl_20sl_last.pth"

# ---------- stage B: lower-3 with the hard-level command ranges ----------
conda activate almi-rl
export WANDB_MODE=offline
cd "$RL"
source "$LOGS/almi_final_policies.txt"   # lower2 upper2 ...
R=lower3wide_$(date +%Y%m%d)_${SHA}
f=$(basename "$lower2")
ALMI_UPPER_POLICY=$upper2 ALMI_INIT_ARM_WEIGHT=1.0 ALMI_MAX_CMD_X=$MAX_CMD_X ALMI_CMD_Y=$CMD_Y ALMI_CMD_YAW=$CMD_YAW \
  python legged_gym/scripts/train.py --task=h1_2_lower --run_name=$R --headless \
  --resume --load_run="$(dirname "$lower2")" --checkpoint="${f//[!0-9]/}" --max_iterations $MAX_ITERS \
  > "$LOGS/$R.log" 2>&1 < /dev/null &
PID=$!; echo $PID > "$LOGS/$R.pid"
bash "$WATCHDOG" $PID "$LOGS/watchdog_$R.log" $PAUSE_AT $RESUME_AT &
say "stage B: launched $R pid=$PID"

b_plateaued() {
  awk -v win=$WIN -v lag=$LAG -v gain=$GAIN -v min=$MIN_ITERS -v cmd=$MAX_CMD_X '
    /Mean reward:/ {r[n++] = $3}
    /arm_curriculum:/ {arm = $4}
    /max_command_x:/ {mc = $4}
    END {
      if (n < min || n < win + lag || arm + 0 < 0.99 || mc + 0 < cmd - 0.01) exit
      for (i = n - win; i < n; i++) now += r[i]
      for (i = n - win - lag; i < n - lag; i++) before += r[i]
      now /= win; before /= win
      if (now < before * (1 + gain / 100)) printf "yes %d %.2f %.2f\n", n, before, now
    }' "$1"
}
latest_ckpt() { local f; f=$(ls "$1" 2>/dev/null | grep "^model_[0-9]*\.pt$" | sort -t_ -k2 -n | tail -1); [ -n "$f" ] && echo "$1/$f"; }
dir=""; stop_after=""
while kill -0 $PID 2>/dev/null; do
  sleep 60
  [ -z "$dir" ] && dir=$(ls -d logs/h1_2_lower/*_"$R" 2>/dev/null | tail -1)
  n=$(grep -ac "Mean reward:" "$LOGS/$R.log")
  if [ -z "$stop_after" ]; then
    res=$(b_plateaued "$LOGS/$R.log")
    if [ -n "$res" ] || [ "$n" -ge "$MAX_ITERS" ]; then
      stop_after=$(latest_ckpt "$dir"); say "$R: stop rule met after $n iterations ($res)"
    fi
  elif [ "$(latest_ckpt "$dir")" != "$stop_after" ]; then
    sleep 20; kill $PID; say "$R: stopped at $(latest_ckpt "$dir")"; break
  fi
done
while kill -0 $PID 2>/dev/null; do sleep 5; done
LOWER3W=$RL/$(latest_ckpt "$dir")
say "$R final policy: $LOWER3W"
echo "lower3wide=$LOWER3W" >> "$LOGS/almi_final_policies.txt"

# ---------- evaluate lower-3 (wide commands) + upper-2 ----------
python legged_gym/scripts/eval_almi.py --lower_policy "$LOWER3W" --upper_policy "$upper2" \
  --motions "$EVAL/eval_motions.pkl" --motion_csv "$EVAL/eval_mean_episode_length.csv" \
  --label "lower-3 wide cmds + upper-2" --out "$RESULTS/lower3wide_upper2.json" > "$RESULTS/lower3wide_upper2.log" 2>&1
say "evaluation exit $?"; grep -a "^\[eval\]" "$RESULTS/lower3wide_upper2.log"
python "$REPO/scripts/results_table.py" "$RESULTS"/lower*_upper2.json > "$RESULTS/RESULTS.md"
say "ALL DONE -> $RESULTS/RESULTS.md"
