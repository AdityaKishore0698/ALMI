#!/usr/bin/env bash
# Unattended pipeline on cir:
#   stage A: continue lower-1 from a checkpoint with the arm curriculum starting at
#            ALMI_INIT_ARM_WEIGHT, until arm_curriculum reaches 1.0 plus EXTRA_ITERS
#            of training at full arm amplitude
#   stage B: train upper-1 with that lower-1 checkpoint as the frozen lower body
# Each stage runs detached with scripts/gpu_watchdog.sh guarding GPU temperature.
#
# Usage: bash pipeline_lower1_upper1.sh <lower1-load-run> <checkpoint> <init-arm-weight> <sha>
set -uo pipefail
LOAD_RUN=$1; CKPT=$2; ARM0=$3; SHA=$4
EXTRA_ITERS=${EXTRA_ITERS:-300}
LOGS=$HOME/ALMI/logs
RL=$HOME/ALMI/ALMI-Open/ALMI_RL
WATCHDOG=${WATCHDOG:-$HOME/ALMI/bin/gpu_watchdog.sh}
PAUSE_AT=${PAUSE_AT:-88}; RESUME_AT=${RESUME_AT:-82}
DAY=$(date +%Y%m%d)
say() { echo "[$(date '+%F %T')] $*"; }

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate almi-rl
export WANDB_MODE=offline
cd "$RL"

launch() {  # $1 run name, rest: train.py args; sets PID
  local run=$1; shift
  python legged_gym/scripts/train.py --run_name="$run" --headless "$@" > "$LOGS/$run.log" 2>&1 < /dev/null &
  PID=$!; echo $PID > "$LOGS/$run.pid"
  bash "$WATCHDOG" $PID "$LOGS/watchdog_$run.log" $PAUSE_AT $RESUME_AT &
  say "launched $run pid=$PID"
}

# ---------- stage A: finish the lower-1 arm curriculum ----------
A=lower1_${DAY}_${SHA}_armcurr
ALMI_INIT_ARM_WEIGHT=$ARM0 launch $A --task=h1_2_wb_curriculum --resume \
  --load_run="$LOAD_RUN" --checkpoint="$CKPT" --max_iterations 6000
A_DIR=""; target=""
while kill -0 $PID 2>/dev/null; do
  sleep 60
  [ -z "$A_DIR" ] && A_DIR=$(ls -d logs/h1_2_wb_curriculum/*_"$A" 2>/dev/null | tail -1)
  if [ -z "$target" ]; then
    # first logged iteration (absolute, as printed) whose arm_curriculum is >= 1.0
    it=$(awk 'match($0, /Learning iteration [0-9]+/) {it = substr($0, RSTART+19, RLENGTH-19)}
              /arm_curriculum/ && $4+0 >= 1.0 {print it; exit}' "$LOGS/$A.log")
    if [ -n "$it" ]; then
      target=$(( (it / 100 + 1) * 100 + EXTRA_ITERS ))
      say "arm_curriculum reached 1.0 at iteration $it; stopping stage A at model_$target.pt"
    fi
  elif [ -n "$A_DIR" ] && [ -f "$A_DIR/model_$target.pt" ]; then
    sleep 20; kill $PID; say "stage A stopped at model_$target.pt"; break
  fi
done
wait $PID 2>/dev/null
# use the chosen checkpoint, or the newest one if stage A ended some other way
LOWER=${A_DIR}/model_${target}.pt
[ -n "$target" ] && [ -f "$LOWER" ] || LOWER=$(ls "$A_DIR"/model_*.pt | sort -t_ -k2 -n | tail -1)
[ -f "$LOWER" ] || { say "no lower-1 checkpoint found, aborting"; exit 1; }
say "lower-1 final policy: $LOWER"
echo "$RL/$LOWER" > "$LOGS/lower1_final_policy.txt"

# ---------- stage B: upper-1 on top of the frozen lower-1 ----------
B=upper1_$(date +%Y%m%d)_${SHA}
ALMI_LOWER_POLICY="$RL/$LOWER" launch $B --task=h1_2_upper --max_iterations 10000
wait $PID
say "upper-1 finished (exit $?)"
