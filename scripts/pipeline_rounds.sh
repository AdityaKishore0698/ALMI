#!/usr/bin/env bash
# Unattended adversarial rounds on cir, each stage stopped automatically at a plateau:
#   upper-1 (already running, attached) -> lower-2 -> upper-2 -> lower-3
# Rounds 2-3 fine-tune from the previous policy of the same body half (not from
# scratch, to save GPU time) with the arm curriculum starting at full amplitude.
# Every stage runs under scripts/gpu_watchdog.sh.
#
# A stage stops at the first checkpoint after all of these hold:
#   - at least MIN_ITERS iterations in this stage
#   - arm_curriculum >= 0.99
#   - mean reward over the last WIN iterations is less than GAIN % above the mean
#     over the WIN iterations ending LAG iterations earlier
# or when it reaches MAX_ITERS.
#
# Usage: bash pipeline_rounds.sh <upper1-pid> <upper1-run> <lower1-ckpt-path> <sha>
set -uo pipefail
U1_PID=$1; U1_RUN=$2; L1_CKPT=$3; SHA=$4
MIN_ITERS=${MIN_ITERS:-1000}; MAX_ITERS=${MAX_ITERS:-6000}
WIN=${WIN:-200}; LAG=${LAG:-500}; GAIN=${GAIN:-2}
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

plateaued() {  # $1 log -> prints "yes <iters>" when the stop rule holds
  awk -v win=$WIN -v lag=$LAG -v gain=$GAIN -v min=$MIN_ITERS '
    /Mean reward:/ {r[n++] = $3}
    /arm_curriculum:/ {arm = $4}
    END {
      if (n < min || n < win + lag || arm + 0 < 0.99) exit
      for (i = n - win; i < n; i++) now += r[i]
      for (i = n - win - lag; i < n - lag; i++) before += r[i]
      now /= win; before /= win
      if (now < before * (1 + gain / 100)) printf "yes %d %.2f %.2f\n", n, before, now
    }' "$1"
}

latest_ckpt() { local f; f=$(ls "$1" 2>/dev/null | grep "^model_[0-9]*\.pt$" | sort -t_ -k2 -n | tail -1); [ -n "$f" ] && echo "$1/$f"; }

run_until_plateau() {  # $1 run name, $2 experiment dir name, $3 pid; sets CKPT
  local run=$1 exp=$2 pid=$3 dir="" stop_after="" n
  while kill -0 $pid 2>/dev/null; do
    sleep 60
    [ -z "$dir" ] && dir=$(ls -d "logs/$exp/"*_"$run" 2>/dev/null | tail -1)
    n=$(grep -ac "Mean reward:" "$LOGS/$run.log")
    if [ -z "$stop_after" ]; then
      res=$(plateaued "$LOGS/$run.log")
      if [ -n "$res" ] || [ "$n" -ge "$MAX_ITERS" ]; then
        stop_after=$(latest_ckpt "$dir")
        say "$run: stop rule met after $n iterations ($res); waiting for the next checkpoint"
      fi
    else
      c=$(latest_ckpt "$dir")
      if [ "$c" != "$stop_after" ]; then sleep 20; kill $pid; say "$run: stopped at $c"; break; fi
    fi
  done
  while kill -0 $pid 2>/dev/null; do sleep 5; done
  CKPT=$RL/$(latest_ckpt "$dir")
  [ -f "$CKPT" ] || { say "$run: no checkpoint found, aborting"; exit 1; }
  say "$run: final policy $CKPT"
  echo "$CKPT" > "$LOGS/${run}_final_policy.txt"
}

# get_load_path() lists logs/<experiment>/ before using an absolute --load_run and
# fails if that folder is empty or missing, so give each experiment folder an entry.
for exp in h1_2_upper h1_2_lower; do mkdir -p "logs/$exp" && touch "logs/$exp/.keep"; done

resume_args() {  # $1 checkpoint path -> --resume args for train.py
  local f=$(basename "$1"); echo "--resume --load_run=$(dirname "$1") --checkpoint=${f//[!0-9]/}"
}

# ---------- upper-1: attach to the running process ----------
say "attaching to $U1_RUN (pid $U1_PID)"
run_until_plateau "$U1_RUN" h1_2_upper $U1_PID; U1=$CKPT

# ---------- lower-2: legs vs upper-1, fine-tuned from lower-1 ----------
R=lower2_${DAY}_${SHA}
ALMI_UPPER_POLICY=$U1 ALMI_INIT_ARM_WEIGHT=1.0 launch $R --task=h1_2_lower $(resume_args "$L1_CKPT") --max_iterations $MAX_ITERS
run_until_plateau $R h1_2_lower $PID; L2=$CKPT

# ---------- upper-2: arms vs lower-2, fine-tuned from upper-1 ----------
R=upper2_$(date +%Y%m%d)_${SHA}
ALMI_LOWER_POLICY=$L2 ALMI_INIT_ARM_WEIGHT=1.0 launch $R --task=h1_2_upper $(resume_args "$U1") --max_iterations $MAX_ITERS
run_until_plateau $R h1_2_upper $PID; U2=$CKPT

# ---------- lower-3: legs vs upper-2, fine-tuned from lower-2 ----------
R=lower3_$(date +%Y%m%d)_${SHA}
ALMI_UPPER_POLICY=$U2 ALMI_INIT_ARM_WEIGHT=1.0 launch $R --task=h1_2_lower $(resume_args "$L2") --max_iterations $MAX_ITERS
run_until_plateau $R h1_2_lower $PID; L3=$CKPT

say "ALL ROUNDS DONE. Evaluation pair: lower-3 $L3 with upper-2 $U2"
printf 'lower1=%s\nupper1=%s\nlower2=%s\nupper2=%s\nlower3=%s\n' "$L1_CKPT" "$U1" "$L2" "$U2" "$L3" > "$LOGS/almi_final_policies.txt"
