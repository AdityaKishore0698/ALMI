#!/usr/bin/env bash
# Waits for pipeline_rounds.sh to finish, then evaluates the policy pairs of the
# paper's Tables 4 and 6 with eval_almi.py and writes one results table.
#   lower-3 + upper-2   (Table 4 ALMI row, Table 6 first row)
#   lower-2 + upper-2   (Table 6)
#   lower-1 + upper-2   (Table 6)
#
# Usage: bash eval_all.sh [results-dir]   (default: ~/ALMI/results/<date>)
set -uo pipefail
LOGS=$HOME/ALMI/logs
RL=$HOME/ALMI/ALMI-Open/ALMI_RL
EVAL=$HOME/ALMI/data/eval
OUT=${1:-$HOME/ALMI/results/$(date +%Y%m%d)}
POLICIES=$LOGS/almi_final_policies.txt
say() { echo "[$(date '+%F %T')] $*"; }

while [ ! -f "$POLICIES" ]; do sleep 120; done
say "training finished; policies:"; cat "$POLICIES"
source "$POLICIES"   # defines lower1 upper1 lower2 upper2 lower3

source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate almi-rl
cd "$RL"
mkdir -p "$OUT"

evaluate() {  # $1 label, $2 lower policy, $3 upper policy, $4 file stem
  say "evaluating $1"
  python legged_gym/scripts/eval_almi.py --lower_policy "$2" --upper_policy "$3" \
    --motions "$EVAL/eval_motions.pkl" --motion_csv "$EVAL/eval_mean_episode_length.csv" \
    --label "$1" --out "$OUT/$4.json" > "$OUT/$4.log" 2>&1
  say "$1: exit $?"; grep -a "^\[eval\]" "$OUT/$4.log"
}
evaluate "lower-3 + upper-2" "$lower3" "$upper2" lower3_upper2
evaluate "lower-2 + upper-2" "$lower2" "$upper2" lower2_upper2
evaluate "lower-1 + upper-2" "$lower1" "$upper2" lower1_upper2

python "$HOME/ALMI/ALMI-Open/scripts/results_table.py" "$OUT"/lower*_upper2.json > "$OUT/RESULTS.md"
say "ALL EVALUATIONS DONE -> $OUT/RESULTS.md"
cat "$OUT/RESULTS.md"
