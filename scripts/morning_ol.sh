#!/usr/bin/env bash
# Morning update after OL training: fetch the final OL model, run the paper's Table 13
# commands in MuJoCo (8 s each, recorded), refresh the report figures and recompile it.
# Run on the Mac from anywhere: bash ALMI-Open/scripts/morning_ol.sh
set -euo pipefail
ALMI=/Users/aditya/ALMI
TR=$ALMI/ALMI-Open/ALMI_trans
RUN=ol_20260930_b128acc
VQ=vq_20260930_8a1wip
OUT=$ALMI/ALMI-Open/results/20260929/FOUNDATION_OL_raw.txt

echo "== OL training state on cir"
ssh cir "pgrep -f '[t]rain_almi_ol.py' >/dev/null && echo 'STILL TRAINING' || echo finished; tail -3 ~/ALMI/logs/supervise_ol.log; tr '\r' '\n' < ~/ALMI/logs/$RUN.log | grep -a 'Train. Iter' | tail -1"

echo "== copy final OL model (checksum-verified)"
REMOTE_SHA=$(ssh cir "cp ~/ALMI/ALMI-Open/ALMI_trans/output/$RUN/almi_trans_ol_last.pth /tmp/ol_final.pth && sha256sum /tmp/ol_final.pth | cut -d' ' -f1")
mkdir -p "$TR/output/$RUN"
scp -q cir:/tmp/ol_final.pth "$TR/output/$RUN/almi_trans_ol_last.pth"
[ "$(shasum -a 256 "$TR/output/$RUN/almi_trans_ol_last.pth" | cut -d' ' -f1)" = "$REMOTE_SHA" ] || { echo "CHECKSUM MISMATCH"; exit 1; }

echo "== Table 13 commands in MuJoCo" | tee "$OUT"
cd "$TR"
for cmd in "forward slowly and wave left" "backward moderately and wave right" "right fast and wave both"; do
  slug=$(echo "$cmd" | tr ' ' '_')
  ../.venv/bin/python deploy/deploy_mujoco/play_ol.py --trans "output/$RUN/almi_trans_ol_last.pth" \
    --vq "output/$VQ/vq_net_last.pth" --text "Robot go $cmd." --duration 8 \
    --record "../results/videos/ol_t13_$slug.mp4" --no_viewer 2>/dev/null | grep -E "^generated|^text|^survival|^average" | tee -a "$OUT"
done

echo "== refresh report figures and PDF"
ssh cir "tr '\r' '\n' < ~/ALMI/logs/$RUN.log | grep -a 'Train. Iter' | sed 's/.*Train. Iter \([0-9]*\) : Loss. \([0-9.]*\).*/\1 \2/'" > "$ALMI/Thesis_Report/Figures/$RUN.loss"
cd "$ALMI/Thesis_Report"
../ALMI-Open/.venv/bin/python figures_src/make_figures.py
tectonic -X compile main.tex >/dev/null 2>&1 && echo "report rebuilt: $ALMI/Thesis_Report/main.pdf"
echo "== done. Raw OL numbers in $OUT"
