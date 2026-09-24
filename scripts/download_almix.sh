#!/usr/bin/env bash
# Download the ALMI-X dataset (https://huggingface.co/datasets/TeleEmbodied/ALMI-X).
# Resumable and robust to the campus proxy resetting HTTP/2 streams:
# it forces HTTP/1.1 and retries each file until it has the expected size.
#
# Usage: bash scripts/download_almix.sh [dest]   (default: ~/ALMI/data/ALMI-X)
set -uo pipefail
DEST=${1:-$HOME/ALMI/data/ALMI-X}
PROXY=${PROXY:-http://172.31.2.3:8080}
export http_proxy=$PROXY https_proxy=$PROXY
BASE=https://huggingface.co/datasets/TeleEmbodied/ALMI-X/resolve/main
mkdir -p "$DEST" && cd "$DEST"

# name  expected-bytes  (from the HF tree API, 2026-09)
FILES="README.md:3855 train.txt:5028008 texts.tar.gz:3978480 select_motions.zip:164762770 data.tar.gz:6536813883"

for entry in $FILES; do
  f=${entry%%:*}; want=${entry##*:}
  for attempt in $(seq 1 200); do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$have" -eq "$want" ] && { echo "[$(date +%T)] $f OK ($have bytes)"; break; }
    [ "$have" -gt "$want" ] && { echo "$f larger than expected, restarting"; rm -f "$f"; }
    echo "[$(date +%T)] $f attempt $attempt ($have/$want)"
    curl --http1.1 -fsSL --retry 5 --retry-all-errors -C - -o "$f" "$BASE/$f" || sleep 10
  done
done
sha256sum data.tar.gz select_motions.zip texts.tar.gz > SHA256SUMS
echo "[$(date +%T)] DONE"
