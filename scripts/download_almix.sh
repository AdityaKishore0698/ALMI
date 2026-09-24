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

# Only one downloader per directory: two concurrent curls appending to the same
# file silently corrupt it (size ends up right, content does not).
exec 9> .download.lock
flock -n 9 || { echo "another download_almix.sh is already running in $DEST"; exit 1; }

# name  expected-bytes  (from the HF tree API, 2026-09)
FILES="README.md:3855 train.txt:5028008 texts.tar.gz:3978480 select_motions.zip:164762770 data.tar.gz:6536813883"

for entry in $FILES; do
  f=${entry%%:*}; want=${entry##*:}
  for attempt in $(seq 1 200); do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    [ "$have" -eq "$want" ] && { echo "[$(date +%T)] $f OK ($have bytes)"; break; }
    [ "$have" -gt "$want" ] && { echo "$f larger than expected, restarting"; rm -f "$f"; }
    echo "[$(date +%T)] $f attempt $attempt ($have/$want)"
    # NB: no curl --retry here. curl computes the -C - offset once, so its own
    # retries truncate the file back to the original offset. This loop re-stats.
    curl --http1.1 -fsSL --connect-timeout 30 --speed-limit 1024 --speed-time 120 \
         -C - -o "$f" "$BASE/$f" || sleep 10
  done
done
# Expected SHA-256 = the Git-LFS oids listed by the HF tree API.
cat > SHA256SUMS.expected <<'SUMS'
b89db5f42c1e7842c99ec3126a75e8b0043b929a8ba9c62a246384aac4fbe1cf  data.tar.gz
aeabebe4f5afdba3bfae6f052b855ca93f40a296f2a4111a30ce410ea133d4ba  select_motions.zip
437bf0401d3c474af333ccee271a68d2a92a3cb4086f8a046b22c9421675ce2d  texts.tar.gz
SUMS
if sha256sum -c SHA256SUMS.expected; then
  echo "[$(date +%T)] DONE (checksums verified)"
else
  echo "[$(date +%T)] CHECKSUM MISMATCH: delete the bad file and re-run"; exit 1
fi
