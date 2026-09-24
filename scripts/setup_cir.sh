#!/usr/bin/env bash
# Idempotent environment setup for the CIR lab machine (Ubuntu 22.04, Quadro RTX 6000).
#
# Creates two conda envs under ~/anaconda3/envs:
#   almi-rl    (py3.8)  : Isaac Gym + rsl_rl + ALMI_RL, also used for Data_Collection
#                         and ALMI_trans MuJoCo deploy
#   almi-trans (py3.10) : ALMI_trans transformer training
#
# Expects Isaac Gym Preview 4 extracted at $ALMI_HOME/third_party/isaacgym
# and the ALMI-X dataset at $ALMI_HOME/data/ALMI-X (see docs/SETUP.md).
#
# Usage:  bash scripts/setup_cir.sh [rl|trans|data|all]   (default: all)
set -euo pipefail

ALMI_HOME=${ALMI_HOME:-$HOME/ALMI}
REPO=$(cd "$(dirname "$0")/.." && pwd)
PROXY=${PROXY:-http://172.31.2.3:8080}
export http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY

source "$HOME/anaconda3/etc/profile.d/conda.sh"
ENVS=$HOME/anaconda3/envs
TARGET=${1:-all}

log() { echo -e "\n\033[1;34m[setup_cir] $*\033[0m"; }

setup_rl() {
  if [ ! -x "$ENVS/almi-rl/bin/python" ]; then
    log "creating env almi-rl (python 3.8)"
    conda create -y -p "$ENVS/almi-rl" python=3.8
  fi
  conda activate "$ENVS/almi-rl"
  log "torch 2.3.1 + cu121"
  pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
  log "pinned requirements"
  pip install -r "$REPO/envs/almi-rl.txt"
  log "isaacgym (editable)"
  pip install -e "$ALMI_HOME/third_party/isaacgym/python" --no-deps
  log "rsl_rl + ALMI_RL (editable, --no-deps: setup.py pins numpy==1.20 which breaks isaacgym)"
  pip install -e "$REPO/ALMI_RL/rsl_rl" --no-deps
  pip install -e "$REPO/ALMI_RL" --no-deps
  log "OpenAI CLIP"
  pip install "setuptools<81" wheel  # CLIP setup.py needs pkg_resources
  pip install --no-build-isolation --no-deps "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
  # Isaac Gym's gymtorch JIT build needs libpython3.8.so on the loader path
  mkdir -p "$ENVS/almi-rl/etc/conda/activate.d"
  echo 'export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}' > "$ENVS/almi-rl/etc/conda/activate.d/isaacgym.sh"
  conda deactivate
}

setup_trans() {
  if [ ! -x "$ENVS/almi-trans/bin/python" ]; then
    log "creating env almi-trans (python 3.10)"
    conda create -y -p "$ENVS/almi-trans" python=3.10
  fi
  conda activate "$ENVS/almi-trans"
  log "torch 2.5.1 + cu124"
  pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
  pip install -r "$REPO/envs/almi-trans.txt"
  pip install "setuptools<81" wheel  # CLIP setup.py needs pkg_resources
  pip install --no-build-isolation --no-deps "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
  log "CLIP ViT-B/32 weights -> ALMI_trans/pretrained/ViT-B-32.pt"
  (cd "$REPO/ALMI_trans" && python -c "import clip; clip.load('ViT-B/32', device='cpu', download_root='pretrained')")
  conda deactivate
}

setup_data() {
  local X=$ALMI_HOME/data/ALMI-X
  local D=$REPO/ALMI_trans/dataset/ALMI
  log "linking ALMI-X into $D"
  mkdir -p "$X/extracted"
  if [ -f "$X/texts.tar.gz" ] && [ ! -d "$X/extracted/texts" ]; then tar xzf "$X/texts.tar.gz" -C "$X/extracted"; fi
  if [ -f "$X/data.tar.gz" ] && [ ! -f "$X/extracted/.data_done" ]; then
    tar xzf "$X/data.tar.gz" -C "$X/extracted" && touch "$X/extracted/.data_done"
  fi
  if [ -f "$X/select_motions.zip" ] && [ ! -f "$X/extracted/.select_done" ]; then
    mkdir -p "$X/extracted/select_motions" && unzip -oq "$X/select_motions.zip" -d "$X/extracted/select_motions" && touch "$X/extracted/.select_done"
  fi
  mkdir -p "$D"
  [ -d "$X/extracted/texts" ]   && ln -sfn "$X/extracted/texts"   "$D/texts"
  [ -d "$X/extracted/actions" ] && ln -sfn "$X/extracted/actions" "$D/actions"
  # the HF split file is called train.txt, the loaders read train_ALMI.txt
  [ -f "$X/train.txt" ] && ln -sfn "$X/train.txt" "$D/train_ALMI.txt"
  ls -la "$D"
}

case $TARGET in
  rl) setup_rl ;;
  trans) setup_trans ;;
  data) setup_data ;;
  all) setup_rl; setup_trans; setup_data ;;
  *) echo "unknown target $TARGET"; exit 1 ;;
esac
log "done ($TARGET)"
