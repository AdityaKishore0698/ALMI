#!/usr/bin/env bash
# Local (macOS arm64) results-viewing environment. No Isaac Gym / CUDA here:
# this env is for tensorboard, MuJoCo playback of exported policies and plotting.
#
# Usage: bash scripts/setup_mac.sh     -> creates .venv at the repo root
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
[ -x .venv/bin/python ] || uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r envs/mac-viz.txt
uv pip install --python .venv/bin/python "setuptools<81" wheel
# CLIP's setup.py imports pkg_resources (removed in setuptools>=81) -> build without isolation
uv pip install --python .venv/bin/python --no-deps --no-build-isolation "git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1"
.venv/bin/python -c "import torch, mujoco, clip; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available(), '| mujoco', mujoco.__version__)"
echo "Activate with: source .venv/bin/activate   (use 'mjpython' instead of 'python' for mujoco.viewer on macOS)"
