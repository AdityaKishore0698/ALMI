#!/usr/bin/env bash
# Gate 0 smoke tests on cir (see docs/REPRODUCTION_PLAN.md).
# Usage: bash scripts/check_env_cir.sh
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
source "$HOME/anaconda3/etc/profile.d/conda.sh"
export WANDB_MODE=offline   # train.py always calls wandb.init; no account needed offline
pass=0; fail=0
check() { local name=$1; shift; echo -e "\n=== $name"; if "$@"; then echo "PASS: $name"; pass=$((pass+1)); else echo "FAIL: $name"; fail=$((fail+1)); fi; }

conda activate almi-rl
check "almi-rl: torch sees GPU" python -c "
import isaacgym, torch
assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))"
check "almi-rl: imports (numpy<1.24, mujoco, legged_gym, rsl_rl, clip)" python -c "
import isaacgym, numpy, mujoco, clip, rsl_rl, legged_gym
from legged_gym.envs import task_registry
assert tuple(map(int, numpy.__version__.split('.')[:2])) < (1, 24)
print('numpy', numpy.__version__, 'mujoco', mujoco.__version__, 'tasks', list(task_registry.task_classes))"
check "almi-rl: lower-1 training, 64 envs x 3 iters" bash -c "
cd '$REPO/ALMI_RL' && python legged_gym/scripts/train.py --task=h1_2_wb_curriculum \
  --run_name=gate0_smoke --headless --num_envs 64 --max_iterations 3"
conda deactivate

conda activate almi-trans
check "almi-trans: torch sees GPU + CLIP weights" bash -c "
cd '$REPO/ALMI_trans' && python -c \"
import torch, clip
assert torch.cuda.is_available()
m, _ = clip.load('pretrained/ViT-B-32.pt', device='cuda', jit=False); print('CLIP ok', torch.__version__)\""
check "ALMI-X linked into ALMI_trans/dataset/ALMI" bash -c "
cd '$REPO/ALMI_trans/dataset/ALMI' && test -f train_ALMI.txt && test -d actions && test -d texts &&
echo \"split: \$(wc -l < train_ALMI.txt) | actions: \$(ls actions | wc -l) | texts: \$(ls texts | wc -l)\""
conda deactivate

echo -e "\nGate 0: $pass passed, $fail failed"
[ "$fail" -eq 0 ]
