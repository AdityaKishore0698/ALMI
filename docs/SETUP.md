# Setup: two machines, one history

| | **cir** (lab, `ssh cir` → `anishma@172.16.21.9`) | **Mac** (local) |
|---|---|---|
| Role | All training, data collection, evaluation | Reading results, TensorBoard, MuJoCo playback, writing docs |
| Hardware | Ubuntu 22.04, Quadro RTX 6000 24 GB, driver 580 (CUDA 13.0), 80 cores, 125 GB RAM | Apple Silicon, no CUDA |
| Repo path | `~/ALMI/ALMI-Open` | `/Users/aditya/ALMI/ALMI-Open` |
| Envs | conda `almi-rl` (py3.8), `almi-trans` (py3.10) | `.venv` (py3.10, uv) |
| Isaac Gym | `~/ALMI/third_party/isaacgym` (Preview 4) | not supported on macOS |
| ALMI-X dataset | `~/ALMI/data/ALMI-X` (raw + `extracted/`) | not downloaded |

Network: both machines sit behind the campus proxy `http://172.31.2.3:8080`. `setup_cir.sh` exports it; the dataset download script does too.

## Directory layout on cir

```
~/ALMI/
├── ALMI-Open/               working copy (remotes: origin, upstream)
├── data/ALMI-X/             HF dataset: data.tar.gz, texts.tar.gz, select_motions.zip, train.txt
│   └── extracted/           actions/, texts/, select_motions/
├── third_party/isaacgym/    Isaac Gym Preview 4
└── logs/                    setup and long-job logs
```

Nothing under `data/`, `third_party/`, `ALMI_RL/logs/` or `ALMI_trans/output/` goes into git. The dataset reaches the code only through symlinks that `setup_cir.sh data` creates in `ALMI_trans/dataset/ALMI/`.

## Version-control model

```
upstream  github.com/TeleHuman/ALMI-Open        ──►  master   pristine upstream, never committed to
                                                              tag upstream-base-93f3eec
origin    github.com/AdityaKishore0698/ALMI     ◄─►  main     all our work: docs, fixes, eval scripts
```

* All work goes on `main` or on short-lived branches (`exp/<name>`, `fix/<name>`) that merge into `main`.
* Mac and cir sync only through GitHub (`origin`). Commit and push on the Mac, then `git pull` on cir. Never copy source files with scp.
* **Commits are made by hand.** Nothing commits automatically, including Claude, which only prepares changes and suggests commands.
* cir clones over HTTPS through the proxy. Pulling needs no credentials because the repo is public. Pushing from cir would need a GitHub token, so the normal flow is to commit on the Mac.
* The repo is **public**. Never commit datasets, checkpoints, wandb runs, Isaac Gym or credentials. `.gitignore` already covers these paths.
* To pick up new upstream work: `git fetch upstream && git checkout master && git merge --ff-only upstream/master && git checkout main && git merge master`.
* Each experiment's config change is its own commit. The run name records the commit hash (see `docs/REPRODUCTION_PLAN.md`), so every checkpoint traces back to exact code.
* Checkpoints and logs stay on cir. To view them on the Mac, pull them with rsync, for example:
  `rsync -av cir:ALMI/ALMI-Open/ALMI_RL/logs/ ALMI_RL/logs/`.

## Recreate the environments

```bash
# cir
cd ~/ALMI/ALMI-Open
bash scripts/setup_cir.sh rl      # almi-rl: torch 2.3.1+cu121, isaacgym, rsl_rl, ALMI_RL, mujoco 3.2.3, CLIP
bash scripts/setup_cir.sh trans   # almi-trans: torch 2.5.1+cu124, CLIP + ViT-B-32 weights
bash scripts/setup_cir.sh data    # extract ALMI-X and symlink it into ALMI_trans/dataset/ALMI

# mac
bash scripts/setup_mac.sh         # .venv with torch (CPU/MPS), mujoco, tensorboard
```

Everything is idempotent, so re-running is safe.

### Deliberate differences from the upstream install instructions

* `ALMI_RL/setup.py` pins `numpy==1.20` while `requirements_rl.txt` pins 1.23.5. We install `ALMI_RL` and `rsl_rl` with `--no-deps` and keep numpy 1.23.5, because Isaac Gym needs numpy below 1.24.
* The upstream `requirements_rl.txt` is a `pip freeze` of the authors' machine: conda-only mkl packages and editable `/home/bcj/...` paths. We use the cleaned list in `envs/almi-rl.txt` instead.
* PyTorch comes from pytorch.org wheel indexes, not conda. The CUDA runtime ships inside the wheel, so the system CUDA (11.5 on PATH) doesn't matter.
* The upstream README has a typo: it creates `almi-rl` but activates `unitree-rl`. Our env is named `almi-rl`.
* The HF split file is named `train.txt`, but the loaders read `train_ALMI.txt`, so we symlink one to the other.

## Daily use

```bash
ssh cir
source ~/anaconda3/etc/profile.d/conda.sh && conda activate almi-rl
cd ~/ALMI/ALMI-Open/ALMI_RL
# Long runs: launch detached with setsid nohup, writing to ~/ALMI/logs/<run>.log and <run>.pid,
# and use tmux only to watch the log (tmux new -s <run> "tail -F ~/ALMI/logs/<run>.log").
# Ctrl-C in a viewer then cannot kill training. Stop a run with: kill $(cat ~/ALMI/logs/<run>.pid)
```

Experiment logging: `train.py` always calls `wandb.init`, which needs a Weights & Biases account. Either run `wandb login` once on cir to get online dashboards, or `export WANDB_MODE=offline` to write logs locally only. TensorBoard logs are written either way.

TensorBoard from the Mac: `ssh -L 6006:localhost:6006 cir`, then on cir run
`tensorboard --logdir ~/ALMI/ALMI-Open/ALMI_RL/logs --port 6006`.

MuJoCo viewer on the Mac: run scripts with `.venv/bin/mjpython`, not `python`, because `mujoco.viewer.launch_passive` needs this on macOS.
