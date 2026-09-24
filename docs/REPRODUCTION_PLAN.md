# ALMI reproduction plan

Paper: *Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning* (Shi et al., NeurIPS 2025, arXiv:2504.14305v3).
Code baseline: upstream `93f3eec` (tag `upstream-base-93f3eec`).

## What the paper claims, and what the released code can reach

| Paper result | Needs | Released code | Status |
|---|---|---|---|
| **Table 4**: ALMI on H1-2, CMU MoCap, easy/medium/hard (Evel, Eang, E_jpe, E_kpe, E_action, Eg, survival) | 3 adversarial rounds (lower-3 + upper-2), the CMU eval set, an eval harness | Training code for rounds 1–2 only; no eval harness; no CMU retargeted motions | **Partial.** Training is feasible; eval harness and CMU data must be built (G1, G2, G3) |
| Table 4 baselines: ALMI(whole), Exbody | Whole-body policy config, Exbody | Not released | Out of scope for phase 1 (G4) |
| **Table 5**: G1 vs OmniH2O / ExBody2 | G1 configs + baselines | Not released | Out of scope (G4) |
| **Table 6**: lower-1/2/3 + upper-2, w/o arm curriculum | Round checkpoints, `arm_curriculum=False` | Flag exists in config | **Feasible** once Table 4 works |
| Tables 14–15: trained adversary / dual-objective variants | Joint-training code | Not released | Out of scope (G4) |
| **Tables 12–13**: foundation model CL-20sl / CL-400sl / OL | ALMI-X (wave subset), `ALMI_trans` | Training + export + MuJoCo deploy | **Feasible.** The eval metric scripts must be written (G5) |
| Figs 3, 8–10: real robot | Unitree H1-2 | Deploy code not released | Not reproducible here |

## Phases

**Phase 0: infrastructure (this commit).** Envs on cir, dataset on cir, a viewer env on the Mac, and the git model (`docs/SETUP.md`).

Gate 0 checks (`bash scripts/check_env_cir.sh`; logs go to wandb offline mode):
1. `python -c "import isaacgym, torch; print(torch.cuda.is_available())"` in `almi-rl`.
2. Isaac Gym's `examples/joint_monkey.py --headless` runs.
3. `train.py --task=h1_2_wb_curriculum --headless --num_envs 64 --max_iterations 5` completes.
4. `ALMI_trans`'s `train_almi_cl_20sl.py` runs a few iterations on the real ALMI-X.

**Phase 1: adversarial RL (Table 4 ALMI row, Table 6).** Run each step as `train.py --task=<task> --run_name=<name>_<gitsha> --headless`:

| Step | Task | Loads | Output |
|---|---|---|---|
| lower-1 | `h1_2_wb_curriculum` | open-loop upper body replaying `all_wave.pkl` | lower-1 |
| upper-1 | `h1_2_upper` | `lower_policy_path` = lower-1 | upper-1 |
| lower-2 | `h1_2_lower` | `upper_policy_path` = upper-1 | lower-2 |
| upper-2 | `h1_2_upper` | lower-2 | upper-2 |
| lower-3 | `h1_2_lower` | upper-2 | lower-3 |
| ablation | `h1_2_wb_curriculum` with `arm_curriculum=False`, `motion_scale=1` | none | w/o arm curriculum |

Paper budget: 4096 envs, three rounds, about 17 h in total. Each policy-path change is committed before its run starts.

**Phase 2: evaluation harness.** Build a play-style script that runs the fixed difficulty settings (Table 3: velocity command, terrain level, pushes) over a held-out motion set and logs all eight metrics. The metric definitions come from §6 of the paper.

**Phase 3: foundation model (Tables 12–13).** Train on the wave subset, export, run the MuJoCo deploy with 5 repeats × 8 s per command, and log survival, mean velocity, and lower- and upper-body success.

## Run-naming convention

`<step>_<YYYYMMDD>_<gitsha7>`, for example `lower1_20260925_3cb0994`. Record every run in `docs/RUN_LOG.md`: date, commit, command, wall time, final reward, and notes.
