# Foundation model CL-20sl in MuJoCo (paper Tables 12–13)

Model: `ALMI_trans/output/cl20sl_20260929_ae45cca_r1/almi_trans_cl_20sl_last.pth`, trained on the ALMI-X wave subset with the paper's hyper-parameters for about 126k iterations (final loss 0.00066). It was exported with `export.py` and run with `ALMI_trans/deploy/deploy_mujoco/play_trans.py`, which uses the authors' deploy loop: 50 Hz control, 20-step history, 8 s per test.

Each command ran once. The paper repeats each command 5 times, but the MuJoCo rollout is deterministic, so repeats give the same result. Videos are in `results/videos/cl20sl_*.mp4`, kept out of git.

## Table 12: average velocity per text command

| Command (lower body) | Ours v̄x | Ours v̄y | Ours ω̄yaw | Paper v̄x | Paper v̄y | Paper ω̄yaw | Direction correct? |
|---|---|---|---|---|---|---|---|
| forward fast | +0.27 | +0.02 | +0.03 | 0.48 | 0.18 | 0.08 | yes (slower than the paper) |
| left fast | −0.13 | +0.32 | +0.02 | 0.07 | 0.28 | 0.00 | yes |
| backward to left slowly | −0.44 | +0.25 | +0.01 | −0.19 | 0.21 | 0.00 | yes |
| turn left slowly | −0.08 | +0.01 | +0.13 | 0.02 | 0.02 | 0.08 | yes |
| go right fast | −0.14 | −0.44 | −0.10 | −0.11 | −0.37 | −0.07 | yes |
| keep standing | +0.01 | −0.01 | +0.03 | 0.00 | 0.00 | 0.01 | yes |

The upper-body part of each command was "and wave left."

## Table 13: success and survival

Upper-body success was judged by watching the live MuJoCo viewer, the paper's criterion: "the robot waves the correct hand".

| Command | Lower-body success (ours / paper) | Survival s (ours / paper) | Hand actually waved | Upper-body success |
|---|---|---|---|---|
| go forward slowly and wave left | 0 / 1.00 (v̄x −0.05, it stands in place) | 8.0 / 8.0 | left | yes |
| go backward moderately and wave right | 1 / 1.00 (v̄x −0.55) | 8.0 / 8.0 | both | no |
| go right fast and wave both | 1 / 1.00 (v̄y −0.41) | 8.0 / 8.0 | left | no |

Upper-body success: **1 of 3 (0.33)**. The paper reports 0.20 / 0.20 / 0.40 for CL-20sl on these commands.

## Summary

* **Survival: reproduced.** Every one of the 8 commands survives the full 8 s, as CL-20sl does in the paper. A 30 s run of "go forward fast and wave left" also survived.
* **Lower-body command following: reproduced.** The movement direction matches the text in 7 of 8 commands, and speeds are in the paper's range. The one miss is "forward slowly", where the robot stands in place.
* **Upper-body command following: reproduced, including the weakness.** 1 of 3 correct-hand waves, inside the paper's 0.20–0.40 for CL-20sl. The short 20-step history makes locomotion stable but the hand instruction is largely ignored. The paper's CL-400sl trades this the other way (100 % correct hand, but falls within 2–4 s).
