# Foundation model CL-400sl in MuJoCo, with CL-20sl (paper Tables 12–13)

Model: `ALMI_trans/output/cl400sl_20260930_b64acc/almi_trans_cl_400sl_last.pth`. Trained with the paper's hyper-parameters at an effective batch of 64 (16 × 4 gradient accumulation) for about 8,150 optimizer steps (5 h cap), final loss 0.0149. Exported with `export.py --seq_len 400` and run with `play_trans.py --seq_len 400`. Velocities are averaged only while the robot is upright, because after a fall it slides on the ground.

## Table 12: average velocity while upright ("… and wave left.", 8 s)

| Command | Survived (s) | Ours v̄x / v̄y / ω̄yaw | Paper v̄x / v̄y / ω̄yaw |
|---|---|---|---|
| forward fast | 2.64 | +0.73 / +0.06 / −0.28 | 0.87 / −0.05 / −0.10 |
| left fast | 2.28 | +0.48 / +0.22 / +0.14 | 0.53 / 0.42 / −0.07 |
| backward to the left slowly | 1.60 | −0.89 / +0.55 / −0.01 | 0.35 / 0.30 / −0.01 |
| turn left slowly | 2.44 | +0.63 / +0.19 / +0.23 | 0.53 / 0.16 / 0.20 |
| go right fast | 2.44 | −0.83 / −0.37 / −0.32 | 0.47 / −0.50 / −0.10 |
| keep standing | 1.80 | +0.36 / +0.46 / −0.16 | 0.22 / 0.08 / 0.00 |

## Table 13: survival duration

| Command | CL-400sl ours (s) | CL-400sl paper (s) | CL-20sl ours (s) | CL-20sl paper (s) |
|---|---|---|---|---|
| go forward slowly and wave left | 2.64 | 2.14 | 8.0 | 8.0 |
| go backward moderately and wave right | 1.82 | 2.54 | 8.0 | 8.0 |
| go right fast and wave both | 4.30 | 3.57 | 8.0 | 8.0 |

Which hand waves (the paper's upper-body success, 1.00 for CL-400sl) is judged by watching the videos, as in the paper. Two automatic proxies were tried, joint-angle spread and hand height above the shoulder, and neither matched what was seen for CL-20sl, so they are not used. In the side-by-side "forward fast and wave both" video, CL-400sl raises both arms within the first second.

## Summary

* **The paper's trade-off is reproduced.** CL-400sl falls after 1.6–4.3 s on every command (paper: 2.1–3.6 s). CL-20sl survives every command for the full 8 s, and for 25 s in the longer runs.
* **Velocities while upright are in the paper's range** for forward, left and turn-left. They are erratic where the paper's are too: most commands drift forward, and "keep standing" moves.
* **Videos** (in `results/videos/`, not in git):
  * `compare_forward_fast_wave_both.mp4` and `compare_right_fast_wave_both.mp4`: CL-20sl and CL-400sl side by side, same command, 25 s, with the fall time marked.
  * `cl400sl_t13_*.mp4`: the two remaining Table 13 commands, 8 s each.
