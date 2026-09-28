# Evaluation: reproducing Tables 4 and 6

The authors did not release an evaluation script (gap G1), so we wrote `ALMI_RL/legged_gym/scripts/eval_almi.py`. This page records how it maps to the paper and where it has to guess.

## What the paper specifies

From Section 6 and Table 3 of arXiv 2504.14305v3:

* Metrics: Evel (m/s), Eang (rad/s), upper-body Ejpe and Ekpe, upper and lower Eaction ("joint difference", rad), Eg ("projected gravity"), and survival rate.
* Three difficulty levels:

| Level | v̂x | v̂y | ω̂yaw | Terrain level | Push |
|---|---|---|---|---|---|
| easy | 0.7 | 0.0 | 0.2 | 0 | no |
| medium | 1.0 | 0.3 | 0.4 | 3 | yes |
| hard | 1.3 | 0.6 | 0.6 | 6 | yes |

* Linear and angular velocity are tested separately, with the other set to zero, and the metrics are averaged.
* Evaluation set: 1122 CMU MoCap clips, simulated in Isaac Gym.

## What our script does

| Item | Our choice | Why |
|---|---|---|
| Policy pair | The `h1_2_upper` environment runs the upper policy; the lower policy is loaded from its checkpoint | Same composition as training and the released play script |
| Tests per level | A linear test (v̂x, v̂y, 0) gives Evel; an angular test (0, 0, ω̂yaw) gives Eang; every other metric is the mean of the two tests | Matches "assess separately … report the average" |
| Episode | One full episode (1000 steps = 20 s) per motion, one motion per environment | The released environments use 1000-step episodes |
| Averaging | Mean over the steps a motion is alive, then mean over motions | Each motion counts equally |
| Survival | Fraction of motions whose episode ends without a termination | Termination = pelvis contact or excessive roll/pitch, as in training |
| Evel | ‖v̂xy − vxy‖ in the base frame | |
| Eang | \|ω̂z − ωz\| | |
| Ejpe | Mean absolute error over the 9 upper-body joints | In radians. The paper writes "m", which looks like a typo for joint angles |
| Ekpe | Mean distance between actual and reference elbow and hand positions, in the pelvis frame | Computed from the URDF by forward kinematics, checked against Isaac Gym to 0.01 mm |
| Eaction (upper / lower) | Mean \|aₜ − aₜ₋₁\| of the joint targets (action × 0.25) | "Joint difference" read as action smoothness |
| Eg | Mean ‖gravity projected into the base frame, x-y part‖ | **Differs from the paper:** their values (0.69–1.2) are too large for this definition, which is at most 1 and about 0.05 for upright walking, so their exact definition is unknown |
| Observation noise | Off | Evaluation, not training |
| Other domain randomisation | As in training | The paper does not say |

## Evaluation set (gap G2)

`scripts/prepare_eval_motions.py` builds it from ALMI-X's 835 selected KIT motions, minus the 88 motions used in training (`all_wave.pkl`). That leaves **747 held-out motions**. These are KIT motions, not the paper's CMU clips, so absolute numbers can differ even with a perfect reproduction. Comparisons between our own policy pairs (Table 6) are the reliable part.

## Running it

```bash
# once
python scripts/prepare_eval_motions.py --select ~/ALMI/data/ALMI-X/extracted/select_motions/select_motions \
    --train ALMI_RL/resources/motions/all_wave.pkl --out ~/ALMI/data/eval
# one pair
cd ALMI_RL && python legged_gym/scripts/eval_almi.py --lower_policy <lower.pt> --upper_policy <upper.pt> \
    --motions ~/ALMI/data/eval/eval_motions.pkl --motion_csv ~/ALMI/data/eval/eval_mean_episode_length.csv \
    --label "lower-3 + upper-2" --out results/lower3_upper2.json
# all Table 6 pairs, after training (waits for ~/ALMI/logs/almi_final_policies.txt)
bash scripts/eval_all.sh
```

A full evaluation of one pair is 3 levels × 2 tests × 1000 steps on 747 environments.

## Validation so far

* Kinematics: the FK keypoints match Isaac Gym's rigid-body positions to within 0.01 mm after random arm motion.
* Smoke run (32 motions, 300 steps, lower-1 + upper-1 at iteration 1400): easy gave Evel 0.169, Eang 0.287 and survival 1.00; hard gave Evel 0.522, Eang 0.515 and survival 0.84. The magnitudes are close to the paper's, apart from Eg.
