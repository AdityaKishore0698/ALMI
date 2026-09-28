"""Build the held-out evaluation motion set for eval_almi.py.

The paper evaluates on 1122 CMU MoCap clips retargeted to H1-2, which were not
released (docs/REPRODUCTION_GAPS.md, G2). As a substitute we use the ALMI-X
selected KIT motions (select_motions.zip) minus every motion that appears in the
training file resources/motions/all_wave.pkl.

Writes, in the format the H1-2 environments load:
  <out>/eval_motions.pkl                 list[motion] of list[frame] of ndarray(1, 19)
  <out>/eval_motions_names.txt           one motion name per line, same order
  <out>/eval_mean_episode_length.csv     env_id,mean_episode_length that keeps the order

Usage (almi-rl env):
  python scripts/prepare_eval_motions.py \
      --select ~/ALMI/data/ALMI-X/extracted/select_motions/select_motions \
      --train ALMI_RL/resources/motions/all_wave.pkl --out ~/ALMI/data/eval
"""
import argparse
import os

import joblib
import numpy as np


def motion_key(motion):
    # the first frames identify a motion; rounding absorbs float noise
    return np.round(np.concatenate([np.ravel(f) for f in motion[:5]]), 4).tobytes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--select", required=True, help="folder with select_motion_dof_pos.pkl and select_motion_name.pkl")
    p.add_argument("--train", required=True, help="training motion file to exclude (all_wave.pkl)")
    p.add_argument("--out", required=True)
    a = p.parse_args()

    motions = joblib.load(os.path.join(a.select, "select_motion_dof_pos.pkl"))
    names = joblib.load(os.path.join(a.select, "select_motion_name.pkl"))
    train_keys = {motion_key(m) for m in joblib.load(a.train)}

    keep = [i for i, m in enumerate(motions) if motion_key(m) not in train_keys]
    held_out = [motions[i] for i in keep]
    print(f"{len(motions)} selected motions, {len(motions) - len(keep)} used in training, {len(keep)} held out")

    os.makedirs(a.out, exist_ok=True)
    joblib.dump(held_out, os.path.join(a.out, "eval_motions.pkl"))
    with open(os.path.join(a.out, "eval_motions_names.txt"), "w") as f:
        f.writelines(names[i] + "\n" for i in keep)
    # the env sorts motions by descending mean_episode_length; descending ids keep the file order
    with open(os.path.join(a.out, "eval_mean_episode_length.csv"), "w") as f:
        f.write("env_id,mean_episode_length\n")
        f.writelines(f"{i},{len(keep) - i}\n" for i in range(len(keep)))


if __name__ == "__main__":
    main()
