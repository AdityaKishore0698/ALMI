"""Evaluate an ALMI policy pair (lower body + upper body) with the metrics of the
paper's Tables 4 and 6 (Shi et al., arXiv 2504.14305v3, Section 6).

Each difficulty level from Table 3 fixes the velocity command, the terrain level
and whether the robot is pushed. As in the paper, linear and angular velocity are
tested separately (the other set to zero):
  linear test   v = (vx, vy), yaw rate 0     -> Evel
  angular test  v = 0, yaw rate wz           -> Eang
and the remaining metrics are averaged over both tests.

Every environment replays one motion of the evaluation set for one episode.
Metrics are averaged over the steps an environment is alive, then over motions.
  Evel      |v_cmd_xy - v_xy| in the base frame                     (m/s)
  Eang      |w_cmd_z - w_z|                                         (rad/s)
  Ejpe      mean |q_ref - q| over the 9 upper-body joints           (rad)
  Ekpe      mean distance between actual and reference elbow and
            hand positions, pelvis frame, from URDF kinematics      (m)
  Eact_up   mean |a_t - a_(t-1)| of the upper-body joint targets    (rad)
  Eact_low  the same for the 12 lower-body joints                   (rad)
  Eg        mean |projected gravity_xy|                             (-)
  Survival  fraction of motions that end the episode without falling
The paper gives only these names, so the exact definitions are ours; see
docs/EVALUATION.md.

Usage (almi-rl env, from ALMI_RL/):
  python legged_gym/scripts/eval_almi.py --lower_policy <model_N.pt> --upper_policy <model_N.pt> \
      --motions ~/ALMI/data/eval/eval_motions.pkl \
      --motion_csv ~/ALMI/data/eval/eval_mean_episode_length.csv \
      --out results/eval_<label>.json --label "lower-3 + upper-2" [--levels easy,medium,hard]
"""
import argparse
import json
import os
import sys
import time
import xml.etree.ElementTree as ET


def parse_own_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--lower_policy", required=True)
    p.add_argument("--upper_policy", required=True)
    p.add_argument("--motions", required=True)
    p.add_argument("--motion_csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="")
    p.add_argument("--levels", default="easy,medium,hard")
    p.add_argument("--num_motions", type=int, default=0, help="0 = all motions in the file")
    p.add_argument("--steps", type=int, default=0, help="0 = one full episode")
    own, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest  # the rest goes to legged_gym's get_args
    return own


OWN = parse_own_args()

import isaacgym  # noqa: E402,F401  (must precede torch)
import joblib  # noqa: E402
import torch  # noqa: E402

from legged_gym import LEGGED_GYM_ROOT_DIR  # noqa: E402
from legged_gym.envs import *  # noqa: E402,F401,F403
from legged_gym.utils import get_args, task_registry  # noqa: E402

TASK = "h1_2_upper"  # runs the trained upper policy with a loaded lower policy
LEVELS = {  # Table 3
    "easy":   dict(vx=0.7, vy=0.0, wz=0.2, terrain=0, push=False),
    "medium": dict(vx=1.0, vy=0.3, wz=0.4, terrain=3, push=True),
    "hard":   dict(vx=1.3, vy=0.6, wz=0.6, terrain=6, push=True),
}
METRICS = ["Evel", "Eang", "Ejpe", "Ekpe", "Eact_up", "Eact_low", "Eg"]


class ArmKinematics:
    """Positions of the elbows and hands in the pelvis frame, from the URDF."""
    KEYPOINTS = ["left_elbow_pitch_link", "L_hand_base_link", "right_elbow_pitch_link", "R_hand_base_link"]

    def __init__(self, urdf, dof_names, device):
        joints = {}
        for j in ET.parse(urdf).getroot().findall("joint"):
            o, a = j.find("origin"), j.find("axis")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            axis = [float(v) for v in a.get("xyz").split()] if a is not None else [0.0, 0.0, 1.0]
            joints[j.find("child").get("link")] = (j.find("parent").get("link"), xyz, rpy, axis, j.get("name"), j.get("type"))
        self.device = device
        self.chains = []
        for link in self.KEYPOINTS:
            chain = []
            while link in joints:
                parent, xyz, rpy, axis, name, jtype = joints[link]
                idx = dof_names.index(name) if jtype == "revolute" and name in dof_names else None
                chain.append((self._origin(xyz, rpy), torch.tensor(axis, device=device), idx))
                link = parent
            self.chains.append(chain[::-1])  # pelvis first

    def _origin(self, xyz, rpy):
        r, p, y = [torch.tensor(v) for v in rpy]
        cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
        R = torch.tensor([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                          [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                          [-sp, cp * sr, cp * cr]])
        T = torch.eye(4)
        T[:3, :3], T[:3, 3] = R, torch.tensor(xyz)
        return T.to(self.device)

    @staticmethod
    def _rotation(axis, q):  # Rodrigues, batched over q
        k = axis / axis.norm()
        K = torch.zeros(3, 3, device=q.device)
        K[0, 1], K[0, 2], K[1, 0], K[1, 2], K[2, 0], K[2, 1] = -k[2], k[1], k[2], -k[0], -k[1], k[0]
        s, c = q.sin()[:, None, None], q.cos()[:, None, None]
        T = torch.eye(4, device=q.device).repeat(len(q), 1, 1)
        T[:, :3, :3] = torch.eye(3, device=q.device) + s * K + (1 - c) * (K @ K)
        return T

    def __call__(self, q):  # q: (N, num_dof) -> (N, K, 3)
        out = []
        for chain in self.chains:
            T = torch.eye(4, device=q.device).repeat(len(q), 1, 1)
            for origin, axis, idx in chain:
                T = T @ origin
                if idx is not None:
                    T = T @ self._rotation(axis, q[:, idx])
            out.append(T[:, :3, 3])
        return torch.stack(out, dim=1)


def build(own, args):
    n_file = len(joblib.load(os.path.expanduser(own.motions)))
    n = min(own.num_motions or n_file, n_file)

    env_cfg, train_cfg = task_registry.get_cfgs(name=TASK)
    env_cfg.env.num_envs = n
    env_cfg.env.test = False            # test=True sleeps to real time
    env_cfg.noise.add_noise = False
    env_cfg.terrain.curriculum = True   # keeps per-env terrain levels; frozen below
    env_cfg.commands.curriculum = False
    env_cfg.commands.heading_command = False
    env_cfg.asset.arm_curriculum = False  # reference motions at full amplitude
    env_cfg.asset.motion_curriculum = False
    env_cfg.asset.motion_path = os.path.expanduser(own.motions)
    env_cfg.asset.mean_episode_length_path = os.path.expanduser(own.motion_csv)
    env, _ = task_registry.make_env(name=TASK, args=args, env_cfg=env_cfg)

    # motion i on env i, also after every reset
    def alloc_motion(env_ids):
        for i in env_ids.tolist():
            env.motion_buffer[i] = torch.stack(env.new_motion[i % n][:env.motion_length])
            env.env_motion_dict[i] = i % n
    env._alloc_motion = alloc_motion

    train_cfg.runner.resume = False
    train_cfg.runner.lower_policy_path = own.lower_policy
    runner, _ = task_registry.make_alg_runner(env=env, name=TASK, args=args, train_cfg=train_cfg, log_root=None)
    runner.load(own.upper_policy, load_optimizer=False)
    runner.alg.actor_critic.eval()
    runner.lower_actor_critic.eval()

    urdf = env_cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
    kin = ArmKinematics(urdf, list(env.dof_names), env.device)
    return env, runner, kin, n


@torch.inference_mode()  # the reset must also run here: env buffers become inference tensors
def run_test(env, runner, kin, cmd, terrain, push, steps):
    ids = torch.arange(env.num_envs, device=env.device)
    env.cfg.terrain.curriculum = False  # no level changes from here on
    env.terrain_levels[:] = min(terrain, env.max_terrain_level - 1)
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    env.command_ranges["lin_vel_x"] = [cmd[0], cmd[0]]
    env.command_ranges["lin_vel_y"] = [cmd[1], cmd[1]]
    env.command_ranges["ang_vel_yaw"] = [cmd[2], cmd[2]]
    env.cfg.domain_rand.push_robots = push
    for ac in (runner.alg.actor_critic, runner.lower_actor_critic):
        ac.memory_a.hidden_states = None
    env.reset()
    obs, lower_obs = env.get_observations(), env.get_lower_observations()

    upper, lower = runner.alg.actor_critic.act_inference, runner.lower_actor_critic.act_inference
    scale = env.cfg.control.action_scale
    last_idx = env.motion_buffer.shape[1] - 1
    alive = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    sums = {k: torch.zeros(env.num_envs, device=env.device) for k in METRICS}
    counts = torch.zeros(env.num_envs, device=env.device)
    act_counts = torch.zeros(env.num_envs, device=env.device)
    prev_up = prev_low = None

    for t in range(steps):
        obs, _, _, dones, _, lower_obs = env.step(upper(obs), lower(lower_obs))
        alive &= ~(dones & ~env.time_out_buf)
        m = alive.float()

        ref = env.motion_buffer[ids, env.episode_length_buf.clamp(max=last_idx)]
        q_ref = env.dof_pos.clone()
        q_ref[:, -9:] = ref
        vals = {
            "Evel": (env.commands[:, :2] - env.base_lin_vel[:, :2]).norm(dim=1),
            "Eang": (env.commands[:, 2] - env.base_ang_vel[:, 2]).abs(),
            "Ejpe": (ref - env.dof_pos[:, -9:]).abs().mean(dim=1),
            "Ekpe": (kin(env.dof_pos) - kin(q_ref)).norm(dim=-1).mean(dim=1),
            "Eg": env.projected_gravity[:, :2].norm(dim=1),
        }
        for k, v in vals.items():
            sums[k] += v * m
        counts += m
        if prev_up is not None:
            sums["Eact_up"] += (env.actions - prev_up).abs().mean(dim=1) * scale * m
            sums["Eact_low"] += (env.lower_actions - prev_low).abs().mean(dim=1) * scale * m
            act_counts += m
        prev_up, prev_low = env.actions.clone(), env.lower_actions.clone()

    res = {}
    for k in METRICS:
        c = act_counts if k.startswith("Eact") else counts
        per_motion = sums[k][c > 0] / c[c > 0]
        res[k] = per_motion.mean().item() if len(per_motion) else float("nan")
    res["Survival"] = alive.float().mean().item()
    return res


def main():
    args = get_args()
    args.task, args.headless = TASK, True
    env, runner, kin, n = build(OWN, args)
    steps = OWN.steps or int(env.max_episode_length)
    print(f"[eval] {n} motions, {steps} steps per test, levels {OWN.levels}")

    results = {}
    for name in OWN.levels.split(","):
        lv = LEVELS[name]
        t0 = time.time()
        lin = run_test(env, runner, kin, (lv["vx"], lv["vy"], 0.0), lv["terrain"], lv["push"], steps)
        ang = run_test(env, runner, kin, (0.0, 0.0, lv["wz"]), lv["terrain"], lv["push"], steps)
        row = {k: (lin[k] + ang[k]) / 2 for k in METRICS + ["Survival"]}
        row["Evel"], row["Eang"] = lin["Evel"], ang["Eang"]
        results[name] = {"metrics": row, "linear_test": lin, "angular_test": ang, "level": lv}
        print(f"[eval] {name} ({time.time() - t0:.0f}s): " + ", ".join(f"{k} {v:.4f}" for k, v in row.items()))

    out = {
        "label": OWN.label, "lower_policy": OWN.lower_policy, "upper_policy": OWN.upper_policy,
        "motions": OWN.motions, "num_motions": n, "steps_per_test": steps, "results": results,
    }
    os.makedirs(os.path.dirname(os.path.abspath(OWN.out)), exist_ok=True)
    with open(OWN.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[eval] wrote {OWN.out}")
    print("| Level | Method | " + " | ".join(METRICS + ["Survival"]) + " |")
    for name, r in results.items():
        print(f"| {name} | {OWN.label} | " + " | ".join(f"{r['metrics'][k]:.4f}" for k in METRICS + ["Survival"]) + " |")


if __name__ == "__main__":
    main()
