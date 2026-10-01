"""Run an exported ALMI foundation model (e.g. CL-20sl) in MuJoCo from a text command.

Same control loop as deploy_mujoco_21dof_trans_clean.py (observation layout, history
window, PD gains, 50 Hz control), but without its legged_gym imports, with the command
and policy as arguments, optional MP4 recording, and the measures of the paper's
Tables 12-13: survival time and average base velocity (vx, vy, yaw rate).

Live viewer on macOS needs mjpython:
  mjpython deploy/deploy_mujoco/play_trans.py --policy output/export/policy_trans_cl20sl.pt \
      --text "Robot go forward slowly and wave left."
Video only (no window, any python):
  python deploy/deploy_mujoco/play_trans.py --policy ... --text "..." --record videos/fwd_wave_left.mp4 --no_viewer
Chunked models (output of 21*k per step) are executed with temporal ensembling: the action for the
current step is the weighted average of the k predictions made for it at the last k steps, with
weight exp(-m*i) for the prediction made i steps ago (--ensemble_m; ACT uses the opposite direction).
--log saves the joint positions and executed actions per control step (.npz) for offline scoring.
Run from ALMI_trans/.
"""
import argparse
import os
import time

import clip
import mujoco
import numpy as np
import torch
import yaml


def gravity_orientation(q):
    qw, qx, qy, qz = q
    return np.array([2 * (-qz * qx + qw * qy), -2 * (qz * qy + qw * qx), 1 - 2 * (qw * qw + qz * qz)])


def world_to_body(q, v):
    qw, qx, qy, qz = q
    R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                  [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                  [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
    return R.T @ v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="TorchScript file from export.py")
    p.add_argument("--text", default="Robot go forward slowly and wave left.")
    p.add_argument("--config", default="deploy/deploy_mujoco/configs/h1_2_21dof_trans.yaml")
    p.add_argument("--seq_len", type=int, default=20, help="history length the model was trained with")
    p.add_argument("--duration", type=float, default=25.0, help="seconds of simulated time (paper tests use 8)")
    p.add_argument("--record", default="", help="write an MP4 to this path")
    p.add_argument("--no_viewer", action="store_true")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--ensemble_m", type=float, default=0.1, help="temporal-ensembling weight decay (chunked models)")
    p.add_argument("--log", default="", help="save joint positions and actions per control step to this .npz")
    a = p.parse_args()

    with open(a.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    dt, decimation = cfg["simulation_dt"], cfg["control_decimation"]
    kps, kds = np.array(cfg["kps"], np.float32), np.array(cfg["kds"], np.float32)
    default = np.array(cfg["default_angles"], np.float32)
    n_act, n_obs = cfg["num_actions"], cfg["num_obs"]

    clip_model, _ = clip.load("./pretrained/ViT-B-32.pt", device="cpu", jit=False)
    clip_model.eval()
    with torch.no_grad():
        text_feat = clip_model.encode_text(clip.tokenize(a.text, truncate=True)).float()
    policy = torch.jit.load(a.policy)
    policy.eval()

    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = dt

    action = np.zeros(n_act, np.float32)
    target = default.copy()
    obs = np.zeros(n_obs, np.float32)
    history = torch.zeros(1, a.seq_len, n_obs)
    counter = frame = 0
    vel_log, fell_at, upright_steps = [], None, 0
    chunk = 0  # read from the first output
    pending = {}  # control step -> list of (age-weighted) predictions for it
    qpos_log, act_log = [], []

    renderer = writer = cam = None
    if a.record:
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(os.path.abspath(a.record)), exist_ok=True)
        m.vis.global_.offwidth, m.vis.global_.offheight = max(a.width, 640), max(a.height, 480)
        renderer = mujoco.Renderer(m, a.height, a.width)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        cam.distance, cam.elevation, cam.azimuth = 3.5, -15, 120
        writer = imageio.get_writer(a.record, fps=round(1 / (dt * decimation)), quality=8)

    viewer = None
    if not a.no_viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)

    n_steps = int(a.duration / dt)
    with torch.no_grad():
        for _ in range(n_steps):
            if viewer is not None and not viewer.is_running():
                break
            t0 = time.time()
            d.ctrl[:] = (target - d.qpos[7:]) * kps - d.qvel[6:] * kds
            mujoco.mj_step(m, d)
            counter += 1
            if counter % decimation:
                continue
            frame += 1

            quat = d.qpos[3:7]
            phase = counter * dt % 0.8 / 0.8
            obs[:3] = d.qvel[3:6] * cfg["ang_vel_scale"]
            obs[3:6] = gravity_orientation(quat)
            obs[6:27] = (d.qpos[7:] - default) * cfg["dof_pos_scale"]
            obs[27:48] = d.qvel[6:] * cfg["dof_vel_scale"]
            obs[48:69] = action
            obs[69:71] = [np.sin(2 * np.pi * phase), np.sin(2 * np.pi * ((phase + 0.5) % 1))]
            o = torch.from_numpy(obs).unsqueeze(0)
            if frame <= a.seq_len:
                history[:, frame - 1] = o
            else:
                history = torch.cat((history[:, 1:], o.unsqueeze(1)), dim=1)
            out = policy(history, text_feat).numpy().squeeze()
            pred = (out[frame] if frame < a.seq_len else out[-1]).astype(np.float32)
            chunk = chunk or len(pred) // n_act
            if chunk == 1:
                action = pred
            else:  # temporal ensembling over the overlapping chunks
                for i, a_future in enumerate(pred.reshape(chunk, n_act)):
                    pending.setdefault(frame + i, []).append(a_future)
                preds = pending.pop(frame)  # oldest first; the newest was made now (age 0)
                w = np.exp(-a.ensemble_m * np.arange(len(preds))[::-1])
                action = (np.average(preds, axis=0, weights=w)).astype(np.float32)
            target = action * cfg["action_scale"] + default

            qpos_log.append(d.qpos[7:].copy())
            act_log.append(action.copy())
            v = world_to_body(quat, d.qvel[0:3])
            vel_log.append([v[0], v[1], d.qvel[5]])
            if fell_at is None and (d.qpos[2] < 0.5 or gravity_orientation(quat)[2] > -0.5):
                fell_at = counter * dt  # pelvis low or torso tilted > 60 degrees
                upright_steps = len(vel_log)

            if writer is not None:
                renderer.update_scene(d, camera=cam)
                writer.append_data(renderer.render())
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0.0, dt * decimation - (time.time() - t0)))

    if writer is not None:
        writer.close()
    if viewer is not None:
        viewer.close()
    # average only while upright: after a fall the robot slides on the ground
    upright = vel_log[:upright_steps] if fell_at is not None else vel_log
    v = np.mean(upright, axis=0) if upright else np.zeros(3)
    print(f'text: "{a.text}"')
    print(f"survival time: {fell_at if fell_at is not None else a.duration:.2f} s of {a.duration:.0f} s"
          f"{'' if fell_at is None else ' (fell)'}")
    print(f"average velocity while upright: vx {v[0]:+.2f} m/s, vy {v[1]:+.2f} m/s, yaw rate {v[2]:+.2f} rad/s")
    if a.record:
        print(f"video: {a.record}")
    if a.log:
        os.makedirs(os.path.dirname(os.path.abspath(a.log)), exist_ok=True)
        np.savez(a.log, text=a.text, qpos=np.array(qpos_log), action=np.array(act_log), dt=dt * decimation,
                 fell_at=-1.0 if fell_at is None else fell_at, chunk=chunk, ensemble_m=a.ensemble_m)


if __name__ == "__main__":
    main()
