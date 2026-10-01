"""Run ALMI's open-loop (OL) foundation model in MuJoCo from a text command.

The released deploy script only runs the closed-loop models. OL works differently: the
Transformer generates the whole sequence of VQ-VAE tokens from the text command alone,
the VQ-VAE decoder turns them into a 71-dim state-action sequence (4 steps per token),
and the action part of each step is executed open-loop at 50 Hz. After the sequence
ends, the last action is held. Reports the same measures as play_trans.py.

Run from ALMI_trans/:
  python deploy/deploy_mujoco/play_ol.py --trans output/<ol run>/almi_trans_ol_last.pth \
      --vq output/<vq run>/vq_net_last.pth --text "Robot go forward slowly and wave left." \
      [--record videos/ol.mp4 --no_viewer]
"""
import argparse
import os
import sys
import time
from types import SimpleNamespace

import clip
import mujoco
import numpy as np
import torch
import yaml

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import models.t2m_trans as trans  # noqa: E402
import models.vqvae as vqvae  # noqa: E402
from play_trans import gravity_orientation, world_to_body  # noqa: E402

if not torch.cuda.is_available():  # models/quantize_cnn.py calls .cuda() unconditionally; run on CPU instead
    torch.Tensor.cuda = lambda self, *args, **kwargs: self

# the arguments train_almi_vq.sh and train_almi_ol.sh use
VQ_ARGS = SimpleNamespace(dataname="almi", quantizer="ema_reset", mu=0.99, nb_code=1024, code_dim=512,
                          output_emb_width=512, down_t=2, stride_t=2, width=512, depth=3,
                          dilation_growth_rate=3, vq_act="relu", vq_norm=None)
ACTION = slice(48, 69)  # the 21 actions inside the 71-dim state-action vector


def generate_actions(trans_path, vq_path, text):
    net = vqvae.VQVAE(VQ_ARGS, VQ_ARGS.nb_code, VQ_ARGS.code_dim, VQ_ARGS.output_emb_width, VQ_ARGS.down_t,
                      VQ_ARGS.stride_t, VQ_ARGS.width, VQ_ARGS.depth, VQ_ARGS.dilation_growth_rate, VQ_ARGS.vq_act)
    net.load_state_dict(torch.load(vq_path, map_location="cpu")["net_vq"], strict=True)
    net.eval()
    model = trans.Text2Motion_Transformer(num_vq=1024, embed_dim=256, clip_dim=512, block_size=176,
                                          num_layers=9, n_head=16, drop_out_rate=0.1, fc_rate=4)
    model.load_state_dict(torch.load(trans_path, map_location="cpu")["trans"], strict=True)
    model.eval()
    clip_model, _ = clip.load("./pretrained/ViT-B-32.pt", device="cpu", jit=False)
    clip_model.eval()
    with torch.no_grad():
        feat = clip_model.encode_text(clip.tokenize(text, truncate=True)).float()
        tokens = model.sample(feat, if_categorial=False)  # greedy: deterministic
        if tokens is None or len(tokens) == 0 or tokens.shape[-1] == 0:
            raise RuntimeError("the model produced an empty token sequence")
        seq = net.forward_decoder(tokens)  # (1, 4 * n_tokens, 71)
    return tokens.shape[-1], seq[0, :, ACTION].numpy().astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trans", required=True)
    p.add_argument("--vq", required=True)
    p.add_argument("--text", default="Robot go forward slowly and wave left.")
    p.add_argument("--config", default="deploy/deploy_mujoco/configs/h1_2_21dof_trans.yaml")
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--record", default="")
    p.add_argument("--no_viewer", action="store_true")
    a = p.parse_args()

    n_tokens, actions = generate_actions(a.trans, a.vq, a.text)
    print(f"generated {n_tokens} tokens -> {len(actions)} steps ({len(actions) / 50:.1f} s of motion)")

    with open(a.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    dt, decimation = cfg["simulation_dt"], cfg["control_decimation"]
    kps, kds = np.array(cfg["kps"], np.float32), np.array(cfg["kds"], np.float32)
    default = np.array(cfg["default_angles"], np.float32)
    m = mujoco.MjModel.from_xml_path(cfg["xml_path"])
    d = mujoco.MjData(m)
    m.opt.timestep = dt

    renderer = writer = cam = None
    if a.record:
        import imageio.v2 as imageio
        os.makedirs(os.path.dirname(os.path.abspath(a.record)), exist_ok=True)
        m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720  # the model's default buffer is 640x480
        renderer = mujoco.Renderer(m, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        cam.trackbodyid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        cam.distance, cam.elevation, cam.azimuth = 3.5, -15, 120
        writer = imageio.get_writer(a.record, fps=round(1 / (dt * decimation)), quality=8)
    viewer = None
    if not a.no_viewer:
        from mujoco import viewer as mj_viewer
        viewer = mj_viewer.launch_passive(m, d)

    target, frame, counter = default.copy(), 0, 0
    vel_log, fell_at, upright_steps = [], None, 0
    for _ in range(int(a.duration / dt)):
        if viewer is not None and not viewer.is_running():
            break
        t0 = time.time()
        d.ctrl[:] = (target - d.qpos[7:]) * kps - d.qvel[6:] * kds
        mujoco.mj_step(m, d)
        counter += 1
        if counter % decimation:
            continue
        target = actions[min(frame, len(actions) - 1)] * cfg["action_scale"] + default
        frame += 1
        quat = d.qpos[3:7]
        v = world_to_body(quat, d.qvel[0:3])
        vel_log.append([v[0], v[1], d.qvel[5]])
        if fell_at is None and (d.qpos[2] < 0.5 or gravity_orientation(quat)[2] > -0.5):
            fell_at, upright_steps = counter * dt, len(vel_log)
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
    upright = vel_log[:upright_steps] if fell_at is not None else vel_log
    v = np.mean(upright, axis=0) if upright else np.zeros(3)
    print(f'text: "{a.text}"')
    print(f"survival time: {fell_at if fell_at is not None else a.duration:.2f} s of {a.duration:.0f} s"
          f"{'' if fell_at is None else ' (fell)'}")
    print(f"average velocity while upright: vx {v[0]:+.2f} m/s, vy {v[1]:+.2f} m/s, yaw rate {v[2]:+.2f} rad/s")
    if a.record:
        print(f"video: {a.record}")


if __name__ == "__main__":
    main()
