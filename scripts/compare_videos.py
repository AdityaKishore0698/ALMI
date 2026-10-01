"""Side-by-side comparison video of several MuJoCo recordings of the same text command.

Each panel is labelled with its model name, the command is shown underneath, and an
optional fall time is marked ("FELL at 2.6 s") from that moment on.

Usage:
  python scripts/compare_videos.py \
      --panel cl20sl.mp4 "CL-20sl (20-step history)" - \
      --panel cl400sl.mp4 "CL-400sl (400-step history)" 2.64 \
      --panel ol.mp4 "OL (open-loop)" 1.16 \
      --text "Robot go forward fast and wave both." --out comparison.mp4
The third value of --panel is the fall time in seconds, or - if the robot never falls.
The video ends with the shortest recording. Needs imageio + imageio-ffmpeg + Pillow.
"""
import argparse

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def font(size):
    for path in ("/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/System/Library/Fonts/Helvetica.ttc",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def label(frame, title, fell_at, t, big, small):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 44], fill=(20, 20, 20))
    draw.text((16, 9), title, font=big, fill=(255, 255, 255))
    if fell_at is not None and t >= fell_at:
        tag = f"FELL at {fell_at:.1f} s"
        w = draw.textlength(tag, font=big)
        draw.rectangle([img.width - w - 32, 54, img.width - 12, 94], fill=(170, 30, 35))
        draw.text((img.width - w - 22, 61), tag, font=big, fill=(255, 255, 255))
    return np.asarray(img)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--panel", nargs=3, action="append", required=True, metavar=("VIDEO", "LABEL", "FELL"))
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    readers = [imageio.get_reader(video) for video, _, _ in a.panel]
    labels = [lab for _, lab, _ in a.panel]
    fells = [None if fell == "-" else float(fell) for _, _, fell in a.panel]
    fps = readers[0].get_meta_data()["fps"]
    big, small = font(24), font(22)
    writer = imageio.get_writer(a.out, fps=fps, quality=8, macro_block_size=8)
    for i, frames in enumerate(zip(*readers)):
        t = i / fps
        panels = [label(f[::2, ::2], lab, fell, t, big, small) for f, lab, fell in zip(frames, labels, fells)]
        spacer = np.full((panels[0].shape[0], 8, 3), 255, np.uint8)
        row = np.concatenate([x for pnl in panels for x in (pnl, spacer)][:-1], axis=1)
        foot = Image.new("RGB", (row.shape[1], 48), (245, 245, 245))
        d = ImageDraw.Draw(foot)
        caption = f"Command: “{a.text}”    t = {t:4.1f} s"
        d.text(((row.shape[1] - d.textlength(caption, font=small)) / 2, 12), caption, font=small, fill=(30, 30, 30))
        writer.append_data(np.concatenate([row, np.asarray(foot)], axis=0))
    writer.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
