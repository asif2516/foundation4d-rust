#!/usr/bin/env python3
"""Assemble the Foundation4D demo GIF from captured CDP frames."""
import os
import subprocess
import sys

SRC = "/tmp/f4d_gif"
OUT = "/home/z/my-project/download/foundation4d_demo.gif"
FRAME_MS = 130          # replay speed for animation frames
HOLD_MS = 1600          # hold time for tab screenshots

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    entries = []
    # Animation frames in capture order (recon_* include play + orbit segments).
    anim = sorted(f for f in os.listdir(SRC) if f.startswith("recon_") and f.endswith(".png"))
    tabs = ["cameras.png", "cameras_depth.png", "bev.png", "arch.png"]
    seq = [(anim, FRAME_MS)]
    # Interleave: first half of animation, then tabs, then rest.
    mid = len(anim) // 2
    seq = [
        (anim[:mid], FRAME_MS),
        (tabs, HOLD_MS),
        (anim[mid:], FRAME_MS),
    ]
    lines = ["ffconcat version 1.0"]
    n = 0
    for files, ms in seq:
        for f in files:
            p = os.path.join(SRC, f)
            if not os.path.isfile(p):
                continue
            lines.append(f"file '{p}'")
            lines.append(f"duration {ms / 1000.0:.3f}")
            n += 1
    # concat demuxer quirk: repeat the final entry.
    lines.append(f"file '{os.path.join(SRC, anim[-1])}'")
    lst = "/tmp/f4d_gif/list.txt"
    with open(lst, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{n} frames -> {OUT}")
    vf = ("scale=900:-2:flags=lanczos,split[a][b];"
          "[a]palettegen=stats_mode=diff:max_colors=192[p];"
          "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", lst,
        "-vf", vf,
        "-loop", "0", OUT,
    ]
    subprocess.run(cmd, check=True)
    size = os.path.getsize(OUT)
    print(f"done: {OUT} ({size / 1e6:.1f} MB)")

if __name__ == "__main__":
    sys.exit(main())
