"""
Renders PickPlaceCan with the default can and several YCB objects, for visual inspection.

Writes one PNG per object (left: agentview, right: frontview, after 10 zero-action steps) plus all_objects.png
stacking them.

Usage:
    MUJOCO_GL=egl python tests/ycb/render_pick_place.py [--out renders/pick_place_ycb] [--objects 011_banana 025_mug]
"""
import argparse
import os

import imageio
import numpy as np
import robosuite as suite

DEFAULT_OBJECTS = ["011_banana", "003_cracker_box", "006_mustard_bottle", "025_mug", "035_power_drill", "029_plate"]


def render(object_type, size):
    env = suite.make(
        "PickPlaceCan", robots="Panda", object_type=object_type, has_renderer=False, has_offscreen_renderer=True,
        use_camera_obs=True, use_object_obs=False, camera_names=["agentview", "frontview"],
        camera_heights=size, camera_widths=size,
    )
    obs = env.reset()
    for _ in range(10):  # let the object settle
        obs, *_ = env.step(np.zeros(env.action_dim))
    env.close()
    # images come out bottom-left origin; flip for viewing
    return np.concatenate([obs["agentview_image"][::-1], obs["frontview_image"][::-1]], axis=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="renders/pick_place_ycb")
    parser.add_argument("--objects", nargs="+", default=DEFAULT_OBJECTS, help="YCB ids (the default can is always included)")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    rows = []
    for obj in [None] + args.objects:
        row = render(obj, args.size)
        path = os.path.join(args.out, f"{obj or 'can_default'}.png")
        imageio.imwrite(path, row)
        rows.append(row)
        print("wrote", path, flush=True)
    imageio.imwrite(os.path.join(args.out, "all_objects.png"), np.concatenate(rows, axis=0))
    print("wrote", os.path.join(args.out, "all_objects.png"))
