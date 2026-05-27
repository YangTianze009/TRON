"""
Ego Motion Compound QA (v4 G14, for Motion-Cam (spatial)).

Targets: spatial-reasoning Motion-Cam -5.41. Failure mode: 100% of flipped cases
had CoT collapse to bare letter. Task requires tracing ego-motion through
compound direction (forward-left, back-right).

Task (non-THOR, 2D top-down): show a room schematic with objects.
Arrow #1 = camera position + facing at time t1. Arrow #2 = camera
position + facing at time t2. Ask: in which direction (forward / back /
left / right / compound like forward-left) did the camera move relative
to its original facing?

Reward: MCQ letter match.

Level axes:
  A) Options: 4-way at L0-3, 8-way at L4+
  B) Rotation during movement: no at L0-5, small yaw change at L6+
"""
import math
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_DIRECTIONS_4 = ["forward", "back", "left", "right"]
_DIRECTIONS_8 = ["forward", "forward-right", "right", "back-right",
                  "back", "back-left", "left", "forward-left"]

_TEMPLATES = [
    "The top-down room layout shows two camera positions: position 1 (blue) and position 2 (red). Each has an arrow showing the direction the camera is facing. Both arrows face the same direction. In which direction did the camera move (relative to its facing) between position 1 and position 2? Options: {opts}. Put the letter in <answer>...</answer>.",
    "Two camera positions (1 and 2) are shown in the top-down room. Both face the same direction. In which direction did the camera move between them, relative to its facing? Pick from {opts}. Put letter in <answer>...</answer>.",
    "Top-down: camera position 1 → position 2 (both facing the same way). Compute the direction of motion relative to the camera's facing. Options: {opts}. Put letter in <answer>...</answer>.",
    "The camera moves from position 1 to position 2 (both arrows same direction). Direction of motion relative to facing? Options: {opts}. Put letter in <answer>...</answer>.",
    "Given the two camera positions and their facings, which direction did the camera move relative to its facing? {opts}. Put letter in <answer>...</answer>.",
    "Camera moved from 1 to 2, maintaining facing. Which direction is 2 from 1, relative to facing? {opts}. Put letter in <answer>...</answer>.",
    "Identify the ego-relative direction of movement from position 1 to position 2 (same facing). {opts}. Put letter in <answer>...</answer>.",
    "The top-down shows camera positions 1 and 2 with identical facing arrows. Movement direction relative to facing? {opts}. Put letter in <answer>...</answer>.",
    "Determine the camera's movement direction (ego-relative) from position 1 to 2. {opts}. Put letter in <answer>...</answer>.",
    "Compute the ego-relative movement direction. Options: {opts}. Put letter in <answer>...</answer>.",
    "In which direction did the camera move relative to its own facing? {opts}. Put letter in <answer>...</answer>.",
    "The camera's ego-relative movement. Options: {opts}. Put letter in <answer>...</answer>.",
    "From position 1 to position 2, movement is in which direction relative to facing? {opts}. Put letter in <answer>...</answer>.",
    "Ego-motion problem: direction 1→2 relative to facing. {opts}. Put letter in <answer>...</answer>.",
    "Compute the direction of 2 from 1, relative to the camera's facing. {opts}. Put letter in <answer>...</answer>.",
    "Ego-relative motion direction (1→2)? {opts}. Put letter in <answer>...</answer>.",
]

class EgoMotionCompoundQA(StandaloneVisualEnv):
    ENV_NAME = "ego_motion_compound"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 3:
            dirs = _DIRECTIONS_4
        else:
            dirs = _DIRECTIONS_8
        return {"dirs": dirs}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 547)
        self._primary_complexity_feature = level

        # Pick a yaw for the camera (0 = facing "north" in 2D = +y)
        yaw_deg = rng.choice([0, 45, 90, 135, 180, 225, 270, 315])
        yaw = math.radians(yaw_deg)
        # Position 1 somewhere in the room
        room_W, room_H = 10, 8
        x1 = rng.randint(3, 7)
        y1 = rng.randint(3, 5)
        # Pick a movement direction relative to facing
        direction = rng.choice(cfg["dirs"])
        # Convert to world dxy (camera facing is yaw)
        # forward = (sin(yaw), cos(yaw))  — local +y
        # right = (cos(yaw), -sin(yaw))   — local +x
        fwd = (math.sin(yaw), math.cos(yaw))
        rgt = (math.cos(yaw), -math.sin(yaw))

        step = 2.0
        comp_map = {
            "forward": (fwd, 0),
            "back": ((-fwd[0], -fwd[1]), 0),
            "left": ((-rgt[0], -rgt[1]), 0),
            "right": (rgt, 0),
            "forward-right": ((fwd[0] + rgt[0], fwd[1] + rgt[1]), 0),
            "forward-left": ((fwd[0] - rgt[0], fwd[1] - rgt[1]), 0),
            "back-right": ((-fwd[0] + rgt[0], -fwd[1] + rgt[1]), 0),
            "back-left": ((-fwd[0] - rgt[0], -fwd[1] - rgt[1]), 0),
        }
        dxy, _ = comp_map[direction]
        # normalize
        n = math.hypot(dxy[0], dxy[1]) or 1.0
        dxy = (dxy[0] / n, dxy[1] / n)
        x2 = x1 + dxy[0] * step
        y2 = y1 + dxy[1] * step

        options = list(cfg["dirs"])
        rng.shuffle(options)
        correct_idx = options.index(direction)
        letter = "ABCDEFGH"[correct_idx]
        opts_str = ", ".join(f"{'ABCDEFGH'[i]}. {opt}" for i, opt in enumerate(options))

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(opts=opts_str)
        if level <= 2:
            q += (
                "\nHint: position 1 (blue) is the starting location and "
                "facing direction. Position 2 (red) is the final location. "
                "Determine the displacement (Δx, Δy) from 1 to 2, then "
                "decompose relative to the facing direction: forward = "
                "component along yaw vector; right = component along yaw + "
                "90° CW. The dominant component gives the move direction."
            )

        img = self._render(x1, y1, x2, y2, yaw, room_W, room_H, rng)
        return q, letter, img

    def _render(self, x1, y1, x2, y2, yaw, W, H, rng):
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-0.5, W + 0.5)
        ax.set_ylim(-0.5, H + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Room
        ax.add_patch(mpatches.Rectangle((0, 0), W, H, fc="#f5f5f5",
                                         ec="black", lw=2.0))

        # Direction arrows at both positions
        arrow_len = 1.5
        dx = math.sin(yaw) * arrow_len
        dy = math.cos(yaw) * arrow_len
        # Position 1 (blue arrow)
        ax.annotate("", xy=(x1 + dx, y1 + dy), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="blue", lw=3))
        ax.text(x1 - 0.3, y1 - 0.3, "1", fontsize=14,
                fontweight="bold", color="blue")
        # Position 2 (red arrow)
        ax.annotate("", xy=(x2 + dx, y2 + dy), xytext=(x2, y2),
                    arrowprops=dict(arrowstyle="-|>", color="red", lw=3))
        ax.text(x2 - 0.3, y2 - 0.3, "2", fontsize=14,
                fontweight="bold", color="red")

        # Compass rose
        cx, cy = W + 0.1, 0
        ax.text(cx, cy + 0.5, "N", fontsize=10, ha="center", fontweight="bold", color="gray")
        ax.text(cx, cy - 0.1, "S", fontsize=10, ha="center", fontweight="bold", color="gray")
        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_emc"
    os.makedirs(out_dir, exist_ok=True)
    env = EgoMotionCompoundQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 113
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[emc L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/emc_s{s}_L{level}.png")
            print(f"[emc L{level} s{s}] A={env._answer}")
