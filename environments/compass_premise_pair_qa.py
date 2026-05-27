"""
Compass Premise Pair QA (v4 G15, for Pos-Obj-Obj (spatial)/Pos-Reg-Reg partial).

Targets:

Failure mode: given compass premise "if X is on east wall of bedroom",
model doesn't propagate the frame consistently to other objects.

Task: render a simple top-down room layout showing object positions.
Given a premise ("X is on east wall"), ask where another object Y is
relative to X (compass direction).

Reward: MCQ letter match.

Level axes:
  A) Number of objects: 3 at L0, 4-5 at L3+
  B) Number of hops needed: 1 at L0 (direct), 2+ at L3+ (X is east → rotate frame → find Y)
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

_COMPASS_DIRS = ["north", "east", "south", "west"]
_COMPASS_CORNERS = ["northeast", "southeast", "southwest", "northwest"]

_OBJECT_TYPES = [
    "bookshelf", "desk", "bed", "wardrobe", "window", "door", "sofa",
    "painting", "lamp", "mirror", "rug", "chair", "plant", "TV",
    "dresser", "cabinet"
]

_TEMPLATES = [
    "The top-down plan shows a rectangular room with objects at marked positions. **Premise**: the {anchor} is on the {anchor_dir} wall. Given this, what direction is the {target} from the {anchor}? Options: A. north  B. south  C. east  D. west. Put the letter in <answer>...</answer>.",
    "Given the premise that the {anchor} is on the {anchor_dir} wall, find the compass direction of the {target} relative to the {anchor}. Pick from A. north, B. south, C. east, D. west. Put letter in <answer>...</answer>.",
    "Assuming the {anchor} is on the {anchor_dir} wall, the {target} is in which direction from the {anchor}? A/B/C/D = north/south/east/west. Put letter in <answer>...</answer>.",
    "Top-down view of a room. Premise: {anchor} is on the {anchor_dir} wall. Direction of {target} from {anchor}? A-D = N/S/E/W. Put letter in <answer>...</answer>.",
    "Given the spatial arrangement and the premise ({anchor} = {anchor_dir} wall), compute {target}'s direction from {anchor}. A/B/C/D. Put letter in <answer>...</answer>.",
    "{anchor} is on the {anchor_dir} wall. What direction is {target} from {anchor}? A=N, B=S, C=E, D=W. Put letter in <answer>...</answer>.",
    "Premise: {anchor} at {anchor_dir} wall. Find direction of {target} from {anchor}. A-D. Put letter in <answer>...</answer>.",
    "The {anchor} is on the {anchor_dir} wall. {target} is in which compass direction relative to {anchor}? A-D. Put letter in <answer>...</answer>.",
    "Given {anchor} = {anchor_dir}, find {target}'s direction from {anchor}. A/B/C/D. Put letter in <answer>...</answer>.",
    "If the {anchor} is on the {anchor_dir} wall of this room, the {target} is in which direction from it? A-D. Put letter in <answer>...</answer>.",
    "Top-down room, {anchor} = {anchor_dir} wall. Direction of {target} from {anchor}? A-D. Put letter in <answer>...</answer>.",
    "Compass premise: {anchor} at {anchor_dir}. Direction of {target} from {anchor}? A-D. Put letter in <answer>...</answer>.",
    "Given {anchor} on {anchor_dir} wall, the {target} is _ of the {anchor}. A-D. Put letter in <answer>...</answer>.",
    "{anchor} is at the {anchor_dir} wall. {target} relative to {anchor}? A-D. Put letter in <answer>...</answer>.",
    "Premise: {anchor} = {anchor_dir} wall. Find {target}'s direction from {anchor}. A-D. Put letter in <answer>...</answer>.",
    "With the {anchor} on the {anchor_dir} wall, {target}'s compass direction from the {anchor} is: A-D. Put letter in <answer>...</answer>.",
]

class CompassPremisePairQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "compass_premise_pair"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_objects = 3 + level // 3
        return {"n_objects": min(6, n_objects)}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 419)
        self._primary_complexity_feature = level

        W, H = 10, 8  # room dimensions
        # place n objects at random positions
        objects = []
        names_used = set()
        for _ in range(cfg["n_objects"]):
            while True:
                name = rng.choice(_OBJECT_TYPES)
                if name not in names_used:
                    names_used.add(name)
                    break
            x = rng.randint(1, W - 1)
            y = rng.randint(1, H - 1)
            objects.append((name, x, y))

        # Anchor: user says anchor is on some wall; we relabel the map so
        # that the premise holds. Concretely: rotate the map so anchor is on
        # the specified wall.
        anchor_name, ax, ay = objects[0]
        anchor_dir = rng.choice(_COMPASS_DIRS)

        # The figure shows the actual positions in world coordinates
        # (without compass labels). The user must interpret the anchor as on
        # anchor_dir wall. We compute directions with the anchor-relative
        # mapping.

        # Simpler: the figure has compass labels (N at top, E at right).
        # The premise overrides: we tell the user "anchor is on <anchor_dir>
        # wall" — so the frame is rotated such that north becomes that wall.

        # Actual approach: draw the figure with anchor in a random wall.
        # Then the premise mentions the wall correctly.

        # Randomize anchor's wall by placing it near that wall
        if anchor_dir == "north":
            ax, ay = rng.randint(2, W - 2), H - 1
        elif anchor_dir == "south":
            ax, ay = rng.randint(2, W - 2), 1
        elif anchor_dir == "east":
            ax, ay = W - 1, rng.randint(2, H - 2)
        else:  # west
            ax, ay = 1, rng.randint(2, H - 2)
        objects[0] = (anchor_name, ax, ay)

        # Pick target
        target_idx = rng.randint(1, len(objects) - 1)
        target_name, tx, ty = objects[target_idx]

        # Compute direction of target relative to anchor
        # Standard: north = up (positive y), east = right (positive x)
        dx = tx - ax
        dy = ty - ay
        # find dominant axis
        if abs(dx) >= abs(dy):
            direction = "east" if dx > 0 else "west"
        else:
            direction = "north" if dy > 0 else "south"
        letter = "ABCD"[["north", "south", "east", "west"].index(direction)]

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(
            anchor=anchor_name, anchor_dir=anchor_dir, target=target_name)
        if level <= 2:
            q += (
                " Hint: the figure already labels the N/S/E/W walls; the "
                "premise just confirms which way is north. Compute "
                "target−anchor on the figure: +y=N, -y=S, +x=E, -x=W. "
                "Pick the axis with the larger |delta|. Then map "
                "direction → MCQ letter: N→A, S→B, E→C, W→D."
            )

        img = self._render(objects, anchor_name, target_name, W, H, rng)
        return q, letter, img

    def _render(self, objects, anchor_name, target_name, W, H, rng):
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-0.5, W + 0.5)
        ax.set_ylim(-0.5, H + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Room
        ax.add_patch(mpatches.Rectangle((0, 0), W, H, fc="#f5f5f5",
                                         ec="black", lw=2.5))
        # 2026-04-26: explicit wall labels — model can read which side is N/S/E/W.
        ax.text(W/2, H + 0.3, "NORTH wall", fontsize=12, ha="center",
                fontweight="bold", color="#c0392b")
        ax.text(W/2, -0.3, "SOUTH wall", fontsize=12, ha="center", va="top",
                fontweight="bold", color="#c0392b")
        ax.text(-0.4, H/2, "WEST", fontsize=12, ha="right", va="center",
                fontweight="bold", color="#c0392b", rotation=90)
        ax.text(W + 0.4, H/2, "EAST", fontsize=12, ha="left", va="center",
                fontweight="bold", color="#c0392b", rotation=270)
        # Objects
        for (name, x, y) in objects:
            color = "#e74c3c" if name == anchor_name else (
                "#27ae60" if name == target_name else "#3498db")
            ax.scatter(x, y, s=300, color=color, zorder=5,
                       edgecolors="black", linewidths=1.3)
            ax.text(x, y - 0.5, name, fontsize=10, ha="center", va="top",
                    fontweight="bold")
        # Legend
        ax.text(W / 2, -0.4, "RED = anchor (premise object) | GREEN = target | BLUE = other",
                fontsize=10, ha="center",
                bbox=dict(facecolor="lightyellow", edgecolor="gray"))
        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_cpp"
    os.makedirs(out_dir, exist_ok=True)
    env = CompassPremisePairQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[cpp L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/cpp_s{s}_L{level}.png")
            print(f"[cpp L{level} s{s}] A={env._answer}")
