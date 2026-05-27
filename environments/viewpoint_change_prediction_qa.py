"""
Viewpoint Change Prediction QA (batch 3, 2026-04-14).

Target: visual-perception Multi-view / MindCubeBench rotation. The model sees a top-down
floor-plan view showing labelled boxes at known (x, y) positions. The
question asks, "if you rotated the view 90° clockwise, what would be on
your LEFT?" — an integer / letter short answer.

Format: constant MCQ letter A/B/C/D (letter of the labelled object).

Difficulty axes:
  A) Pattern A: n_objects (3..6).
  B) Pattern H: angle pool {90} at L0 → {90, 180, 270} at L3 → also
     180-left / 270-right at L6.
  C) Pattern D: objects closer together at higher levels.
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

class ViewpointChangePredictionQA(StandaloneVisualEnv):
    ENV_NAME = "viewpoint_change_prediction"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # Fewer objects = fewer candidates; 180 deg rotation is simplest (flip).
        # Previous L0 used 5 objects + 180 and scored only 0.35 — too many
        # landmarks made left/right judgment confusing at the jittered
        # positions. We lower object count at L0 and grow with level.
        # Iter 3 (2026-04-17): previous schedule produced INVERTED pass-rate
        # 0.35/0.45/0.50/0.75. L9 was actually the easiest because
        # dir_pool={front, back} with 5 candidates is close to a 2-way choice
        # between two longitudinal extremes (first landmark in front, last
        # behind). Reverse qtype ordering: L9 keeps full 4-way direction
        # pool + 90°/270°; L0 keeps the friendly 180° + front/back pair.
        if level <= 2:
            return {
                "n_objects": 3,
                "angles": [180],
                "dir_pool": ["front", "back"],
                "jitter": 0.15,
            }
        if level <= 4:
            return {
                "n_objects": 4,
                "angles": [180],
                "dir_pool": ["left", "right", "front", "back"],
                "jitter": 0.2,
            }
        if level <= 6:
            return {
                "n_objects": 4,
                "angles": [90, 180, 270],
                "dir_pool": ["left", "right", "front", "back"],
                "jitter": 0.25,
            }
        if level <= 8:
            return {
                "n_objects": 5,
                "angles": [90, 270],                       # no easy 180
                "dir_pool": ["left", "right", "front", "back"],
                "jitter": 0.35,
            }
        # L9: hardest — 5 landmarks, only 90/270 rotation, full 4-way pool
        # with max jitter.
        return {
            "n_objects": 5,
            "angles": [90, 270],
            "dir_pool": ["left", "right", "front", "back"],
            "jitter": 0.45,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["n_objects"] * 10 + (1 + len(cfg["angles"]))

        for _ in range(25):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        n = cfg["n_objects"]
        # Random positions on a 5x5 grid, excluding origin (person stands
        # at origin and would otherwise be occluded by a box).
        positions = []
        cells = [(x, y) for x in range(-2, 3) for y in range(-2, 3)
                 if not (x == 0 and y == 0)]
        rng.shuffle(cells)
        positions = cells[:n]
        labels = [chr(ord("A") + i) for i in range(n)]

        angle = rng.choice(cfg["angles"])
        direction = rng.choice(cfg["dir_pool"])

        # After rotating the camera by `angle` clockwise, what's on the
        # specified side of the camera? Camera originally facing +y
        # (camera-frame: left=-x, right=+x, front=+y, back=-y).
        # To express world positions in the rotated camera frame, we apply
        # the inverse of the camera rotation: if camera rotated θ clockwise
        # in world, objects rotate θ counter-clockwise in the camera frame.
        # For 180° this is symmetric (old code worked); for 90°/270° the
        # old code used cos(-θ) which is CW and inverted front/back.
        theta = math.radians(angle)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        transformed = []
        for (px, py), lbl in zip(positions, labels):
            # CCW rotation by theta: (x, y) -> (x*cos - y*sin, x*sin + y*cos)
            nx = cos_t * px - sin_t * py
            ny = sin_t * px + cos_t * py
            transformed.append((nx, ny, lbl))

        # In camera frame now: left = -x, right = +x, front = +y, back = -y
        def _score(item, d):
            nx, ny, _ = item
            if d == "left":
                return -nx - abs(ny) * 0.3
            if d == "right":
                return nx - abs(ny) * 0.3
            if d == "front":
                return ny - abs(nx) * 0.3
            if d == "back":
                return -ny - abs(nx) * 0.3

        best = max(transformed, key=lambda it: _score(it, direction))
        answer = best[2]

        q = (f"A top-down floor plan shows {n} labelled boxes. A person "
             f"stands at the centre initially facing the +y direction "
             f"(upward). If that person rotates {angle}° clockwise, which "
             f"labelled box will be most to their {direction}? Answer with "
             "a single letter.")
        if level <= 2:
            q += (
                f" Hint: after rotating {angle}° clockwise from facing +y, "
                "the person's 'forward' vector points to a new direction "
                "(at 0°: +y; 90° CW: +x; 180°: -y; 270° CW: -x). Compute "
                "the rotation algebraically: forward = (sin(angle), cos(angle)). "
                "The 'right' direction is forward rotated 90° CW; 'left' is "
                "+90° CCW; 'front'=forward; 'back'=-forward. For each box, "
                "compute its dot product with the desired direction; the "
                "box with maximum projection is the answer."
            )

        image = self._render(labels, positions, cfg, rng, angle, direction)
        return q, answer, image

    def _render(self, labels, positions, cfg, rng, angle, direction) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]

        fig, ax = plt.subplots(figsize=(6 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])
        ax.set_aspect("equal")

        # Draw grid
        ax.set_xticks(range(-3, 4))
        ax.set_yticks(range(-3, 4))
        ax.grid(True, alpha=0.3, linestyle="--")

        # Center person (triangle pointing +y)
        tri = mpatches.Polygon(
            [(0, 0.4), (-0.3, -0.2), (0.3, -0.2)],
            facecolor="#e74c3c", edgecolor="#000", linewidth=1.0)
        ax.add_patch(tri)
        ax.text(0, -0.45, "you (facing up)", fontsize=fs - 2,
                ha="center", color="#e74c3c")

        # Place labelled boxes
        for (px, py), lbl in zip(positions, labels):
            color = palette[(ord(lbl) - ord("A")) % len(palette)]
            j_x = px + rng.uniform(-cfg["jitter"] * 0.2, cfg["jitter"] * 0.2)
            j_y = py + rng.uniform(-cfg["jitter"] * 0.2, cfg["jitter"] * 0.2)
            rect = mpatches.Rectangle(
                (j_x - 0.3, j_y - 0.3), 0.6, 0.6,
                facecolor=color, edgecolor="#000", linewidth=1.2)
            ax.add_patch(rect)
            ax.text(j_x, j_y, lbl, fontsize=fs + 2, fontweight="bold",
                    ha="center", va="center", color="#fff")

        ax.set_xlim(-3.5, 3.5)
        ax.set_ylim(-3.5, 3.5)
        ax.set_title(f"Top-down view (north = +y)", fontsize=fs + 1)
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = ViewpointChangePredictionQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[viewpoint_change_prediction L{level} s{s}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"viewpoint_change_prediction_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[viewpoint_change_prediction L{level} s{s}] A={env._answer}")
