"""
Reflection Path Length QA (v4 G2b / G6b, for length / trans-geo).

Targets:

    billiard ball reflection)

Failure mode from pulled cases: v3 doesn't apply the mirror-image trick,
so it gives up or guesses.

Task: billiard / light-ray problem where a particle starts at (x0, y0),
travels at 45° (or given angle), bounces off k horizontal / vertical
walls, and ends at a target or pocket. Ask for total path length.

Reward: numeric within 2% relative tolerance, accepts closed forms like
'2*sqrt(13)'.

Level axes:
  A) Number of bounces: 1 at L0-2, 2 at L3-5, 3 at L6+
  B) Angle restricted to 45° at L0-3, general at L4+
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "A billiard ball starts at {start} and travels at 45° (up and to the right). It reflects off the walls of the rectangular {rect_desc} table. Compute the total path length until it reaches {target_desc}. Use the mirror-image trick. Round to 2 decimal places; put in <answer>...</answer>.",
    "A light ray starts at {start}, travels at 45° toward the upper right, and bounces off the walls of a {rect_desc} rectangle. Find total distance traveled until it reaches {target_desc}. Round to 2 decimal places; put in <answer>...</answer>.",
    "In a {rect_desc} box, a particle starts at {start}, moves at 45°, and reflects off the walls. Compute total path length until it reaches {target_desc}. Put in <answer>...</answer>.",
    "Ball on a {rect_desc} table at {start}, 45° angle. Find path length to {target_desc}. Round to 2dp. Put in <answer>...</answer>.",
    "Starting point {start} on a {rect_desc} table, direction 45°. Path length until {target_desc}? Round to 2dp. Put in <answer>...</answer>.",
    "A 45° trajectory from {start} on a {rect_desc} table. Total path to {target_desc}? Round to 2dp. Put in <answer>...</answer>.",
    "From {start} on a {rect_desc} table, moving 45°, compute path length to {target_desc}. Put in <answer>...</answer>.",
    "Particle at {start}, 45° direction, {rect_desc} table. Path to {target_desc}? Put in <answer>...</answer>.",
    "Compute the reflected path length: start {start}, 45°, table is {rect_desc}, target {target_desc}. Round 2dp. Put in <answer>...</answer>.",
    "{rect_desc} table, start {start}, 45°. Path length to {target_desc}? 2dp in <answer>...</answer>.",
    "Light ray at {start} on {rect_desc} table, 45° up-right. Total distance to {target_desc}? 2dp in <answer>...</answer>.",
    "Reflected path from {start} at 45° in {rect_desc} table. Total length to {target_desc}. Put in <answer>...</answer>.",
    "A {rect_desc} rectangle has a particle at {start} moving 45°. Path to {target_desc}? 2dp in <answer>...</answer>.",
    "Ball at {start} in a {rect_desc} table, 45°. Compute the full path length until it reaches {target_desc}. Put in <answer>...</answer>.",
    "Compute path length: {rect_desc} table, start {start}, angle 45°, target {target_desc}. 2dp in <answer>...</answer>.",
    "Start {start}, 45° in {rect_desc} rectangle. Path length to {target_desc}? Put in <answer>...</answer>.",
]

class ReflectionPathLengthQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "reflection_path_length"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            n_bounces = 1
        elif level <= 5:
            n_bounces = 2
        else:
            n_bounces = 3
        return {"n_bounces": n_bounces, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 167)
        self._primary_complexity_feature = level

        # Rectangle dims
        W = rng.randint(4, 10)
        H = rng.randint(3, 8)
        # Start at a random bottom-edge point (x0, 0)
        x0 = rng.randint(0, W - 1)
        start = (x0, 0)

        # At 45°, the trajectory unfolds into a straight line of slope 1 in
        # the mirror-image plane. Target can be a specific pocket.
        # Let's compute where the ball ends up after n_bounces.

        n_bounces = cfg["n_bounces"]
        # simple: ball travels at 45° up-right, bouncing off walls
        # simulate until n_bounces + terminate at a side wall or top
        pos = list(start)
        direction = [1, 1]  # (dx, dy) each ±1 at 45°
        bounces = 0
        total_length = 0.0
        path_pts = [tuple(pos)]

        max_steps = 50
        while bounces <= n_bounces and max_steps > 0:
            max_steps -= 1
            # time to hit vertical walls
            tx = (W - pos[0]) / direction[0] if direction[0] > 0 else (0 - pos[0]) / direction[0]
            ty = (H - pos[1]) / direction[1] if direction[1] > 0 else (0 - pos[1]) / direction[1]
            t = min(tx, ty)
            if t <= 0:
                break
            # advance
            pos[0] += direction[0] * t
            pos[1] += direction[1] * t
            total_length += t * math.sqrt(2)
            path_pts.append(tuple(pos))
            # reflect
            if abs(pos[0] - 0) < 1e-6 or abs(pos[0] - W) < 1e-6:
                direction[0] = -direction[0]
                bounces += 1
            if abs(pos[1] - 0) < 1e-6 or abs(pos[1] - H) < 1e-6:
                direction[1] = -direction[1]
                bounces += 1
            if bounces > n_bounces:
                break

        if bounces < n_bounces or total_length < 1:
            return None

        answer = f"{round(total_length, 2)}"
        target_desc = f"the point {path_pts[-1]} after {n_bounces} bounce(s)"
        rect_desc = f"{W}x{H}"

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(start=start, rect_desc=rect_desc,
                                     target_desc=target_desc)

        img = self._render(W, H, path_pts, rng)
        return q, answer, img

    def _render(self, W, H, path_pts, rng):
        fig, ax = plt.subplots(figsize=(7, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-0.5, W + 0.5)
        ax.set_ylim(-0.5, H + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        # Table boundary
        ax.add_patch(mpatches.Rectangle((0, 0), W, H, fc="#eafaf1",
                                         ec="black", lw=2.5))
        # Path
        for i in range(len(path_pts) - 1):
            x0, y0 = path_pts[i]
            x1, y1 = path_pts[i + 1]
            ax.plot([x0, x1], [y0, y1], color="red", lw=2.0)
        # Start + end markers
        ax.plot(*path_pts[0], "o", color="green", markersize=12)
        ax.text(path_pts[0][0] + 0.15, path_pts[0][1] - 0.4, f"START {path_pts[0]}",
                fontsize=11, fontweight="bold", color="darkgreen")
        ax.plot(*path_pts[-1], "s", color="darkred", markersize=12)
        ax.text(path_pts[-1][0] + 0.15, path_pts[-1][1] + 0.25,
                f"END ≈ ({path_pts[-1][0]:.1f}, {path_pts[-1][1]:.1f})",
                fontsize=11, fontweight="bold", color="darkred")

        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().rstrip(".").rstrip(",").rstrip()
        gt = ground_truth.strip().lower().rstrip(".").rstrip(",").rstrip()
        try:
            p = float(pred)
            g = float(gt)
            return abs(p - g) / max(abs(g), 1e-9) < 0.02 or abs(p - g) < 0.1
        except ValueError:
            pass
        # Symbolic eval via base _try_eval_expr — handles 3*sqrt(2), 3√2,
        # \sqrt{18}, etc. (the upgraded base verifier accepts implicit
        # multiplication and LaTeX).
        try:
            p_val = self._try_eval_expr(pred)
            g_val = self._try_eval_expr(gt)
            if p_val is not None and g_val is not None:
                return (abs(p_val - g_val) / max(abs(g_val), 1e-9) < 0.02
                        or abs(p_val - g_val) < 0.1)
        except Exception:
            pass
        return super()._check_answer(predicted, ground_truth)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_rpl"
    os.makedirs(out_dir, exist_ok=True)
    env = ReflectionPathLengthQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 89
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[rpl L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/rpl_s{s}_L{level}.png")
            print(f"[rpl L{level} s{s}] A={env._answer}")
