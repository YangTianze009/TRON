"""
Rotation Tracking Discrete QA (v4 G4b/G2c, for trans-geo).

Targets: transformation geometry -11.31 (esp. ferris-wheel /
N-turn rotation cases).

Task: clock face / wheel / regular N-gon with a marked pointer at some
start position. Rotate by k steps (each step = 360°/N). Ask for the final
position of the pointer (as a clock-face number or N-gon vertex).

Reward: exact integer or short string.

Level axes:
  A) N (gon size): 4 at L0, 6 at L2, 8 at L4, 12 at L6 (clock), 16 at L9
  B) k (rotations) up to 3 × N at higher levels
  C) Direction: CW only at L0-3, mix CW/CCW at L4+
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

_TEMPLATES_GON = [
    "The figure shows an N-gon with N={N} vertices labeled 1 to {N} (clockwise from the top). A marker starts at vertex {start}. The figure rotates {direction} by {k} positions. Which vertex does the marker land on? Put the integer in <answer>...</answer>.",
    "A regular {N}-gon has the marker at vertex {start}. Rotate {direction} by {k} steps. Where does the marker go? Integer in <answer>...</answer>.",
    "{N}-gon, marker at {start}. Rotation: {k} steps {direction}. Final vertex? Integer in <answer>...</answer>.",
    "Vertex {start} on a {N}-gon. Rotate {direction} by {k}. Resulting vertex? Put integer in <answer>...</answer>.",
    "Regular polygon with {N} vertices; marker at {start}. After {k} steps {direction}, marker is at which vertex? Integer in <answer>...</answer>.",
    "On a {N}-gon with vertices 1..{N}, the pointer starts at {start} and rotates {k} positions {direction}. Final vertex? Put integer in <answer>...</answer>.",
    "The pointer on the {N}-gon at vertex {start} rotates {direction} by {k} steps. Final vertex number? Integer in <answer>...</answer>.",
    "{N}-gon. Start: vertex {start}. Rotate by {k} steps {direction}. End at vertex? Integer in <answer>...</answer>.",
    "Rotate the marker on a {N}-gon from vertex {start} by {k} steps {direction}. Ending vertex? Integer in <answer>...</answer>.",
    "Starting vertex {start} on {N}-gon, rotate {k} {direction}. End vertex? Integer in <answer>...</answer>.",
    "Given {N}-gon with marker at {start}, rotate {k} steps {direction}. End position? Integer in <answer>...</answer>.",
    "{N}-gon marker rotation: from {start}, {k} steps {direction}. Where? Integer in <answer>...</answer>.",
    "Compute the marker's final vertex on a {N}-gon: start {start}, rotation {k} {direction}. Integer in <answer>...</answer>.",
    "{N}-gon: marker at {start} rotates {direction} by {k}. End at? Integer in <answer>...</answer>.",
    "Track rotation of marker on {N}-gon: {start} → rotate {k} {direction}. End? Integer in <answer>...</answer>.",
    "Marker on {N}-gon: starts at vertex {start}, rotates {direction} {k} steps. Final vertex? Integer in <answer>...</answer>.",
]

class RotationTrackingDiscreteQA(StandaloneVisualEnv):
    ALLOW_ROTATION = False  # orientation-sensitive: disable rotation augmentation
    ENV_NAME = "rotation_tracking_discrete"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        N_opts = [4, 4, 6, 6, 8, 8, 12, 12, 16, 16]
        N = N_opts[level]
        max_k = N * (2 + level // 3)  # can exceed full revolution
        allow_ccw = level >= 4
        return {"N": N, "max_k": max_k, "allow_ccw": allow_ccw}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 659)
        self._primary_complexity_feature = level

        N = cfg["N"]
        start = rng.randint(1, N)
        k = rng.randint(1, cfg["max_k"])
        direction = rng.choice(["clockwise", "counterclockwise"]) if cfg["allow_ccw"] else "clockwise"

        # Compute final
        if direction == "clockwise":
            final = ((start - 1 + k) % N) + 1
        else:
            final = ((start - 1 - k) % N) + 1

        answer = str(final)
        sidx = (self.seed or 0) % 16
        q = _TEMPLATES_GON[sidx].format(N=N, start=start, direction=direction, k=k)

        img = self._render(N, start, direction, k, rng)
        return q, answer, img

    def _render(self, N, start, direction, k, rng):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_aspect("equal")
        ax.axis("off")

        # Vertex positions: vertex 1 at top, numbered clockwise
        positions = {}
        for i in range(1, N + 1):
            theta = math.pi / 2 - 2 * math.pi * (i - 1) / N
            positions[i] = (math.cos(theta), math.sin(theta))

        # Draw the N-gon
        pts = [positions[i] for i in range(1, N + 1)] + [positions[1]]
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color="black", lw=1.5)

        # Draw vertices with labels
        for i in range(1, N + 1):
            x, y = positions[i]
            is_start = (i == start)
            color = "#e74c3c" if is_start else "#3498db"
            ax.scatter(x, y, s=260, color=color, zorder=5,
                       edgecolors="black", linewidths=1.3)
            ax.text(x, y, str(i), fontsize=11, ha="center", va="center",
                    color="white", fontweight="bold", zorder=6)

        # Label the start vertex
        sx, sy = positions[start]
        ax.annotate(f"MARKER", xy=(sx, sy), xytext=(sx * 1.3, sy * 1.3),
                    fontsize=10, fontweight="bold", color="red",
                    arrowprops=dict(arrowstyle="->", color="red"))

        ax.set_title(f"Regular {N}-gon, pointer at vertex {start}; "
                     f"rotate {k} steps {direction}", fontsize=11)
        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_rtd"
    os.makedirs(out_dir, exist_ok=True)
    env = RotationTrackingDiscreteQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 421
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[rtd L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/rtd_s{s}_L{level}.png")
            print(f"[rtd L{level} s{s}] A={env._answer}")
