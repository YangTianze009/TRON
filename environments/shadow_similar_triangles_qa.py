"""
Two vertical poles in sunlight cast shadows on flat ground. The first pole's
height h1 and shadow length s1 are known; the second pole's shadow length
s2 is given. Because both shadows are cast by the same sun rays, the two
right triangles (pole, shadow, hypotenuse light ray) are similar, so

    h2 / s2 = h1 / s1   ⇒   h2 = h1 * s2 / s1

The image shows the two poles, both shadows, and the (parallel) sun rays
forming similar right triangles. The dimensions h1, s1, s2 are labeled.
The model is asked for h2.

L0: pole 4 m, shadow 2 m, second shadow 6 m → h2 = 12 m.
"""
import math
import random
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Two vertical poles stand on flat ground in sunlight. The first pole has height {h1} m and casts a shadow of length {s1} m. The second pole, at the same time of day, casts a shadow of length {s2} m. Find the height of the second pole (in meters). Place the numeric value in <answer>...</answer>.",
    "At the same moment two vertical poles cast shadows on level ground. Pole 1 is {h1} m tall with a shadow of {s1} m. Pole 2's shadow is {s2} m long. Compute the height of pole 2. Numeric answer in <answer>...</answer>.",
    "The image shows two upright poles on horizontal ground. Pole 1 has height {h1} m and shadow {s1} m. Pole 2 has shadow {s2} m. Using similar triangles, find the height of pole 2. Place the value in <answer>...</answer>.",
    "Two poles stand vertically and cast shadows in the same sunlight. The first is {h1} m tall and casts {s1} m of shadow; the second casts {s2} m of shadow. Find the height of the second pole. Place the answer in <answer>...</answer>.",
    "Sun rays make the same angle with the ground for both poles in the figure. Pole A is {h1} m tall with shadow {s1} m; pole B has shadow {s2} m. Compute pole B's height. Numeric value in <answer>...</answer>.",
    "Use similar triangles to find the unknown pole height. Known: pole 1 = {h1} m, shadow 1 = {s1} m, shadow 2 = {s2} m. Find pole 2's height. Place the answer in <answer>...</answer>.",
    "The image shows two poles and their shadows on level ground. The first pole, height {h1} m, casts a shadow {s1} m long. The second pole's shadow is {s2} m long. Determine the height of the second pole. Place the value in <answer>...</answer>.",
    "Two vertical poles in sunlight: pole 1 of height {h1} m casts shadow {s1} m; pole 2 casts shadow {s2} m. Determine the height of pole 2. Place the numeric answer in <answer>...</answer>.",
    "From the diagram: pole A has height {h1} m and shadow {s1} m; pole B has shadow {s2} m at the same moment. Find pole B's height. Place the value in <answer>...</answer>.",
    "Two upright posts cast shadows under the same sunlight. The first post is {h1} m tall and casts a {s1}-m shadow; the second post casts a {s2}-m shadow. Compute the second post's height. Place the answer in <answer>...</answer>.",
    "On the same flat ground at the same time of day, pole 1 of height {h1} m has shadow {s1} m, and pole 2 has shadow {s2} m. By similar triangles, find pole 2's height. Numeric value in <answer>...</answer>.",
    "Apply similar triangles to two poles in sunlight. Pole 1: height {h1} m, shadow {s1} m. Pole 2: shadow {s2} m. Compute pole 2's height. Place the value in <answer>...</answer>.",
    "Two vertical poles cast shadows on level ground. Given pole 1 = {h1} m, shadow 1 = {s1} m, and shadow 2 = {s2} m, find the height of pole 2. Place the answer in <answer>...</answer>.",
    "The figure shows two poles standing perpendicular to the ground with their shadows. Pole 1 has height {h1} m and shadow {s1} m; pole 2 has shadow {s2} m. Compute pole 2's height. Place the value in <answer>...</answer>.",
    "Same sun, two poles, two shadows: pole A {h1} m / shadow {s1} m, pole B shadow {s2} m. What is pole B's height? Place the numeric value in <answer>...</answer>.",
    "Refer to the diagram of two vertical poles and their parallel light-ray shadows. Pole 1 is {h1} m tall and casts {s1} m of shadow. Pole 2 casts {s2} m of shadow. Find pole 2's height. Place the answer in <answer>...</answer>.",
    "Two right triangles (pole + shadow + light ray) are similar in the figure. Pole 1 = {h1} m, shadow 1 = {s1} m, shadow 2 = {s2} m. Compute the second pole's height. Place the value in <answer>...</answer>.",
]


class ShadowSimilarTrianglesQA(StandaloneVisualEnv):
    ENV_NAME = "shadow_similar_triangles"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {"level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        rng = random.Random((self.seed or 0) * 1217 + level * 53 + 7)

        # Choose ratio r = h1 / s1 (sun angle), then s2 -> h2 = r * s2.
        # Keep numbers integer at low levels by selecting r as a small fraction.
        if level == 0:
            # Canonical L0 from task: 4 m pole, 2 m shadow, 6 m shadow → h2 = 12.
            h1, s1, s2 = 4, 2, 6
        elif level <= 2:
            # integer ratio (r = 1, 2, or 3)
            r = rng.choice([1, 2, 3])
            s1 = rng.choice([2, 3, 4, 5])
            h1 = r * s1
            s2_mult = rng.choice([2, 3, 4, 5, 6])
            s2 = s2_mult  # h2 = r * s2 always integer
        elif level <= 4:
            # ratio could be 1/2, 3/2, etc. Keep h2 integer-ish.
            r_num, r_den = rng.choice([(1, 1), (2, 1), (3, 1), (3, 2), (1, 2), (5, 2), (4, 1)])
            # pick s1 a multiple of r_den, s2 a multiple of r_den
            s1 = r_den * rng.choice([2, 3, 4])
            h1 = r_num * (s1 // r_den)
            s2 = r_den * rng.choice([2, 3, 4, 5, 6])
        elif level <= 6:
            # decimals appear, possibly non-integer h2
            r_num, r_den = rng.choice([(3, 4), (5, 4), (7, 4), (5, 3), (4, 3), (7, 2)])
            s1 = r_den * rng.choice([2, 3, 4])
            h1 = r_num * (s1 // r_den)
            s2 = rng.choice([3, 4, 5, 6, 7, 8, 9, 10])
        else:
            # higher level: realistic measurements with decimals
            h1 = rng.choice([3.5, 4.2, 5.6, 6.4, 7.5, 8.0, 9.6, 10.5, 12.0])
            s1 = rng.choice([1.5, 2.0, 2.4, 2.8, 3.2, 3.5, 4.0, 5.0])
            s2 = rng.choice([2.0, 3.0, 4.5, 5.5, 6.4, 7.2, 8.5, 10.0, 12.0])

        if abs(s1) < 1e-9:
            return None
        h2 = h1 * s2 / s1

        # Format
        if abs(h2 - round(h2)) < 1e-6:
            ans_str = f"{int(round(h2))}"
        else:
            ans_str = f"{h2:.4g}"

        def _fmt(v):
            if abs(v - round(v)) < 1e-9:
                return f"{int(round(v))}"
            return f"{v:g}"

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(
            h1=_fmt(h1), s1=_fmt(s1), s2=_fmt(s2))

        img = self._render(h1, s1, s2)
        return question, ans_str, img

    def _render(self, h1, s1, s2) -> Image.Image:
        fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Layout: pole 1 starts at x = 0, base goes to x = s1.
        # Pole 2 starts at some x > s1 + small gap; shadow extends to x + s2.
        # Both shadows are to the right (sun on the left).
        gap = max(s1, s2) * 0.3 + 1.0
        p1_x = 0.0
        p1_top_y = h1
        p1_shadow_end = p1_x + s1

        p2_x = p1_shadow_end + gap
        # Use the same ratio so the second pole's height is "drawn correctly" too
        h2 = h1 * s2 / s1
        p2_top_y = h2
        p2_shadow_end = p2_x + s2

        # Ground line
        x_min = -0.6
        x_max = p2_shadow_end + 1.0
        y_max = max(h1, h2) + 1.5
        ax.plot([x_min, x_max], [0, 0], color="#7d6e58", linewidth=2)

        # Pole 1 (vertical line)
        ax.plot([p1_x, p1_x], [0, p1_top_y], color="#2c3e50", linewidth=4)
        # Pole 2
        ax.plot([p2_x, p2_x], [0, p2_top_y], color="#2c3e50", linewidth=4)

        # Shadow 1 (thicker line on ground, slightly below)
        ax.plot([p1_x, p1_shadow_end], [0, 0], color="#5d4e3a", linewidth=6, alpha=0.55)
        # Shadow 2
        ax.plot([p2_x, p2_shadow_end], [0, 0], color="#5d4e3a", linewidth=6, alpha=0.55)

        # Sun ray (hypotenuse) for pole 1: from top of pole down to shadow tip
        ax.plot([p1_x, p1_shadow_end], [p1_top_y, 0],
                color="#e67e22", linewidth=1.6, linestyle="--")
        # Same for pole 2
        ax.plot([p2_x, p2_shadow_end], [p2_top_y, 0],
                color="#e67e22", linewidth=1.6, linestyle="--")

        # Sun (a small disc)
        sun_x = x_min - 0.4
        sun_y = max(p1_top_y, p2_top_y) + 0.6
        from matplotlib.patches import Circle
        ax.add_patch(Circle((sun_x, sun_y), 0.35, color="#f1c40f"))

        # Labels
        def _fmt(v):
            if abs(v - round(v)) < 1e-9:
                return f"{int(round(v))}"
            return f"{v:g}"
        # h1 label (left of pole 1)
        ax.annotate("", xy=(p1_x - 0.25, p1_top_y), xytext=(p1_x - 0.25, 0),
                    arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4))
        ax.text(p1_x - 0.45, p1_top_y / 2, f"h₁ = {_fmt(h1)} m",
                ha="right", va="center", fontsize=11, color="#c0392b",
                fontweight="bold")
        # s1 label (below shadow 1)
        ax.annotate("", xy=(p1_shadow_end, -0.4), xytext=(p1_x, -0.4),
                    arrowprops=dict(arrowstyle="<->", color="#1f4e79", lw=1.4))
        ax.text((p1_x + p1_shadow_end) / 2, -0.7, f"s₁ = {_fmt(s1)} m",
                ha="center", va="top", fontsize=11, color="#1f4e79",
                fontweight="bold")
        # h2 label - mark with "?" since it's the unknown
        ax.annotate("", xy=(p2_x - 0.25, p2_top_y), xytext=(p2_x - 0.25, 0),
                    arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.4))
        ax.text(p2_x - 0.45, p2_top_y / 2, "h₂ = ?",
                ha="right", va="center", fontsize=12, color="#c0392b",
                fontweight="bold")
        # s2 label
        ax.annotate("", xy=(p2_shadow_end, -0.4), xytext=(p2_x, -0.4),
                    arrowprops=dict(arrowstyle="<->", color="#1f4e79", lw=1.4))
        ax.text((p2_x + p2_shadow_end) / 2, -0.7, f"s₂ = {_fmt(s2)} m",
                ha="center", va="top", fontsize=11, color="#1f4e79",
                fontweight="bold")

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(-1.2, y_max)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Two poles, parallel sun rays — find h₂",
                     fontsize=12)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
