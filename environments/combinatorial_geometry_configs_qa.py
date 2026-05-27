"""
Combinatorial Geometry Configs QA (v4 G23, for combinatorial geometry).

Targets: combinatorial geometry -2.60.

Task: combinatorial counting on a geometric configuration. Examples:
  - "How many distinct triangles can be formed using these 5 points?"
  - "How many diagonals does an n-gon have?"
  - "How many rectangles are in this k-by-m grid?"
  - "How many line segments are there between these n points?"

Reward: exact integer.

Level axes:
  A) Problem type: simpler counting at L0-3 (C(n,2) line segments), harder at L4-6 (triangle count), hardest at L7+ (rectangle count on grid)
  B) Number of points: 3-6 at L0, 7-10 at L9
"""
import math
import random
from itertools import combinations
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES_SEG = [
    "{n} points are shown on the plane. How many distinct line segments can be drawn using pairs of these points? Put the integer in <answer>...</answer>.",
    "Count the number of line segments connecting any two of the {n} shown points. Integer in <answer>...</answer>.",
    "Among {n} points, how many distinct segments (each connecting a pair)? Integer in <answer>...</answer>.",
    "{n} points → how many segments from pairs? Integer in <answer>...</answer>.",
    "Given {n} points, total pairs (segments)? Integer in <answer>...</answer>.",
    "Count line segments between pairs of the {n} shown points. Integer in <answer>...</answer>.",
    "Integer: total distinct segments from {n} points. Put in <answer>...</answer>.",
    "From {n} points, how many segments? Integer in <answer>...</answer>.",
    "Number of pairwise line segments between {n} points? Integer in <answer>...</answer>.",
    "{n} points, all-pair segments count? Integer in <answer>...</answer>.",
    "Count segments drawn between each pair of {n} points. Integer in <answer>...</answer>.",
    "Segments between {n} points? Integer in <answer>...</answer>.",
    "Given {n} points shown, count segments. Integer in <answer>...</answer>.",
    "All segments from {n} points? Integer in <answer>...</answer>.",
    "Compute segment count for {n} points. Integer in <answer>...</answer>.",
    "{n} points → segment count? Integer in <answer>...</answer>.",
]

_TEMPLATES_TRI = [
    "{n} points are shown, no three collinear. How many distinct triangles can be formed using three of these points? Put the integer in <answer>...</answer>.",
    "Triangles from {n} points (no three collinear)? Integer in <answer>...</answer>.",
    "Count all distinct triangles from {n} points (general position). Integer in <answer>...</answer>.",
    "From {n} points in general position, how many triangles? Integer in <answer>...</answer>.",
    "Integer: triangles formed by triples of {n} points. Put in <answer>...</answer>.",
    "How many triangles are there using 3 of the {n} points? Integer in <answer>...</answer>.",
    "Distinct triangles from {n} points? Integer in <answer>...</answer>.",
    "{n} points (no 3 collinear). Triangle count? Integer in <answer>...</answer>.",
    "Count distinct triangles over {n} points. Integer in <answer>...</answer>.",
    "Total triangles from {n} points? Integer in <answer>...</answer>.",
    "How many distinct triangles? ({n} points, general position) Integer in <answer>...</answer>.",
    "Count triangles from {n} shown points. Integer in <answer>...</answer>.",
    "Triangles formed by any 3 of {n} points? Integer in <answer>...</answer>.",
    "{n}-point triangle count? Integer in <answer>...</answer>.",
    "Integer count of triangles from {n} points? Put in <answer>...</answer>.",
    "From {n} points, total triangles? Integer in <answer>...</answer>.",
]

_TEMPLATES_RECT = [
    "How many axis-aligned rectangles (of any size) can be formed on the {m}x{n} grid of points? Put the integer in <answer>...</answer>.",
    "Count axis-aligned rectangles in a {m}x{n} grid of points. Integer in <answer>...</answer>.",
    "Axis-aligned rectangle count on {m}x{n} grid? Integer in <answer>...</answer>.",
    "{m}x{n} grid of points → how many axis-aligned rectangles? Integer in <answer>...</answer>.",
    "Number of rectangles in {m}x{n} grid (axis-aligned)? Integer in <answer>...</answer>.",
    "Compute rectangle count ({m}x{n} grid, axis-aligned). Integer in <answer>...</answer>.",
    "Count rectangles on an {m}x{n} grid. Integer in <answer>...</answer>.",
    "{m}×{n} grid rectangles? Integer in <answer>...</answer>.",
    "How many axis-aligned rectangles in {m}×{n} grid? Integer in <answer>...</answer>.",
    "Axis-aligned rectangles ({m}×{n})? Integer in <answer>...</answer>.",
    "Rectangle count on the {m}x{n} grid (axis-aligned)? Integer in <answer>...</answer>.",
    "Count every axis-aligned rectangle ({m}x{n} grid). Integer in <answer>...</answer>.",
    "Integer: rectangles on {m}x{n} grid (axis-aligned). Put in <answer>...</answer>.",
    "Number of axis-aligned rectangles on {m}x{n} grid? Integer in <answer>...</answer>.",
    "{m}x{n} grid — how many rectangles? Integer in <answer>...</answer>.",
    "All axis-aligned rectangles in an {m}x{n} grid? Integer in <answer>...</answer>.",
]

class CombinatorialGeometryConfigsQA(StandaloneVisualEnv):
    ENV_NAME = "combinatorial_geometry_configs"
    TEXTBOOK_POSTPROCESS = True

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 2:
            qtype = "segments"
        elif level <= 5:
            qtype = "triangles"
        else:
            qtype = "rectangles"
        return {"qtype": qtype, "level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 353)
        self._primary_complexity_feature = level

        if cfg["qtype"] == "segments":
            n = rng.randint(4, 8)
            answer = str(n * (n - 1) // 2)
            sidx = (self.seed or 0) % 16
            q = _TEMPLATES_SEG[sidx].format(n=n)
            img = self._render_points(n, show_segments=False, rng=rng)
        elif cfg["qtype"] == "triangles":
            n = rng.randint(5, 9)
            answer = str(n * (n - 1) * (n - 2) // 6)
            sidx = (self.seed or 0) % 16
            q = _TEMPLATES_TRI[sidx].format(n=n)
            img = self._render_points(n, show_segments=False, rng=rng)
        else:  # rectangles
            m = rng.randint(3, 5)
            n = rng.randint(3, 5)
            # Number of axis-aligned rectangles = C(m, 2) * C(n, 2)
            answer = str((m * (m - 1) // 2) * (n * (n - 1) // 2))
            sidx = (self.seed or 0) % 16
            q = _TEMPLATES_RECT[sidx].format(m=m, n=n)
            img = self._render_grid(m, n, rng)

        return q, answer, img

    def _render_points(self, n, show_segments, rng):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-0.5, 6); ax.set_ylim(-0.5, 6)
        ax.set_aspect("equal")
        ax.axis("off")

        # Generate n points in general position on a circle-ish layout
        pts = []
        for i in range(n):
            theta = 2 * math.pi * i / n + rng.uniform(-0.1, 0.1)
            r = 2.3 + rng.uniform(-0.3, 0.3)
            x = 2.75 + r * math.cos(theta)
            y = 2.75 + r * math.sin(theta)
            pts.append((x, y))

        # points
        for i, (x, y) in enumerate(pts):
            ax.scatter(x, y, s=150, color="#e74c3c", zorder=5,
                       edgecolors="black", linewidths=1.5)
            ax.text(x + 0.15, y + 0.15, str(i + 1), fontsize=11,
                    fontweight="bold")

        if show_segments:
            for i, j in combinations(range(n), 2):
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        color="gray", lw=0.5, alpha=0.6)

        return self.fig_to_pil(fig)

    def _render_grid(self, m, n, rng):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-0.5, m + 0.5); ax.set_ylim(-0.5, n + 0.5)
        ax.set_aspect("equal")
        ax.axis("off")

        for i in range(m):
            for j in range(n):
                ax.scatter(i, j, s=200, color="#3498db", zorder=5,
                           edgecolors="black", linewidths=1.5)

        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_cgc"
    os.makedirs(out_dir, exist_ok=True)
    env = CombinatorialGeometryConfigsQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 91
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[cgc L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/cgc_s{s}_L{level}.png")
            print(f"[cgc L{level} s{s}] A={env._answer}")
