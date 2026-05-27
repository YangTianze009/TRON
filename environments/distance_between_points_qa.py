"""
Distance Between Points QA (batch 3, 2026-04-14).

Target: Coordinate . Two labelled points P and Q on a
coordinate plane — answer is the Euclidean distance between them.

Format: numeric short answer (decimal, rounded to 1 decimal place, or
integer when exact).

Difficulty axes:
  A) Pattern H: coordinate magnitude range (2..3 at L0 → 7..9 at L9)
  B) Pattern H: at L <= 2, require integer distances (Pythagorean triples);
     above, allow irrational sqrt → 1-decimal rounding.
  C) Pattern G: label hiding — point coordinates printed next to the dot
     only at L<=3; above that only "P" and "Q" remain.
  D) Pattern D: the axis_pad grows so the points look smaller.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_PYTHAG_TRIPLES = [
    (3, 4, 5), (6, 8, 10), (5, 12, 13), (8, 15, 17),
    (9, 12, 15), (7, 24, 25), (20, 21, 29), (12, 16, 20),
    (4, 3, 5), (12, 5, 13),
]

class DistanceBetweenPointsQA(StandaloneVisualEnv):
    ENV_NAME = "distance_between_points"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        return {
            "coord_range": 3 + level,               # 3..12
            "integer_distance": level <= 2,
            "show_coord_labels": level <= 3,
            "axis_pad": 1 + level // 3,
            "show_grid": level <= 6,
            "show_connector": level <= 5,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        self._primary_complexity_feature = cfg["coord_range"] * 2 + (0 if cfg["integer_distance"] else 3)

        for _ in range(20):
            r = self._try_generate(rng, cfg, level)
            if r is not None:
                return r
        return None

    def _try_generate(self, rng, cfg, level):
        lim = cfg["coord_range"]

        if cfg["integer_distance"]:
            # Pick Pythagorean / horizontal / vertical — need variety so no
            # single answer dominates at L0. Include axis-aligned deltas.
            mode = rng.choice(["horizontal", "vertical", "triple"])
            if mode == "horizontal":
                dy = 0
                dx = rng.randint(1, lim) * rng.choice([-1, 1])
                c = abs(dx)
            elif mode == "vertical":
                dx = 0
                dy = rng.randint(1, lim) * rng.choice([-1, 1])
                c = abs(dy)
            else:
                a, b, c = rng.choice(_PYTHAG_TRIPLES)
                if a > lim or b > lim:
                    small = [(3, 4, 5), (4, 3, 5)]
                    a, b, c = rng.choice(small)
                dx = a * rng.choice([-1, 1])
                dy = b * rng.choice([-1, 1])
            # Ensure both points in [-lim, lim]
            try:
                x1 = rng.randint(-lim + max(0, -dx), lim - max(0, dx))
                y1 = rng.randint(-lim + max(0, -dy), lim - max(0, dy))
            except ValueError:
                return None
            x2, y2 = x1 + dx, y1 + dy
            if not (-lim <= x2 <= lim and -lim <= y2 <= lim):
                return None
            dist = float(c)
            answer = str(int(c))
        else:
            for _ in range(10):
                x1 = rng.randint(-lim, lim)
                y1 = rng.randint(-lim, lim)
                x2 = rng.randint(-lim, lim)
                y2 = rng.randint(-lim, lim)
                if (x1, y1) == (x2, y2):
                    continue
                dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                break
            else:
                return None
            # Round to 1 decimal
            answer = f"{dist:.1f}"

        _P = [
            "Two labelled points P and Q are shown on the coordinate plane. What is the distance between P and Q? Give your answer as a number (round to 1 decimal place if not an integer).",
            "Points P and Q appear on the coordinate grid. Compute the distance |PQ|. Round to 1 decimal place if non-integer.",
            "From the plot, determine the Euclidean distance between P and Q. Round to 1 decimal if non-integer.",
            "The figure shows labelled points P and Q. Find |PQ|. Give a number; round to 1 decimal if needed.",
            "Compute the distance between P and Q using the coordinate diagram. Round to 1 decimal if non-integer.",
            "Looking at the plotted P and Q, what is the distance between them? Round to 1 decimal if not an integer.",
            "Based on the labelled P and Q in the figure, find |PQ|. Number answer; 1 decimal if not integer.",
            "From the coordinate figure, calculate |PQ|. Integer or 1-decimal number.",
            "Find the distance between P and Q plotted on the coordinate plane. 1 decimal place (if non-integer).",
            "Points P, Q are shown on the grid. Compute their distance. Round to 1 decimal if non-integer.",
            "Determine the length of PQ from the coordinate plot. Integer or 1-decimal.",
            "Given the plotted P and Q, what is the Euclidean distance between them? 1 decimal place if non-integer.",
            "Using the coordinate plane, find |PQ|. Integer if exact, else round to 1 decimal.",
            "Compute the distance between the labelled points P and Q in the figure. Round to 1 decimal if needed.",
            "Based on P and Q shown on the plot, what is |PQ|? Number with 1 decimal if non-integer.",
            "Find |PQ| from the figure showing two labelled points. Round to 1 decimal if non-integer.",
        ]
        sidx = (self.seed or 0) % 16
        q = _P[sidx]

        image = self._render((x1, y1), (x2, y2), cfg)
        return q, answer, image

    def _render(self, P, Q, cfg) -> Image.Image:
        style = self._random_style()
        sc = style["figsize_scale"]
        fs = style["font_size_base"]
        palette = style["palette"]

        pad = cfg["axis_pad"]
        rng_lim = cfg["coord_range"] + pad
        fig, ax = plt.subplots(figsize=(6 * sc, 6 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        if cfg["show_grid"]:
            ax.grid(True, which="major", alpha=0.3, linestyle="--")

        ax.axhline(0, color="#333", linewidth=1.2, zorder=1)
        ax.axvline(0, color="#333", linewidth=1.2, zorder=1)

        # Plot connector line segment (L <= 5)
        if cfg["show_connector"]:
            ax.plot([P[0], Q[0]], [P[1], Q[1]],
                    color=palette[0 % len(palette)], linewidth=1.6,
                    linestyle="--", zorder=3)

        # Plot points
        for label, pt in (("P", P), ("Q", Q)):
            ax.plot([pt[0]], [pt[1]], "o", color=palette[1 % len(palette)],
                    markersize=9, zorder=4,
                    markeredgecolor="#000", markeredgewidth=0.9)
            if cfg["show_coord_labels"]:
                ax.annotate(f"{label} ({pt[0]}, {pt[1]})", pt,
                            xytext=(8, 6), textcoords="offset points",
                            fontsize=fs, fontweight="bold")
            else:
                ax.annotate(f"{label}", pt,
                            xytext=(8, 6), textcoords="offset points",
                            fontsize=fs + 2, fontweight="bold")

        ax.set_xlim(-rng_lim, rng_lim)
        ax.set_ylim(-rng_lim, rng_lim)
        ax.set_aspect("equal")

        ax.set_xticks(range(-rng_lim, rng_lim + 1))
        ax.set_yticks(range(-rng_lim, rng_lim + 1))
        ax.tick_params(labelsize=fs - 2)
        ax.set_xlabel("x", fontsize=fs + 1)
        ax.set_ylabel("y", fontsize=fs + 1)
        ax.set_title("Find distance between P and Q", fontsize=fs + 1)
        self._apply_style(fig, ax, style)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_b3"
    os.makedirs(out_dir, exist_ok=True)
    env = DistanceBetweenPointsQA()
    for level in (0, 3, 6):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 17
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[distance_between_points L{level} s{s}] FAILED")
                continue
            path = os.path.join(
                out_dir, f"distance_between_points_s{s}_L{level}.png")
            env.render().save(path)
            print(f"[distance_between_points L{level} s{s}] A={env._answer}")
