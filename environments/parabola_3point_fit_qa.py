"""
Parabola through 3 lattice points: figure shows three labeled points on a
quadratic y=ax²+bx+c. Ask for one of {a, b, c, vertex coords, axis x = -b/2a}.

Difficulty axes:
  - magnitude of coefs
  - target choice (a/b/c easier than axis or vertex y)
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TARGET_TEMPLATES = {
    "a": [
        "A quadratic curve y = ax² + bx + c passes through the three points labeled in the image. Find the coefficient a.",
        "Three points on a parabola y = ax² + bx + c are shown. Find a.",
        "Compute the leading coefficient a of the parabola y = ax² + bx + c that passes through the three labeled points.",
        "Determine a in y = ax² + bx + c, given the three points marked. Put the",
    ],
    "b": [
        "A parabola y = ax² + bx + c passes through the three labeled points. Find the coefficient b.",
        "Find b in y = ax² + bx + c given the three points shown.",
        "Determine the linear coefficient b of the parabola passing through the three labeled points.",
        "From the three marked points on parabola y = ax² + bx + c, compute b.",
    ],
    "c": [
        "The parabola y = ax² + bx + c passes through the three points in the image. Find c.",
        "Compute the constant term c of the parabola through the three marked points.",
        "Find c in y = ax² + bx + c given the three labeled points.",
        "Determine c such that y = ax² + bx + c passes through the three shown points.",
    ],
    "axis": [
        "A parabola y = ax² + bx + c passes through the three labeled points. Find the equation of its axis of symmetry as `x = ?`.",
        "As shown in the figure, compute the axis of symmetry x = -b/(2a) for the parabola through the three points. Format `x = ?`.",
        "Find x = -b/(2a) for the parabola through the three labeled points. Format the answer as `x = ?`.",
        "Determine the axis of symmetry of the parabola passing through the three marked points. Format `x = ?`.",
    ],
    "vy": [
        "A parabola y = ax² + bx + c passes through the three labeled points. Find the y-coordinate of its vertex.",
        "Compute the vertex's y-coordinate for the parabola through the three points shown.",
        "Find the minimum (or maximum) y-value of the parabola passing through the three labeled points.",
        "As shown in the figure, determine the vertex y-coordinate of the parabola fit to the three given points.",
    ],
}


def _solve_quadratic_3pt(p1, p2, p3):
    """Solve for (a, b, c) of y = ax^2 + bx + c through three points."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    A = np.array([[x1 * x1, x1, 1.0],
                  [x2 * x2, x2, 1.0],
                  [x3 * x3, x3, 1.0]])
    Y = np.array([y1, y2, y3], dtype=float)
    try:
        sol = np.linalg.solve(A, Y)
        return float(sol[0]), float(sol[1]), float(sol[2])
    except np.linalg.LinAlgError:
        return None


class Parabola3PointFitQA(StandaloneVisualEnv):
    ENV_NAME = "parabola_3point_fit"
    # Tighten numeric tolerance to 1% relative tolerance
    # (env default is 5% rel + 0.5 abs floor = ~5x looser). Tighten to match.
    BENCHMARK_NUM_TOLERANCE_REL = 0.01

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        max_a = 1 + level // 3  # 1 → 4
        max_root_abs = 3 + level // 2  # 3 → 7
        if level >= 6:
            targets = ["a", "b", "c", "axis", "vy"]
        elif level >= 3:
            targets = ["a", "b", "c", "axis"]
        else:
            targets = ["a", "c"]
        return {
            "level": level, "max_a": max_a,
            "max_root_abs": max_root_abs, "targets": targets,
        }

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1019 + level * 71 + 23)

        # Choose integer roots r1, r2 (or non-integer to vary), then a
        for _ in range(50):
            r1 = rng.randint(-cfg["max_root_abs"], cfg["max_root_abs"])
            r2 = rng.randint(-cfg["max_root_abs"], cfg["max_root_abs"])
            if r1 == r2:
                continue
            a = rng.choice([-cfg["max_a"], -1, -1, 1, 1, cfg["max_a"]])
            if a == 0:
                continue
            # y = a (x - r1)(x - r2) = ax^2 - a(r1+r2)x + a*r1*r2
            b_coef = -a * (r1 + r2)
            c_coef = a * r1 * r2
            # Pick three distinct lattice points on this parabola
            x_pool = list(range(-cfg["max_root_abs"] - 2, cfg["max_root_abs"] + 3))
            x_pool = [x for x in x_pool if x != r1 and x != r2 and -10 <= a * x * x + b_coef * x + c_coef <= 100]
            if len(x_pool) < 3:
                # use the two roots + 1 lattice
                if not x_pool:
                    continue
                xs = [r1, r2, rng.choice(x_pool)]
            else:
                xs = rng.sample(x_pool, 3)
            pts = [(x, a * x * x + b_coef * x + c_coef) for x in xs]
            break
        else:
            return None

        target = rng.choice(cfg["targets"])
        if target == "a":
            ans = a
        elif target == "b":
            ans = b_coef
        elif target == "c":
            ans = c_coef
        elif target == "axis":
            ans_val = -b_coef / (2 * a)
            ans = f"x = {ans_val:g}"
        elif target == "vy":
            ans = c_coef - (b_coef * b_coef) / (4 * a)
        else:
            return None

        templates = _TARGET_TEMPLATES[target]
        sidx = (self.seed or 0) % len(templates)
        question = templates[sidx]

        img = self._render(pts, a, b_coef, c_coef)
        return question, str(ans), img

    def _render(self, pts, a, b, c) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Range
        xs_pts = [p[0] for p in pts]
        ys_pts = [p[1] for p in pts]
        x_min, x_max = min(xs_pts) - 1, max(xs_pts) + 1
        y_min, y_max = min(ys_pts) - 2, max(ys_pts) + 2
        x_min = min(x_min, -3)
        x_max = max(x_max, 3)
        x_curve = np.linspace(x_min, x_max, 200)
        y_curve = a * x_curve * x_curve + b * x_curve + c
        ax.plot(x_curve, y_curve, color="#2c3e80", linewidth=2)
        # Plot points
        labels = ["A", "B", "C"]
        for (xp, yp), lab in zip(pts, labels):
            ax.scatter([xp], [yp], color="#c0392b", s=70, zorder=5)
            ax.annotate(f"{lab} ({xp:g}, {yp:g})", (xp, yp),
                        textcoords="offset points", xytext=(7, 7),
                        fontsize=10, color="#c0392b", fontweight="bold")
        # Axes
        ax.axhline(0, color="#888", linewidth=0.8)
        ax.axvline(0, color="#888", linewidth=0.8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(min(y_curve.min(), y_min), max(y_curve.max(), y_max))
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
