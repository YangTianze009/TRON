"""
Find the (x, y) intersection point of two distinct, non-parallel lines drawn
on a coordinate plane. Both lines are labeled by their equations
y = m1 x + b1 and y = m2 x + b2 in the figure.

Solving:  m1 x + b1 = m2 x + b2  ⇒  x = (b2 - b1) / (m1 - m2)
          y = m1 x + b1

L0: y = x and y = 2 → intersect at (2, 2).
"""
import math
import random
import re
from io import BytesIO
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "The image shows two lines y = {l1} and y = {l2} on the coordinate plane. Find the (x, y) coordinates of their intersection point.",
    "Two lines are drawn: y = {l1} and y = {l2}. Compute the intersection point (x, y).",
    "Determine the intersection of the lines y = {l1} and y = {l2}. Output the (x, y) coordinates as `(x, y)`.",
    "Find where the two lines y = {l1} and y = {l2} cross.",
    "On the figure, the lines y = {l1} and y = {l2} intersect at one point. Compute its coordinates and write them as `(x, y)`.",
    "Solve the system y = {l1} and y = {l2}. Express the unique solution as the point (x, y).",
    "From the figure: lines y = {l1} and y = {l2} meet at a single point. Find its coordinates `(x, y)`.",
    "Two non-parallel lines, y = {l1} and y = {l2}, are shown. Find their intersection (x, y) and",
    "Compute the intersection point of y = {l1} and y = {l2}.",
    "Where do y = {l1} and y = {l2} cross?",
    "Locate the intersection point of the two lines y = {l1} and y = {l2}.",
    "Find the unique solution to the linear system y = {l1} and y = {l2}, expressed as (x, y).",
    "Determine the (x, y) coordinates where the lines y = {l1} and y = {l2} intersect.",
    "Read the two line equations from the figure: y = {l1} and y = {l2}. Compute their intersection (x, y).",
    "Both lines in the figure are labeled. y = {l1} meets y = {l2} at one point. Give that point as (x, y).",
    "Find the point common to lines y = {l1} and y = {l2}.",
    "Two straight lines are drawn: y = {l1} and y = {l2}. Compute the intersection (x, y) and",
]


def _format_linear(m: float, b: float) -> str:
    """Format y = m x + b. m and b are integer-valued floats here."""
    mi = int(round(m))
    bi = int(round(b))
    parts = []
    if mi == 0:
        return f"{bi}"
    if mi == 1:
        parts.append("x")
    elif mi == -1:
        parts.append("-x")
    else:
        parts.append(f"{mi}x")
    if bi == 0:
        return parts[0]
    if bi > 0:
        parts.append(f"+ {bi}")
    else:
        parts.append(f"- {abs(bi)}")
    return " ".join(parts)


class TwoLinesIntersectionQA(StandaloneVisualEnv):
    ENV_NAME = "two_lines_intersection"
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
        rng = random.Random((self.seed or 0) * 1597 + level * 59 + 17)

        if level == 0:
            # Canonical L0 from task: y = x and y = 2 → (2, 2).
            m1, b1 = 1, 0
            m2, b2 = 0, 2
            x_int = 2.0
            y_int = 2.0
        else:
            # Pick integer (x, y) intersection target, then pick two slopes
            # m1 != m2, derive intercepts so both lines pass through (x_int, y_int).
            for _ in range(80):
                if level <= 2:
                    x_int = rng.randint(-3, 3)
                    y_int = rng.randint(-3, 3)
                    slopes_pool = [-2, -1, 0, 1, 2]
                elif level <= 4:
                    x_int = rng.randint(-5, 5)
                    y_int = rng.randint(-5, 5)
                    slopes_pool = [-3, -2, -1, 0, 1, 2, 3]
                elif level <= 6:
                    x_int = rng.randint(-6, 6)
                    y_int = rng.randint(-6, 6)
                    slopes_pool = [-3, -2, -1, 0, 1, 2, 3, 4, -4]
                else:
                    # 2026-05-04: L9 was 100%. Wider range + harder slopes.
                    # 2026-05-04: bumped L9 difficulty further — even wider range.
                    # 2026-05-04 R3: softened bump — keep wider range from
                    # first bump but drop the second wider-range bump (was
                    # -25..25 / slopes ±7). L9 dropped to 0.70; revert to the
                    # ±10 / slopes ±5 first bump.
                    x_int = rng.randint(-10, 10)
                    y_int = rng.randint(-10, 10)
                    slopes_pool = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

                m1 = rng.choice(slopes_pool)
                m2 = rng.choice(slopes_pool)
                if m1 == m2:
                    continue
                # Intercepts to pass through (x_int, y_int)
                b1 = y_int - m1 * x_int
                b2 = y_int - m2 * x_int
                # Bound intercepts so the figure stays readable
                # 2026-05-04: bigger intercept bound at L>=7 for harder L9 range.
                # 2026-05-04 R3: tightened bound back to 40 (was 80) since
                # L9 range narrowed to ±10.
                b_bound = 40 if level >= 7 else 18
                if abs(b1) > b_bound or abs(b2) > b_bound:
                    continue
                break
            else:
                return None

        # Format answer as (x, y)
        def _fmt(v):
            if abs(v - round(v)) < 1e-9:
                return f"{int(round(v))}"
            return f"{v:g}"
        answer = f"({_fmt(x_int)}, {_fmt(y_int)})"

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(
            l1=_format_linear(m1, b1),
            l2=_format_linear(m2, b2))

        img = self._render(m1, b1, m2, b2, x_int, y_int)
        return question, answer, img

    def _render(self, m1, b1, m2, b2, x_int, y_int) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # x-range based on intersection and intercepts
        max_extent = max(abs(x_int), abs(y_int), abs(b1), abs(b2)) + 3
        x_lo, x_hi = -max_extent, max_extent
        xs = np.linspace(x_lo, x_hi, 200)

        ys1 = m1 * xs + b1
        ys2 = m2 * xs + b2
        ax.plot(xs, ys1, color="#1f77b4", linewidth=2,
                label=f"y = {_format_linear(m1, b1)}")
        ax.plot(xs, ys2, color="#d62728", linewidth=2,
                label=f"y = {_format_linear(m2, b2)}")
        # Intersection
        ax.scatter([x_int], [y_int], color="#2ca02c", s=80, zorder=5,
                   edgecolor="black", linewidth=1.0)

        def _fmt(v):
            if abs(v - round(v)) < 1e-9:
                return f"{int(round(v))}"
            return f"{v:g}"
        ax.annotate(f"({_fmt(x_int)}, {_fmt(y_int)})", (x_int, y_int),
                    textcoords="offset points",
                    xytext=(10, 10), fontsize=12, color="#2ca02c",
                    fontweight="bold")

        # Axes & grid
        ax.axhline(0, color="#222", linewidth=0.9)
        ax.axvline(0, color="#222", linewidth=0.9)
        ax.grid(True, alpha=0.3, linestyle="--")

        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(x_lo, x_hi)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best", fontsize=10)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    # ---------- custom answer checking for (x, y) tuple ---------- #
    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        gt = self._parse_pair(ground_truth)
        pred = self._parse_pair(predicted)
        if gt is None:
            return super()._check_answer(predicted, ground_truth)
        if pred is None:
            return False
        gx, gy = gt
        px, py = pred
        # Tolerance: 5% relative or 0.1 absolute
        if abs(px - gx) > max(0.05 * abs(gx), 0.1) and abs(px - gx) > 0.05:
            return False
        if abs(py - gy) > max(0.05 * abs(gy), 0.1) and abs(py - gy) > 0.05:
            return False
        return True

    @staticmethod
    def _parse_pair(s: str):
        """Parse '(2, 3)', 'x=2, y=3', '2,3', '{2,3}', '[2, 3]' → (2.0, 3.0).
        Returns None if can't parse exactly two numeric values."""
        if s is None:
            return None
        t = s.strip().lower()
        # Strip wrappers
        t = t.strip("()[]{}")
        # Remove "x=" / "y=" prefixes embedded in the text
        t = re.sub(r"\bx\s*=\s*", "", t)
        t = re.sub(r"\by\s*=\s*", "", t)
        # split by comma, semicolon, "and"
        parts = re.split(r"[,;]|\band\b", t)
        vals = []
        for p in parts:
            p = p.strip().strip("()[]{} ")
            if not p:
                continue
            try:
                vals.append(float(p))
                continue
            except ValueError:
                pass
            m = re.match(r"^(-?\d+)\s*/\s*(\d+)$", p)
            if m:
                vals.append(int(m.group(1)) / int(m.group(2)))
                continue
            return None
        if len(vals) != 2:
            return None
        return tuple(vals)
