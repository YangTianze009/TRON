"""
Linear graph + axes triangle area: a line y = ax + b is shown on the
coordinate plane, crossing both the x-axis and y-axis. Together with the
origin O, the line and the two axes enclose a right triangle. Compute
the area of this triangle.

The image labels the equation, the two intercepts, and (for higher
levels) only the equation — the model must read the line itself and
identify both intercepts.

Difficulty axes:
  - integer vs irrational intercepts
  - whether the equation is shown vs has to be inferred from two labeled
    points
  - magnitude of the slope / intercept

L0: y = -x + 4 → intercepts (4, 0), (0, 4) → area 8.
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
    "The line shown in the image is y = {eq}. Together with the x-axis and the y-axis it forms a right triangle with the origin O. Compute the area of this triangle.",
    "A straight line y = {eq} appears on the coordinate plane. With the two coordinate axes it bounds a triangle. Find the area of that triangle.",
    "The diagram shows a line y = {eq}. Together with the x-axis and the y-axis it encloses a triangle with vertex O at the origin. Find the area of this triangle.",
    "Find the area of the triangle that the line y = {eq} forms with the two coordinate axes.",
    "The line y = {eq} crosses both axes. Compute the area of the triangle formed by the line and the two axes.",
    "Given the line y = {eq} drawn in the figure, determine the area of the right triangle it makes with the x-axis and the y-axis.",
    "On the coordinate plane the line y = {eq} is shown. It bounds a triangle with the two axes. Find the triangle's area.",
    "Compute the area enclosed by the line y = {eq} and the two coordinate axes.",
    "A line y = {eq} cuts off a triangle from the first/second/third/fourth quadrant region together with the x-axis and y-axis. Find that triangle's area.",
    "Look at the line y = {eq} in the image. Compute the area of the triangle it forms with the two axes.",
    "The image shows the line y = {eq}. Calculate the area of the triangle bounded by this line, the x-axis, and the y-axis.",
    "From the figure, identify the line y = {eq}. Together with the coordinate axes it forms a triangle with the origin. Find its area.",
    "The straight line shown has equation y = {eq}. It meets both axes; with them it encloses a triangle. What is the triangle's area?",
    "Determine the area of the triangle whose three sides are the line y = {eq} and the two coordinate axes.",
    "A line with equation y = {eq} is drawn. It and the two axes form a triangle with the origin O as one vertex. Compute that triangle's area.",
    "Use the image. The line y = {eq}, together with the x-axis and y-axis, bounds a triangle. Find its area and",
    "Inspect the line y = {eq} in the figure. Compute the area of the triangle it makes with the two coordinate axes.",
]


def _format_linear(slope: float, intercept: float) -> str:
    """Format y = slope*x + intercept as a clean string like '-x + 4'."""
    # slope term
    if abs(slope - round(slope)) < 1e-9:
        slope_int = int(round(slope))
        if slope_int == 1:
            slope_str = "x"
        elif slope_int == -1:
            slope_str = "-x"
        elif slope_int == 0:
            slope_str = ""
        else:
            slope_str = f"{slope_int}x"
    else:
        slope_str = f"{slope:g}x"

    if abs(intercept - round(intercept)) < 1e-9:
        intercept_val = int(round(intercept))
    else:
        intercept_val = float(f"{intercept:g}")

    if slope_str == "":
        return f"{intercept_val}"

    if intercept_val == 0:
        return slope_str
    if intercept_val > 0:
        return f"{slope_str} + {intercept_val}"
    return f"{slope_str} - {abs(intercept_val)}"


class LineAxesTriangleAreaQA(StandaloneVisualEnv):
    ENV_NAME = "line_axes_triangle_area"
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
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1153 + level * 67 + 11)

        # Pick (slope a, intercept b) with a != 0, b != 0 so the line really
        # crosses both axes off-origin.
        # x-intercept: -b / a; y-intercept: b. Area = (1/2) |x_int * y_int|
        #            = (1/2) |b^2 / a| = b*b / (2*|a|).
        if level == 0:
            # Easiest: y = -x + 4 (use the canonical L0 dictated by the task)
            a = -1
            b = 4
        elif level <= 2:
            a = rng.choice([-1, 1, -2, 2])
            b = rng.choice([2, 3, 4, 5, 6, 8])
            if rng.random() < 0.5:
                b = -b
        elif level <= 4:
            a = rng.choice([-3, -2, -1, 1, 2, 3])
            b_abs = rng.choice([2, 3, 4, 5, 6, 8, 9, 10, 12])
            b = b_abs if rng.random() < 0.5 else -b_abs
        elif level <= 6:
            # decimals appear, intercepts can be non-integer
            a = rng.choice([-3, -2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2, 3])
            b = rng.choice([3, 4, 5, 6, 7.5, 9, 10, 12])
            if rng.random() < 0.5:
                b = -b
        elif level <= 8:
            # higher level: irrational-ish slopes via fractions, larger nums
            a = rng.choice([-4, -3, -2.5, -2, -1.5, 1.5, 2, 2.5, 3, 4])
            b = rng.choice([3, 5, 6, 7, 8, 9, 12, 15])
            if rng.random() < 0.5:
                b = -b
        else:
            # 2026-05-04: bumped L9 difficulty (was 100% saturated → wider
            # fractional slope pool + much larger b values, producing
            # area = b^2 / (2|a|) with large numerators).
            a = rng.choice([-7, -6, -5, -4, -3.5, -2.5, -1.5, 1.5, 2.5, 3.5,
                            4, 5, 6, 7])
            b = rng.choice([7, 9, 11, 13, 14, 16, 18, 20, 22, 25, 28])
            if rng.random() < 0.5:
                b = -b

        x_int = -b / a
        y_int = b
        area = abs(x_int * y_int) / 2.0

        # Skip degenerate
        if abs(area) < 1e-9:
            return None

        # Format
        eq_str = _format_linear(a, b)
        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(eq=eq_str)

        if abs(area - round(area)) < 1e-9:
            answer = f"{int(round(area))}"
        else:
            answer = f"{area:.4g}"

        img = self._render(a, b, x_int, y_int)
        return question, answer, img

    def _render(self, a, b, x_int, y_int) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        # Determine plot range so the triangle is visible with margin
        max_extent = max(abs(x_int), abs(y_int)) + 2
        xs = np.linspace(-max_extent, max_extent, 200)
        ys = a * xs + b
        ax.plot(xs, ys, color="#1f77b4", linewidth=2.2,
                label=f"y = {_format_linear(a, b)}")

        # Fill the triangle (origin, x-intercept, y-intercept)
        tri_x = [0, x_int, 0]
        tri_y = [0, 0, y_int]
        ax.fill(tri_x, tri_y, color="#ffcccc", alpha=0.45, edgecolor="none")

        # Axes
        ax.axhline(0, color="#222", linewidth=1.0)
        ax.axvline(0, color="#222", linewidth=1.0)
        ax.grid(True, alpha=0.3, linestyle="--")

        # Annotate intercepts
        # x-intercept point
        ax.scatter([x_int], [0], color="#c0392b", s=55, zorder=5)
        ax.scatter([0], [y_int], color="#c0392b", s=55, zorder=5)
        ax.scatter([0], [0], color="#222", s=45, zorder=5)
        # Pretty integer/decimal labels
        def _fmt(v):
            if abs(v - round(v)) < 1e-9:
                return f"{int(round(v))}"
            return f"{v:g}"
        ax.annotate(f"({_fmt(x_int)}, 0)", (x_int, 0),
                    textcoords="offset points",
                    xytext=(8, -14), fontsize=11, color="#c0392b",
                    fontweight="bold")
        ax.annotate(f"(0, {_fmt(y_int)})", (0, y_int),
                    textcoords="offset points",
                    xytext=(8, 6), fontsize=11, color="#c0392b",
                    fontweight="bold")
        ax.annotate("O", (0, 0), textcoords="offset points",
                    xytext=(-12, -12), fontsize=11, color="#222",
                    fontweight="bold")

        ax.set_xlim(-max_extent, max_extent)
        ax.set_ylim(-max_extent, max_extent)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="best", fontsize=11)
        ax.set_title(f"y = {_format_linear(a, b)}", fontsize=12)

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
