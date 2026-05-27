"""
Tangent line graph QA -- function plot with tangent line at a point.
Questions: slope, y-intercept, equation, angle with x-axis.
"""
import math
import random
from typing import Dict, Optional, Tuple
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from .standalone_base import StandaloneVisualEnv

class TangentLineGraphQA(StandaloneVisualEnv):
    ENV_NAME = "tangent_line_graph"
    # Tighten numeric tolerance for tight-precision scoring (0.001 abs) for
    # float answers (50-5000x stricter than the default 5%/0.5 path).
    # Tighten to match.
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001

    # ------------------------------------------------------------------ #
    # Per-level configuration
    # ------------------------------------------------------------------ #

    def _level_config(self, level: int) -> dict:
        if level <= 1:
            return {"qtypes": ["slope", "tangent_y_value"],
                    "ftypes": ["quadratic"]}
        if level <= 3:
            return {"qtypes": ["slope", "y_intercept", "tangent_y_value"],
                    "ftypes": ["quadratic", "cubic"]}
        if level <= 5:
            return {"qtypes": ["slope", "y_intercept", "tangent_equation", "tangent_y_value"],
                    "ftypes": ["quadratic", "cubic"]}
        if level <= 7:
            return {"qtypes": ["slope", "y_intercept", "tangent_equation", "angle_with_x", "tangent_y_value"],
                    "ftypes": ["quadratic", "cubic", "sin"]}
        return {"qtypes": ["slope", "y_intercept", "tangent_equation", "angle_with_x", "tangent_y_value"],
                "ftypes": ["quadratic", "cubic", "sin"]}

    def _generate_problem(self, seed: int, parameter: Dict) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        # Use level-aware sub_rng for EVERYTHING (previously `rng = self._rng`
        # was seed-only so L0/L3/L6/L9 at same seed produced identical images).
        sub_rng = random.Random((self.seed or 0) * 1000 + level * 37 + 991)
        rng = sub_rng
        style = self._random_style()
        qtype = parameter.get("question_type", sub_rng.choice(cfg["qtypes"]))

        # Pick function type
        ftype = sub_rng.choice(cfg["ftypes"])
        if ftype == "quadratic":
            a = rng.choice([-2, -1, 1, 2, 3])
            b = rng.randint(-3, 3)
            c = rng.randint(-4, 4)
            x0 = rng.randint(-2, 3)
            f = lambda x: a * x**2 + b * x + c
            fp = lambda x: 2 * a * x + b
            label = f"f(x) = {a}x^2 + {b}x + {c}"
        elif ftype == "cubic":
            a = rng.choice([-1, 1])
            b = rng.randint(-2, 2)
            x0 = rng.randint(-2, 2)
            f = lambda x: a * x**3 + b * x
            fp = lambda x: 3 * a * x**2 + b
            label = f"f(x) = {a}x^3 + {b}x"
        else:
            amp = rng.choice([1, 2, 3])
            x0 = rng.choice([0, 1, -1, 2])
            f = lambda x: amp * np.sin(x)
            fp = lambda x: amp * np.cos(x)
            label = f"f(x) = {amp}*sin(x)"

        y0 = float(f(x0))
        slope = round(float(fp(x0)), 2)
        y_int = round(y0 - slope * x0, 2)
        angle = round(math.degrees(math.atan(slope)), 2)

        # Draw
        sc = style["figsize_scale"]
        fig, ax = plt.subplots(figsize=(7 * sc, 5 * sc))
        fig.patch.set_facecolor(style["bg_color"])
        ax.set_facecolor(style["bg_color"])

        xs = np.linspace(x0 - 4, x0 + 4, 300)
        ys = [f(x) for x in xs]
        ax.plot(xs, ys, color=style["palette"][0], linewidth=style["line_width"] + 0.5, label=label)

        # Tangent line
        tx = np.linspace(x0 - 2.5, x0 + 2.5, 100)
        ty = slope * (tx - x0) + y0
        ax.plot(tx, ty, color=style["palette"][2], linewidth=style["line_width"],
                linestyle="--", label="tangent")

        ax.plot(x0, y0, "o", color=style["palette"][3], markersize=8, zorder=5)
        ax.annotate(f"({x0}, {round(y0, 2)})", xy=(x0, y0),
                    xytext=(x0 + 0.5, y0 + 1), fontsize=style["font_size_base"],
                    fontweight="bold", arrowprops=dict(arrowstyle="->", color="gray"))

        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        self._apply_style(fig, ax, style)
        ax.legend(fontsize=style["font_size_base"] - 1)
        ax.set_title("Function with Tangent Line", fontsize=style["font_size_base"] + 2, fontweight="bold")
        img = self.fig_to_pil(fig, dpi=style["dpi"])

        if qtype == "slope":
            q = f"What is the slope of the tangent line at x = {x0}?"
            return q, str(slope), img
        elif qtype == "y_intercept":
            q = f"What is the y-intercept of the tangent line at x = {x0}? Round to 2 decimals."
            return q, str(y_int), img
        elif qtype == "tangent_equation":
            q = f"The tangent line at x = {x0} has the form y = mx + b. What is m?"
            return q, str(slope), img
        elif qtype == "angle_with_x":
            q = f"What angle (in degrees) does the tangent line at x = {x0} make with the positive x-axis? Round to 2 decimals."
            return q, str(angle), img
        elif qtype == "tangent_y_value":
            xq = x0 + rng.choice([-1, 1, 2])
            yq = round(slope * (xq - x0) + y0, 2)
            q = f"Using the tangent line at x = {x0}, estimate y at x = {xq}. Round to 2 decimals."
            return q, str(yq), img
        return None
