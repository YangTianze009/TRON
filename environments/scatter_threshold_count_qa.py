"""
Scatter plot threshold counting: render N points and ask the model to count
how many points satisfy a numeric condition such as X > T, Y > T, or both.

Tests visual extraction + filter + count primitive (Number-in-General gap).
Integer answer.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_X = [
    "How many points in the scatter plot have an x-coordinate greater than {T}? Integer count in <answer>...</answer>.",
    "Count the points whose x-value exceeds {T}. Integer in <answer>...</answer>.",
    "How many of the plotted points lie to the right of x = {T}? Integer in <answer>...</answer>.",
    "Tally the points with x > {T} in the scatter plot. Integer in <answer>...</answer>.",
    "How many points satisfy x-coordinate > {T}? Integer in <answer>...</answer>.",
    "Count the scatter points where x exceeds {T}. Integer in <answer>...</answer>.",
    "Number of points to the right of x = {T}? Integer in <answer>...</answer>.",
    "How many points have x-value strictly greater than {T}? Integer in <answer>...</answer>.",
    "Inspect the scatter plot — how many points have x > {T}? Integer in <answer>...</answer>.",
    "Report the count of points with x-coordinate > {T}. Integer in <answer>...</answer>.",
    "Of the displayed points, how many lie at x > {T}? Integer in <answer>...</answer>.",
    "Among the scatter points, how many have x greater than {T}? Integer in <answer>...</answer>.",
    "Find the number of points with x > {T}. Integer in <answer>...</answer>.",
    "How many scatter points are to the right of x = {T}? Integer in <answer>...</answer>.",
    "Filter the scatter points to those with x > {T} and report the count. Integer in <answer>...</answer>.",
    "Count: how many points show x-value above {T}? Integer in <answer>...</answer>.",
]

_TEMPLATES_Y = [
    "How many points in the scatter plot have a y-coordinate greater than {T}? Integer count in <answer>...</answer>.",
    "Count the points whose y-value exceeds {T}. Integer in <answer>...</answer>.",
    "How many of the plotted points lie above y = {T}? Integer in <answer>...</answer>.",
    "Tally the points with y > {T} in the scatter plot. Integer in <answer>...</answer>.",
    "How many points satisfy y-coordinate > {T}? Integer in <answer>...</answer>.",
    "Count the scatter points where y exceeds {T}. Integer in <answer>...</answer>.",
    "Number of points above y = {T}? Integer in <answer>...</answer>.",
    "How many points have y-value strictly greater than {T}? Integer in <answer>...</answer>.",
    "Inspect the scatter plot — how many points have y > {T}? Integer in <answer>...</answer>.",
    "Report the count of points with y-coordinate > {T}. Integer in <answer>...</answer>.",
    "Of the displayed points, how many lie at y > {T}? Integer in <answer>...</answer>.",
    "Among the scatter points, how many have y greater than {T}? Integer in <answer>...</answer>.",
    "Find the number of points with y > {T}. Integer in <answer>...</answer>.",
    "How many scatter points are above y = {T}? Integer in <answer>...</answer>.",
    "Filter the scatter points to those with y > {T} and report the count. Integer in <answer>...</answer>.",
    "Count: how many points show y-value above {T}? Integer in <answer>...</answer>.",
]

_TEMPLATES_BOTH = [
    "How many points in the scatter plot satisfy both x > {Tx} and y > {Ty}? Integer count in <answer>...</answer>.",
    "Count the points with x > {Tx} AND y > {Ty}. Integer in <answer>...</answer>.",
    "How many plotted points lie in the region x > {Tx} and y > {Ty} (upper-right quadrant)? Integer in <answer>...</answer>.",
    "Tally the points satisfying both x > {Tx} and y > {Ty}. Integer in <answer>...</answer>.",
    "How many points are in the upper-right region where x > {Tx} and y > {Ty}? Integer in <answer>...</answer>.",
    "Number of scatter points with x > {Tx} and y > {Ty}? Integer in <answer>...</answer>.",
    "Filter to points where both x > {Tx} and y > {Ty}, then count them. Integer in <answer>...</answer>.",
    "How many points have x exceeding {Tx} AND y exceeding {Ty}? Integer in <answer>...</answer>.",
    "Among the scatter points, how many satisfy both x > {Tx} and y > {Ty}? Integer in <answer>...</answer>.",
    "How many plotted dots have x > {Tx} and also y > {Ty}? Integer in <answer>...</answer>.",
    "Count the points lying both to the right of x = {Tx} and above y = {Ty}. Integer in <answer>...</answer>.",
    "Report the count of points with (x > {Tx}) AND (y > {Ty}). Integer in <answer>...</answer>.",
    "How many points are simultaneously above y = {Ty} and to the right of x = {Tx}? Integer in <answer>...</answer>.",
    "Find the number of scatter points satisfying x > {Tx} and y > {Ty}. Integer in <answer>...</answer>.",
    "Joint condition: x > {Tx} and y > {Ty}. How many points? Integer in <answer>...</answer>.",
    "Count points in the region x > {Tx}, y > {Ty}. Integer in <answer>...</answer>.",
]


_AXIS_LABELS = [
    ("x", "y"),
    ("X", "Y"),
    ("Time", "Value"),
    ("Feature 1", "Feature 2"),
    ("a", "b"),
    ("Input", "Output"),
]

_MARKERS = ["o", "s", "^", "D", "v", "p", "h"]


def _format_int(x) -> str:
    return str(int(x))


class ScatterThresholdCountQA(StandaloneVisualEnv):
    ENV_NAME = "scatter_threshold_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            n_points = 5
            modes = ["x"]
        elif level <= 2:
            n_points = 7
            modes = ["x", "y"]
        elif level <= 4:
            n_points = 10
            modes = ["x", "y"]
        elif level <= 6:
            n_points = 14
            modes = ["x", "y", "both"]
        elif level <= 8:
            n_points = 20
            modes = ["x", "y", "both"]
        else:
            n_points = 28
            modes = ["x", "y", "both"]
        return {"level": level, "n_points": n_points, "modes": modes}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1061 + level * 79 + 31)

        n = cfg["n_points"]
        mode = rng.choice(cfg["modes"])

        # Generate points on a 0-100 grid for clean integer reading
        xs = [rng.randint(5, 95) for _ in range(n)]
        ys = [rng.randint(5, 95) for _ in range(n)]

        # L0 special: very obvious — 5 points, threshold splits 2 above / 3 below clearly
        if level == 0:
            xs = [10, 20, 30, 70, 85]
            ys = [40, 30, 60, 25, 70]
            T = 50
            mode = "x"
            count = sum(1 for x in xs if x > T)
            sidx = (self.seed or 0) % len(_TEMPLATES_X)
            question = _TEMPLATES_X[sidx].format(T=T)
        elif mode == "x":
            T = rng.choice([20, 30, 40, 50, 60, 70])
            count = sum(1 for x in xs if x > T)
            sidx = (self.seed or 0) % len(_TEMPLATES_X)
            question = _TEMPLATES_X[sidx].format(T=T)
        elif mode == "y":
            T = rng.choice([20, 30, 40, 50, 60, 70])
            count = sum(1 for y in ys if y > T)
            sidx = (self.seed or 0) % len(_TEMPLATES_Y)
            question = _TEMPLATES_Y[sidx].format(T=T)
        else:  # both
            Tx = rng.choice([30, 40, 50, 60])
            Ty = rng.choice([30, 40, 50, 60])
            count = sum(1 for x, y in zip(xs, ys) if x > Tx and y > Ty)
            sidx = (self.seed or 0) % len(_TEMPLATES_BOTH)
            question = _TEMPLATES_BOTH[sidx].format(Tx=Tx, Ty=Ty)

        answer = _format_int(count)

        # Compute thresholds for visual reference lines
        if mode == "x":
            thresh = {"x": T}
        elif mode == "y":
            thresh = {"y": T}
        else:
            thresh = {"x": Tx, "y": Ty}

        img = self._render_scatter(xs, ys, thresh, rng)
        return question, answer, img

    def _render_scatter(self, xs, ys, thresh, rng):
        palette = rng.choice(self._COLOR_PALETTES)
        marker = rng.choice(_MARKERS)
        ax_label = rng.choice(_AXIS_LABELS)
        c = palette[0]
        fig, ax = plt.subplots(figsize=(6.5, 5.0), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.scatter(xs, ys, s=90, color=c, marker=marker,
                   edgecolor="#1a1a1a", linewidth=0.6, alpha=0.85, zorder=3)
        if "x" in thresh:
            ax.axvline(thresh["x"], color="#c0392b", linewidth=1.6,
                       linestyle="--", label=f"x = {thresh['x']}", zorder=2)
        if "y" in thresh:
            ax.axhline(thresh["y"], color="#16a085", linewidth=1.6,
                       linestyle="--", label=f"y = {thresh['y']}", zorder=2)
        ax.set_xlabel(ax_label[0], fontsize=12)
        ax.set_ylabel(ax_label[1], fontsize=12)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=10, loc="upper left")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
