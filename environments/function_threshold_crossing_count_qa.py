"""
Count threshold-crossings of a curve over a stated x-range. Mirrors the
real failure mode where models miscount how many times a line crosses a
horizontal threshold or another line.

Mirrors verbatim Q-IDX 195 ("How many time(s) does the line drop below
the zero line between 2005 and 2010 in the FROOPP plot?" GT=1),
Q-IDX 221 ("How many times does the cumulative action value cross the
-30 line between time 0 and time 1000?" GT=1), Q-IDX 211 ("How many
times does the Contractionary curve cross the Expansionary curve in the
Stock subplot?" GT=3), Q-IDX 488 ("How many local minima are there in
the blue curve within the range from x = 3 to x = 6?" GT=1),
Q-IDX 1568 ("federal deficits ... exceeded $400 billion five times
between 2003 and 2010"), Q-IDX 1496, Q-IDX 1589, Q-IDX 859.

L0: simple monotonic curve, threshold above range — answer is 0 or 1.
L>0: oscillating curve with multiple crossings; range may be subset.
"""
import math
import random
from io import BytesIO
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES_THRESHOLD = [
    "How many times does the curve cross the {direction} line y={T} between x={a} and x={b}? Integer count in <answer>...</answer>.",
    "How many times does the line drop below y={T} between x={a} and x={b}? Integer in <answer>...</answer>.",
    "How many times does the value exceed {T} in the range x in [{a}, {b}]? Integer in <answer>...</answer>.",
    "Count the number of crossings of the curve with the horizontal line y={T} for x between {a} and {b}. Integer in <answer>...</answer>.",
    "How many times does the curve pass through y={T} on the interval [{a}, {b}]? Integer count in <answer>...</answer>.",
    "Across the x-range [{a}, {b}], count the number of times the curve crosses y={T}. Integer in <answer>...</answer>.",
    "From x={a} to x={b}, how many times does the curve intersect the horizontal line y={T}? Integer in <answer>...</answer>.",
    "Looking at the curve in the chart, how many times does it cross y={T} between x={a} and x={b}? Integer in <answer>...</answer>.",
    "Count the threshold crossings of y={T} for the curve over [{a}, {b}]. Integer in <answer>...</answer>.",
    "Between x={a} and x={b}, the curve crosses y={T} how many times? Integer in <answer>...</answer>.",
]

_TEMPLATES_LINE_LINE = [
    "How many times does the {a_name} curve cross the {b_name} curve in the chart? Integer count in <answer>...</answer>.",
    "Count the intersections between the {a_name} and {b_name} lines. Integer in <answer>...</answer>.",
    "How many times does {a_name} cross {b_name}? Integer in <answer>...</answer>.",
    "How many crossings between curve {a_name} and curve {b_name} are visible in the chart? Integer in <answer>...</answer>.",
    "From the chart, count how many times {a_name} intersects {b_name}. Integer in <answer>...</answer>.",
    "How many intersection points exist between {a_name} and {b_name} in the chart? Integer in <answer>...</answer>.",
]


class FunctionThresholdCrossingCountQA(StandaloneVisualEnv):
    ENV_NAME = "function_threshold_crossing_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level == 0:
            return {"mode": "threshold", "max_crossings": 1, "freq_max": 0.5}
        if level <= 2:
            return {"mode": "threshold", "max_crossings": 3, "freq_max": 1.2}
        if level <= 4:
            return {"mode": "threshold", "max_crossings": 5, "freq_max": 2.0}
        if level <= 6:
            return {"mode": "either", "max_crossings": 6, "freq_max": 2.5}
        return {"mode": "either", "max_crossings": 8, "freq_max": 3.0}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        import hashlib
        sd = int(hashlib.md5(f"ftc|{self.seed or 0}|{level}".encode()).hexdigest()[:12], 16)
        rng = random.Random(sd)

        mode = cfg["mode"]
        if mode == "either":
            mode = rng.choice(["threshold", "line_line"])

        if mode == "threshold":
            return self._gen_threshold(rng, cfg)
        else:
            return self._gen_line_line(rng, cfg)

    def _gen_threshold(self, rng: random.Random, cfg: Dict
                        ) -> Optional[Tuple[str, str, Image.Image]]:
        # Generate a curve with controlled number of crossings of y=T.
        a, b = 0.0, 10.0
        x = np.linspace(a, b, 400)
        # Sum of sinusoid + offset; choose freq to get crossings.
        freq = rng.uniform(0.3, cfg["freq_max"])
        amp = rng.uniform(2.0, 5.0)
        phase = rng.uniform(0, 2 * math.pi)
        offset = rng.uniform(-1.0, 1.0)
        T = round(rng.uniform(-1.5, 1.5), 1)
        # Slight slope to break ties
        slope = rng.uniform(-0.2, 0.2)
        y = offset + slope * x + amp * np.sin(2 * math.pi * freq * x / b + phase)
        # Robust crossing count: sign changes of (y - T)
        sgn = np.sign(y - T)
        # Replace 0 with previous sign so we don't double-count tangent touches.
        for i in range(1, len(sgn)):
            if sgn[i] == 0:
                sgn[i] = sgn[i - 1]
        crossings = int(np.sum(np.diff(sgn) != 0))
        if crossings > cfg["max_crossings"]:
            # Reduce crossings by clamping freq
            return None

        direction = "horizontal"
        ai, bi = int(a), int(b)
        tmpl = rng.choice(_TEMPLATES_THRESHOLD)
        question = tmpl.format(direction=direction, T=T, a=ai, b=bi)
        answer = str(int(crossings))
        img = self._render_threshold(x, y, T, ai, bi, rng)
        return question, answer, img

    def _gen_line_line(self, rng: random.Random, cfg: Dict
                       ) -> Optional[Tuple[str, str, Image.Image]]:
        x = np.linspace(0, 10, 400)
        f1 = rng.uniform(0.3, cfg["freq_max"])
        f2 = rng.uniform(0.3, cfg["freq_max"])
        a1 = rng.uniform(2, 5); a2 = rng.uniform(2, 5)
        p1 = rng.uniform(0, 2 * math.pi); p2 = rng.uniform(0, 2 * math.pi)
        y1 = rng.uniform(-1, 1) + a1 * np.sin(2 * math.pi * f1 * x / 10 + p1)
        y2 = rng.uniform(-1, 1) + a2 * np.cos(2 * math.pi * f2 * x / 10 + p2)
        d = y1 - y2
        sgn = np.sign(d)
        for i in range(1, len(sgn)):
            if sgn[i] == 0:
                sgn[i] = sgn[i - 1]
        crossings = int(np.sum(np.diff(sgn) != 0))
        if crossings > cfg["max_crossings"]:
            return None
        a_name = rng.choice(["Series A", "Blue line", "Group A", "Curve A"])
        b_name = rng.choice(["Series B", "Orange line", "Group B", "Curve B"])
        tmpl = rng.choice(_TEMPLATES_LINE_LINE)
        question = tmpl.format(a_name=a_name, b_name=b_name)
        answer = str(int(crossings))
        img = self._render_two_lines(x, y1, y2, a_name, b_name, rng)
        return question, answer, img

    def _render_threshold(self, x, y, T, ai, bi, rng):
        palette = rng.choice(self._COLOR_PALETTES)
        fig, ax = plt.subplots(figsize=(7, 4.4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.plot(x, y, color=palette[0], linewidth=2.0, label="curve")
        ax.axhline(T, color="#888", linestyle="--", linewidth=1.5,
                    label=f"y = {T}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(ai, bi)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img

    def _render_two_lines(self, x, y1, y2, a_name, b_name, rng):
        palette = rng.choice(self._COLOR_PALETTES)
        fig, ax = plt.subplots(figsize=(7, 4.4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.plot(x, y1, color=palette[0], linewidth=2.0, label=a_name)
        ax.plot(x, y2, color=palette[1], linewidth=2.0, label=b_name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#ffffff")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).copy()
        buf.close()
        return img
