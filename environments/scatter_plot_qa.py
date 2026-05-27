"""
Scatter Plot Visual QA Environment.

Capabilities: V3 (chart extraction), V2 (label reading), R1 (arithmetic), R4 (statistical)

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 5 points, ask "max y" (simple value reading).
L1: 5 points, ask "max x" (simple value reading).
L2: 8 points, ask "max y" (noisy data).
L3: 5 points with clear trend, ask "trend direction".
L4: 8 points with clear trend, ask "trend direction".
L5: 10 points, ask "count above y=T".
L6: 15 points, ask "count in quadrant".
L7: 18 points with outlier, ask "outlier coordinates".
L8: 20 points clustered, ask "max x".
L9: 20 points with outlier, ask "closest pair distance".

parameter = {"level": int in [0, 9]}
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .base import StandaloneVisualEnv
from ._mcq_letter_helper import maybe_mcq_letter_wrap

_X_LABELS = [
    "Temperature (°C)", "Age (years)", "Weight (kg)", "Height (cm)",
    "Income ($K)", "Experience (years)", "Distance (km)", "Price ($)",
    "Study Hours", "Marketing Spend ($K)", "Square Footage",
]
_Y_LABELS = [
    "Sales ($K)", "Performance Score", "Revenue ($M)", "Satisfaction Rating",
    "Test Score", "Productivity Index", "Fuel Efficiency",
    "Response Time", "Profit Margin (%)", "Customer Count", "Output (units)",
]
_TITLE_TEMPLATES = [
    "{x} vs {y}",
    "{y} as a Function of {x}",
    "Scatter Analysis — {x} / {y}",
    "{x} vs {y} Distribution",
]
_MARKER_STYLES = ["o", "s", "^", "D", "v", "p", "h"]

class ScatterPlotQA(StandaloneVisualEnv):
    ENV_NAME = "scatter_plot"
    # Visual scatter precision is ~5 units; tighter tol penalizes legit visual estimates.
    BENCHMARK_NUM_TOLERANCE_ABS = 5.0
    BENCHMARK_NUM_TOLERANCE_REL = 0.10

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        for _ in range(15):
            try:
                result = self._dispatch(level)
                if result is not None:
                    self._primary_complexity_feature = level * 5 + len(result[1])
                    return result
            except Exception:
                continue
        return None

    def _sub_rng(self, level: int) -> random.Random:
        return random.Random((self.seed or 0) * 1000 + level * 37 + 991)

    def _level_config(self, level: int) -> Dict:
        # Reordered: simple value-reading at L0-L2, trend at L3-L5,
        # counting/quadrant at L6-L7, computed tasks at L8-L9.
        # Fixes L0=0.40 vs L3=1.00 inversion (trend harder than max_y read).
        if level == 0:
            return {"n": 5, "data": "trend_clear", "qtype": "max_y"}
        if level == 1:
            return {"n": 5, "data": "no_trend", "qtype": "max_x"}
        if level == 2:
            return {"n": 8, "data": "trend_noisy", "qtype": "max_y"}
        if level == 3:
            return {"n": 5, "data": "trend_clear", "qtype": "trend"}
        if level == 4:
            return {"n": 8, "data": "trend_clear", "qtype": "trend"}
        if level == 5:
            return {"n": 10, "data": "no_trend", "qtype": "count_above"}
        if level == 6:
            return {"n": 15, "data": "no_trend", "qtype": "count_quadrant"}
        if level == 7:
            return {"n": 18, "data": "no_trend_outlier", "qtype": "outlier"}
        if level == 8:
            return {"n": 20, "data": "clusters", "qtype": "max_x"}
        # L9: "closest_pair" asked for Euclidean distance to 2 decimals,
        # which is not reliably recoverable from visual inspection alone
        # (answer rounding too tight for visual estimation). Use outlier
        # (1-decimal coord tuple) on 20 points — harder than L7 due to
        # more clutter and random data.
        return {"n": 20, "data": "no_trend_outlier", "qtype": "outlier"}

    def _gen_points(self, rng, n, mode):
        np_rng = np.random.RandomState(rng.randint(0, 1_000_000))
        if mode == "trend_clear":
            slope = rng.choice([1.5, 2.0, -1.5, -2.0])
            intercept = rng.uniform(20, 50)
            xs = sorted([round(rng.uniform(10, 90), 1) for _ in range(n)])
            ys = [round(slope * x + intercept + rng.uniform(-3, 3), 1) for x in xs]
            return xs, ys
        if mode == "no_trend":
            xs = [round(rng.uniform(10, 90), 1) for _ in range(n)]
            ys = [round(rng.uniform(10, 90), 1) for _ in range(n)]
            return xs, ys
        if mode == "trend_noisy":
            slope = rng.uniform(-1.5, 1.5)
            intercept = rng.uniform(20, 60)
            xs = [round(rng.uniform(5, 95), 1) for _ in range(n)]
            ys = [round(slope * x + intercept + rng.uniform(-15, 15), 1) for x in xs]
            return xs, ys
        if mode == "no_trend_outlier":
            xs = [round(rng.uniform(20, 80), 1) for _ in range(n - 1)]
            ys = [round(rng.uniform(20, 80), 1) for _ in range(n - 1)]
            # outlier at edge
            xs.append(round(rng.choice([2, 95]), 1))
            ys.append(round(rng.choice([2, 95]), 1))
            return xs, ys
        if mode == "clusters":
            xs, ys = [], []
            num_c = 3
            per = n // num_c
            for _ in range(num_c):
                cx = rng.uniform(15, 85)
                cy = rng.uniform(15, 85)
                spread = rng.uniform(3, 8)
                for _ in range(per):
                    xs.append(round(cx + rng.uniform(-spread, spread), 1))
                    ys.append(round(cy + rng.uniform(-spread, spread), 1))
            while len(xs) < n:
                xs.append(round(rng.uniform(10, 90), 1))
                ys.append(round(rng.uniform(10, 90), 1))
            return xs, ys
        return [], []

    def _dispatch(self, level: int):
        rng = self._sub_rng(level)
        cfg = self._level_config(level)
        xs, ys = self._gen_points(rng, cfg["n"], cfg["data"])
        if len(xs) < 2:
            return None
        question, answer = self._make_qa(rng, cfg["qtype"], xs, ys)
        if question is None:
            return None
        image = self._render(rng, xs, ys)
        # C20 MCQ-letter-only mode (30% wrap when answer is numeric).
        wrapped = maybe_mcq_letter_wrap(rng, question, answer, rate=0.30)
        if wrapped is not None:
            return wrapped[0], wrapped[1], image
        return question, str(answer), image

    def _make_qa(self, rng, qtype, xs, ys):
        n = len(xs)
        if qtype == "trend":
            x_arr = np.array(xs)
            y_arr = np.array(ys)
            x_mean = x_arr.mean()
            y_mean = y_arr.mean()
            num = float(np.sum((x_arr - x_mean) * (y_arr - y_mean)))
            den = float(np.sum((x_arr - x_mean) ** 2))
            if abs(den) < 1e-9:
                return None, None
            slope = num / den
            if abs(slope) < 0.15:
                direction = "no clear trend"
            elif slope > 0:
                direction = "positive"
            else:
                direction = "negative"
            stems = [
                "What is the overall trend direction of the data? "
                "Answer 'positive', 'negative', or 'no clear trend'.",
                "Looking at the scatter plot, is there a positive correlation, "
                "negative correlation, or no clear trend between x and y?",
            ]
            return rng.choice(stems), direction
        if qtype == "max_y":
            return ("What is the y-coordinate of the point with the largest y value? "
                    "Answer with a single number.", max(ys))
        if qtype == "max_x":
            return ("What is the x-coordinate of the point with the largest x value? "
                    "Answer with a single number.", max(xs))
        if qtype == "count_above":
            t = round(float(np.median(ys)), 1)
            count = sum(1 for y in ys if y > t)
            return (f"How many points have a y-value strictly above {t}? "
                    f"Answer with a single integer.", count)
        if qtype == "count_quadrant":
            med_x = round(float(np.median(xs)), 1)
            med_y = round(float(np.median(ys)), 1)
            quadrant = rng.choice(["upper-right", "upper-left",
                                    "lower-right", "lower-left"])
            count = 0
            for x, y in zip(xs, ys):
                if quadrant == "upper-right" and x > med_x and y > med_y:
                    count += 1
                elif quadrant == "upper-left" and x < med_x and y > med_y:
                    count += 1
                elif quadrant == "lower-right" and x > med_x and y < med_y:
                    count += 1
                elif quadrant == "lower-left" and x < med_x and y < med_y:
                    count += 1
            return (f"Using the median x ({med_x}) and median y ({med_y}) as dividers, "
                    f"how many points are in the {quadrant} quadrant?", count)
        if qtype == "outlier":
            cx = sum(xs) / n
            cy = sum(ys) / n
            dists = [(x - cx) ** 2 + (y - cy) ** 2 for x, y in zip(xs, ys)]
            idx = int(np.argmax(dists))
            return (f"Which point is farthest from the centroid of all points? "
                    f"Give its coordinates as (x, y).", f"({xs[idx]}, {ys[idx]})")
        if qtype == "closest_pair":
            min_dist = float("inf")
            for i in range(n):
                for j in range(i + 1, n):
                    d = (xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2
                    if d < min_dist:
                        min_dist = d
            return ("What is the Euclidean distance between the two closest points? "
                    "Round to 2 decimal places.", round(math.sqrt(min_dist), 2))
        return None, None

    def _render(self, rng, xs, ys):
        vs = self._random_style()
        palette = list(vs["palette"])
        rng.shuffle(palette)
        x_label = rng.choice(_X_LABELS)
        y_label = rng.choice(_Y_LABELS)
        title = rng.choice(_TITLE_TEMPLATES).format(x=x_label, y=y_label)
        color = palette[0]
        marker = rng.choice(_MARKER_STYLES)

        fig_w = rng.uniform(6, 8) * vs["figsize_scale"]
        fig_h = rng.uniform(5, 7) * vs["figsize_scale"]
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))

        ax.scatter(xs, ys, c=color, marker=marker,
                   s=rng.choice([30, 40, 50, 60]),
                   alpha=rng.uniform(0.6, 0.9),
                   edgecolors="white", linewidths=0.5)

        ax.set_xlabel(x_label, fontsize=vs["font_size_base"])
        ax.set_ylabel(y_label, fontsize=vs["font_size_base"])
        ax.set_title(title, fontsize=vs["font_size_base"] + 3, pad=10)
        self._apply_style(fig, ax, vs)
        ax.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=vs["dpi"])

if __name__ == "__main__":
    env = ScatterPlotQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: n_distinct={len(gt)}")
