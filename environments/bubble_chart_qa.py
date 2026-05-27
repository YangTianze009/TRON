"""Bubble Chart Visual QA Environment.

Capabilities: V3 (chart extraction), V2 (label reading), R1 (arithmetic)

======================================================================
Level design (parameter["level"], 0-9)
======================================================================

L0: 3 bubbles with VERY different sizes (small/medium/huge), distinct colors.
    Q: "which bubble is the biggest".
L1: 3 bubbles, ask "which is smallest".
L2: 4 bubbles, distinct sizes (1.5x ratios), ask largest/smallest.
L3: 5 bubbles, ask largest. Sizes still 1.5x apart.
L4: 5 bubbles, ask "how many bubbles to the right of x = T".
L5: 6 bubbles, ask "closest to point (a,b)".
L6: 6 bubbles, ask "which quadrant has the most bubbles".
L7: 7 bubbles with subtle (1.2x) size differences.
L8: 8 bubbles, ask "total size of bubbles above y = T".
L9: 8 bubbles with tight clustering, subtle sizes.

parameter = {"level": int in [0, 9]}
"""

import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv
from ._mcq_letter_helper import maybe_mcq_letter_wrap

_THEME_POOLS = ["Sales Performance", "City Statistics", "Product Metrics",
                "Student Scores", "Country Data", "Experiment Results",
                "Quarterly Report", "Market Share Overview"]
_X_LABEL_POOLS = ["Revenue ($K)", "Experience (years)", "Population (M)",
                   "Price ($)", "Temperature (C)", "Distance (km)", "Age",
                   "Score"]
_Y_LABEL_POOLS = ["Profit ($K)", "Satisfaction", "GDP per capita ($K)",
                   "Rating", "Humidity (%)", "Speed (km/h)", "Output", "Index"]
_SIZE_LABEL_POOLS = ["Market Share", "Employees", "Area", "Volume",
                      "Budget", "Users", "Headcount", "Capacity"]
_NAME_POOLS = [
    [f"Item {chr(65 + i)}" for i in range(10)],
    [f"P{i + 1}" for i in range(10)],
    [f"Region {i + 1}" for i in range(10)],
    ["Apollo", "Boreal", "Cygnus", "Draco", "Eridanus", "Fornax",
     "Gemini", "Hydra", "Indus", "Lyra"],
]

class BubbleChartQA(StandaloneVisualEnv):
    ENV_NAME = "bubble_chart"

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        for attempt in range(15):
            try:
                result = self._dispatch(level, attempt)
                if result is not None:
                    self._primary_complexity_feature = level * 5 + len(result[1])
                    return result
            except Exception:
                continue
        return None

    def _sub_rng(self, level: int, attempt: int = 0) -> random.Random:
        return random.Random(
            (self.seed or 0) * 1000 + level * 37 + 991 + attempt * 7919)

    def _level_config(self, level: int) -> Dict:
        # Reordered: size-comparison tasks first (easy), counting/position
        # tasks next, computed tasks last. Fixes L6=0.20 (quadrant) vs
        # L9=0.90 (closest_to=easy) inversion.
        if level == 0:
            return {"n": 3, "size_spread": "huge", "qtype": "largest"}
        if level == 1:
            return {"n": 3, "size_spread": "huge", "qtype": "smallest"}
        if level == 2:
            return {"n": 4, "size_spread": "wide", "qtype": "largest"}
        if level == 3:
            return {"n": 5, "size_spread": "wide", "qtype": "largest"}
        if level == 4:
            return {"n": 5, "size_spread": "wide", "qtype": "count_right_of"}
        if level == 5:
            return {"n": 6, "size_spread": "medium", "qtype": "closest_to"}
        if level == 6:
            return {"n": 7, "size_spread": "narrow", "qtype": "largest"}
        if level == 7:
            return {"n": 8, "size_spread": "narrow", "qtype": "count_right_of"}
        if level == 8:
            return {"n": 8, "size_spread": "narrow", "qtype": "total_size_above"}
        return {"n": 10, "size_spread": "narrow", "qtype": "quadrant_count"}

    def _gen_sizes(self, rng, n, spread):
        if spread == "huge":
            # ratios 1:5:25
            base = [10 + rng.randint(0, 5)]
            base.append(50 + rng.randint(-5, 5))
            base.append(150 + rng.randint(-15, 15))
            base = base[:n]
            while len(base) < n:
                base.append(80 + rng.randint(-10, 10))
            rng.shuffle(base)
            return base
        if spread == "wide":
            # ratio ~1.5x
            sizes = []
            v = rng.randint(20, 30)
            for _ in range(n):
                sizes.append(v + rng.randint(-3, 3))
                v = int(v * 1.55)
            rng.shuffle(sizes)
            return sizes
        if spread == "medium":
            sizes = []
            v = rng.randint(25, 35)
            for _ in range(n):
                sizes.append(v + rng.randint(-3, 3))
                v = int(v * 1.3)
            rng.shuffle(sizes)
            return sizes
        # narrow: ~1.15x
        sizes = []
        v = rng.randint(40, 60)
        for _ in range(n):
            sizes.append(v + rng.randint(-2, 2))
            v = int(v * 1.18)
        rng.shuffle(sizes)
        return sizes

    def _dispatch(self, level: int, attempt: int = 0):
        rng = self._sub_rng(level, attempt)
        cfg = self._level_config(level)
        n = cfg["n"]

        names = list(rng.choice(_NAME_POOLS))[:n]
        sizes = self._gen_sizes(rng, n, cfg["size_spread"])

        # Spread coordinates so bubbles don't overlap too much
        if level <= 2:
            xs = sorted([rng.uniform(15, 85) for _ in range(n)])
            ys = sorted([rng.uniform(15, 85) for _ in range(n)])
            rng.shuffle(ys)
            xs = [round(v, 1) for v in xs]
            ys = [round(v, 1) for v in ys]
        else:
            xs = [round(rng.uniform(10, 90), 1) for _ in range(n)]
            ys = [round(rng.uniform(10, 90), 1) for _ in range(n)]

        question, answer = self._build_qa(rng, cfg["qtype"], names, xs, ys, sizes)
        if question is None:
            return None
        image = self._render(rng, names, xs, ys, sizes)
        # C20 MCQ-letter-only mode (30% wrap when answer is numeric).
        wrapped = maybe_mcq_letter_wrap(rng, question, answer, rate=0.30)
        if wrapped is not None:
            return wrapped[0], wrapped[1], image
        return question, str(answer), image

    def _build_qa(self, rng, qtype, names, xs, ys, sizes):
        n = len(names)
        if qtype == "largest":
            idx = sizes.index(max(sizes))
            stems = [
                "Which bubble is the biggest? Answer with its name only.",
                "Identify the largest bubble in the chart by name.",
                "Which item corresponds to the largest bubble?",
            ]
            return rng.choice(stems), names[idx]
        if qtype == "smallest":
            idx = sizes.index(min(sizes))
            stems = [
                "Which bubble is the smallest? Answer with its name only.",
                "Identify the smallest bubble in the chart by name.",
            ]
            return rng.choice(stems), names[idx]
        if qtype == "count_right_of":
            t = round(rng.choice([30, 40, 50, 60, 70]), 0)
            cnt = sum(1 for v in xs if v > t)
            return (f"How many bubbles have their x-coordinate greater than {int(t)}? "
                    f"Answer with a single integer.", cnt)
        if qtype == "closest_to":
            tx = rng.choice([25.0, 50.0, 75.0])
            ty = rng.choice([25.0, 50.0, 75.0])
            dists = [((xs[i] - tx) ** 2 + (ys[i] - ty) ** 2) ** 0.5
                     for i in range(n)]
            idx = dists.index(min(dists))
            return (f"Which bubble's center is closest to the point ({tx}, {ty})? "
                    f"Answer with the bubble name.", names[idx])
        if qtype == "quadrant_count":
            x_mid = (max(xs) + min(xs)) / 2
            y_mid = (max(ys) + min(ys)) / 2
            quads = {"upper-right": 0, "upper-left": 0,
                     "lower-right": 0, "lower-left": 0}
            for i in range(n):
                key = (("upper" if ys[i] >= y_mid else "lower") + "-" +
                       ("right" if xs[i] >= x_mid else "left"))
                quads[key] += 1
            # Strict argmax: reject ties (dict-order tie-break was biasing
            # the answer toward "upper-right"). If the top count is shared
            # by two or more quadrants, fail this attempt so the retry
            # loop draws fresh coordinates.
            top_count = max(quads.values())
            winners = [k for k, v in quads.items() if v == top_count]
            if len(winners) != 1:
                return None, None
            best = winners[0]
            return ("Divide the chart into four quadrants at the midpoints of the "
                    "x and y ranges. Which quadrant contains the most bubbles? "
                    "Answer 'upper-right', 'upper-left', 'lower-right', or 'lower-left'.",
                    best)
        if qtype == "total_size_above":
            t = round(rng.choice([30.0, 50.0, 70.0]), 0)
            total = sum(sizes[i] for i in range(n) if ys[i] > t)
            return (f"What is the total size value of all bubbles with y-coordinate above "
                    f"{int(t)}? Answer with a single integer.", total)
        return None, None

    def _render(self, rng, names, xs, ys, sizes):
        style = self._random_style()
        x_lab = rng.choice(_X_LABEL_POOLS)
        y_lab = rng.choice(_Y_LABEL_POOLS)
        s_lab = rng.choice(_SIZE_LABEL_POOLS)
        title = rng.choice(_THEME_POOLS)

        fig, ax = plt.subplots(
            figsize=(7 * style["figsize_scale"], 5.5 * style["figsize_scale"]))
        palette = list(style["palette"])
        rng.shuffle(palette)
        n = len(names)
        colors = [palette[i % len(palette)] for i in range(n)]
        ax.scatter(xs, ys, s=[s * 30 for s in sizes],
                   c=colors, alpha=0.65, edgecolors="white", linewidth=0.8)
        for i, name in enumerate(names):
            ax.annotate(name, (xs[i], ys[i]),
                        fontsize=style["font_size_base"] - 2,
                        ha="center", va="bottom",
                        fontfamily=style["font_family"])
        ax.set_xlabel(x_lab, fontsize=style["font_size_base"])
        ax.set_ylabel(y_lab, fontsize=style["font_size_base"])
        ax.set_title(f"{title} (bubble size = {s_lab})",
                     fontsize=style["font_size_base"] + 2)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        self._apply_style(fig, ax, style)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return self.fig_to_pil(fig, dpi=style["dpi"])

if __name__ == "__main__":
    env = BubbleChartQA()
    for level in [0, 3, 6, 9]:
        gt = {}
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                gt[env._answer] = gt.get(env._answer, 0) + 1
        print(f"L{level}: n_distinct={len(gt)}")
