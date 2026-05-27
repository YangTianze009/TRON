"""
Continuity At Point Judgment QA (D17-add).

Reference task:
  qid 79 (UG MCQ): "Is the function continuous at x = 0? choice: (A) Yes
   (B) No." Ans: A.

Renders a function curve on a plot with a marked vertical line at x = c.
Asks Yes/No whether the function is continuous at that point.

Continuous candidates: smooth analytical curves through c.
Discontinuous candidates: piecewise functions with jump or removable
discontinuity at c.

Verifier: Yes/No.
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


_TEMPLATES = [
    "Is the function continuous at x = {c}? Answer Yes or No. Place answer in <answer>...</answer>.",
    "Examine the curve. Is the function continuous at x = {c}? Answer Yes or No in <answer>...</answer>.",
    "Determine whether the function shown is continuous at x = {c}. Answer Yes or No in <answer>...</answer>.",
    "Looking at the plot, is the function continuous at x = {c}? Answer Yes or No in <answer>...</answer>.",
    "Decide whether the displayed function is continuous at x = {c}. Answer Yes or No in <answer>...</answer>.",
    "Is f(x) shown above continuous at x = {c}? Answer Yes or No in <answer>...</answer>.",
    "Does the curve show f to be continuous at x = {c}? Answer Yes or No in <answer>...</answer>.",
    "At x = {c}, is the function continuous? Answer Yes or No in <answer>...</answer>.",
]


class ContinuityAtPointQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "continuity_at_point"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 3:
            return {"jump_size_min": 1.0, "jump_size_max": 3.0}
        if level <= 6:
            return {"jump_size_min": 0.5, "jump_size_max": 2.0}
        return {"jump_size_min": 0.3, "jump_size_max": 1.0}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4007 + level * 71 + 53)

        c = rng.choice([-2, -1, 0, 1, 2])
        x_min, x_max = c - 4, c + 4
        is_continuous = rng.random() < 0.5

        if is_continuous:
            family = rng.choice(["sin_smooth", "polynomial", "linear",
                                 "abs_offset"])
            if family == "sin_smooth":
                a = rng.choice([1, 2])
                b = rng.choice([0.5, 1.0, 1.5])
                xs = np.linspace(x_min, x_max, 600)
                ys = a * np.sin(b * (xs - c))
                # Will pass through y=0 at x=c
            elif family == "polynomial":
                xs = np.linspace(x_min, x_max, 600)
                ys = (xs - c) ** 2 + rng.choice([0, 1, -1])
            elif family == "linear":
                m = rng.choice([-2, -1, 1, 2])
                b_int = rng.choice([-1, 0, 1])
                xs = np.linspace(x_min, x_max, 600)
                ys = m * (xs - c) + b_int
            else:  # abs_offset
                xs = np.linspace(x_min, x_max, 600)
                ys = np.abs(xs - c) + rng.choice([0, 1])
            xs_left, ys_left = xs[xs <= c], ys[xs <= c]
            xs_right, ys_right = xs[xs > c], ys[xs > c]
            point_left, point_right = None, None
        else:
            # Jump discontinuity: f(x) = a for x<c, b for x>=c
            jump = rng.uniform(cfg["jump_size_min"], cfg["jump_size_max"])
            if rng.random() < 0.5:
                jump = -jump
            base = rng.choice([0, 1, 2])
            xs_l = np.linspace(x_min, c, 300)
            xs_r = np.linspace(c, x_max, 300)
            slope = rng.choice([-0.5, 0, 0.5])
            ys_l = slope * (xs_l - c) + base
            ys_r = slope * (xs_r - c) + base + jump
            xs_left, ys_left = xs_l, ys_l
            xs_right, ys_right = xs_r[xs_r > c], ys_r[xs_r > c]
            point_left = (c, ys_l[-1])
            point_right = (c, ys_r[0])
            xs = None
            ys = None

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(c=c)
        answer = "Yes" if is_continuous else "No"
        img = self._render(xs_left, ys_left, xs_right, ys_right,
                           c, point_left, point_right, x_min, x_max)
        return question, answer, img

    def _render(self, xs_l, ys_l, xs_r, ys_r, c,
                point_l, point_r, x_min, x_max) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.plot(xs_l, ys_l, color="#003566", linewidth=2)
        ax.plot(xs_r, ys_r, color="#003566", linewidth=2)
        if point_l is not None:
            # Draw open / filled circles to indicate jump
            ax.plot(*point_l, "o", color="#003566", markersize=7,
                    markerfacecolor="#003566")  # filled
            ax.plot(*point_r, "o", color="#003566", markersize=7,
                    markerfacecolor="#ffffff")  # open
        ax.axhline(0, color="#888", linewidth=0.8)
        ax.axvline(0, color="#888", linewidth=0.8)
        # Mark x = c with a vertical dashed line
        ax.axvline(c, color="#b00020", linewidth=1.2, linestyle="--")
        ax.text(c + 0.1, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1,
                f"x = {c}", color="#b00020", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("x", fontsize=13)
        ax.set_ylabel("y", fontsize=13)
        ax.set_title("Function f(x)", fontsize=14, pad=8)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = ContinuityAtPointQA()
    for level in [0, 3, 6, 9]:
        ans = []
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                ans.append(env._answer)
        print(f"L{level}: yes={ans.count('Yes')} no={ans.count('No')}")
