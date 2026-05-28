"""
Differentiable At Point Judgment QA (D19-add).

Reference task:
  an external reference (UG MCQ): "Is the function differentiable at x = 0? choice: (A) Yes
   (B) No." Ans: A.

Renders a function curve on a plot with a marked vertical line at x = c.
Asks Yes/No whether the function is differentiable at that point.

Differentiable candidates: smooth analytical curves through c.
Non-differentiable candidates: continuous-but-corner functions (|x-c|),
piecewise functions with slope discontinuity, cusps.

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
    "Is the function differentiable at x = {c}? Answer Yes or No. Place answer in <answer>...</answer>.",
    "Examine the curve. Is the function differentiable at x = {c}? Answer Yes or No in <answer>...</answer>.",
    "Determine whether the function shown is differentiable at x = {c}. Answer Yes or No in <answer>...</answer>.",
    "Looking at the plot, is the function differentiable at x = {c}? Answer Yes or No in <answer>...</answer>.",
    "Decide whether the displayed function is differentiable at x = {c}. Answer Yes or No in <answer>...</answer>.",
    "Is f(x) shown above differentiable at x = {c}? Answer Yes or No in <answer>...</answer>.",
    "Does the curve show f to be differentiable at x = {c}? Answer Yes or No in <answer>...</answer>.",
    "At x = {c}, is the function differentiable? Answer Yes or No in <answer>...</answer>.",
]


class DifferentiableAtPointQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "differentiable_at_point"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        return {"level": level}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        rng = random.Random((self.seed or 0) * 4421 + level * 71 + 67)

        c = rng.choice([-2, -1, 0, 1, 2])
        x_min, x_max = c - 4, c + 4
        is_diff = rng.random() < 0.5

        xs = np.linspace(x_min, x_max, 600)

        if is_diff:
            family = rng.choice(["sin_smooth", "polynomial", "linear",
                                 "exp_offset"])
            if family == "sin_smooth":
                a = rng.choice([1, 2])
                b = rng.choice([0.5, 1.0, 1.5])
                ys = a * np.sin(b * (xs - c))
            elif family == "polynomial":
                ys = (xs - c) ** 2 + rng.choice([0, 1, -1])
            elif family == "linear":
                m = rng.choice([-2, -1, 1, 2])
                b_int = rng.choice([-1, 0, 1])
                ys = m * (xs - c) + b_int
            else:  # exp_offset
                ys = np.exp(0.3 * (xs - c)) + rng.choice([-1, 0])
        else:
            family = rng.choice(["abs_corner", "piecewise_kink", "cusp"])
            if family == "abs_corner":
                offset = rng.choice([0, 1, 2])
                ys = np.abs(xs - c) + offset
            elif family == "piecewise_kink":
                m1 = rng.choice([-2, -1])
                m2 = rng.choice([1, 2])
                base = rng.choice([0, 1])
                ys = np.where(xs <= c, m1 * (xs - c) + base, m2 * (xs - c) + base)
            else:  # cusp
                # f(x) = (x-c)^(2/3)  has a cusp at c
                ys = np.cbrt(np.abs(xs - c)) ** 2 * np.sign(xs - c) ** 2

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(c=c)
        answer = "Yes" if is_diff else "No"
        img = self._render(xs, ys, c, x_min, x_max)
        return question, answer, img

    def _render(self, xs, ys, c, x_min, x_max) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.plot(xs, ys, color="#003566", linewidth=2)
        ax.axhline(0, color="#888", linewidth=0.8)
        ax.axvline(0, color="#888", linewidth=0.8)
        ax.axvline(c, color="#b00020", linewidth=1.2, linestyle="--")
        ax.text(c + 0.1,
                ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 1,
                f"x = {c}", color="#b00020", fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel("x", fontsize=13)
        ax.set_ylabel("y", fontsize=13)
        ax.set_title("Function f(x)", fontsize=14, pad=8)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = DifferentiableAtPointQA()
    for level in [0, 3, 6, 9]:
        ans = []
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                ans.append(env._answer)
        print(f"L{level}: yes={ans.count('Yes')} no={ans.count('No')}")
