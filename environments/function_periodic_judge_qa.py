"""
Function Periodic Judgment QA (D14-add).

Reference task:
  an external reference (HS MCQ): "Is this a periodic function? Choices: (A) Yes (B) No."
  Ans: A.

Renders a function curve on a plot. Asks Yes/No whether it is periodic.
The function family is sampled per seed. Periodic candidates: sin/cos/sawtooth.
Non-periodic candidates: linear, polynomial, exponential, log.

Verifier: Yes/No (also accepts A/B as MCQ letter equivalents).
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
    "Is the function shown in the plot periodic? Answer Yes or No. Place answer in <answer>...</answer>.",
    "Examine the curve in the image. Is this a periodic function? Answer Yes or No in <answer>...</answer>.",
    "Determine whether the displayed function is periodic. Answer Yes or No. Place in <answer>...</answer>.",
    "Looking at the graph, is this function periodic? Answer Yes or No in <answer>...</answer>.",
    "Decide if the function in the plot is periodic. Answer Yes or No in <answer>...</answer>.",
    "Is the function f(x) shown above periodic? Answer Yes or No in <answer>...</answer>.",
    "Is the curve depicted in the image a periodic function? Answer Yes or No. Place in <answer>...</answer>.",
    "Is this a periodic function? Answer Yes or No in <answer>...</answer>.",
]


class FunctionPeriodicJudgeQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "function_periodic_judge"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # Levels mostly affect distractor choice and noise added to plots
        if level <= 3:
            return {"complexity": "low"}
        if level <= 6:
            return {"complexity": "med"}
        return {"complexity": "high"}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 4517 + level * 71 + 41)

        is_periodic = rng.random() < 0.5
        if is_periodic:
            choices = ["sin", "cos", "sawtooth", "abs_sin", "triangle"]
        else:
            choices = ["linear", "quadratic", "cubic", "exp", "log"]
        family = rng.choice(choices)
        a = rng.choice([1, 2, 3])
        b = rng.choice([0.5, 1.0, 1.5, 2.0])
        x_min, x_max = -8, 8
        xs = np.linspace(x_min, x_max, 600)
        if family == "sin":
            ys = a * np.sin(b * xs)
            label = f"y = {a}·sin({b:g}x)"
        elif family == "cos":
            ys = a * np.cos(b * xs)
            label = f"y = {a}·cos({b:g}x)"
        elif family == "sawtooth":
            period = 2 * math.pi / max(b, 0.5)
            ys = a * (((xs / period) - np.floor(xs / period + 0.5)) * 2)
            label = "sawtooth"
        elif family == "abs_sin":
            ys = a * np.abs(np.sin(b * xs))
            label = f"y = {a}|sin({b:g}x)|"
        elif family == "triangle":
            period = 2 * math.pi / max(b, 0.5)
            phase = (xs / period) - np.floor(xs / period)
            ys = a * (2 * np.abs(2 * (phase - 0.5)) - 1)
            label = "triangle wave"
        elif family == "linear":
            slope = rng.choice([-2, -1, 1, 2])
            ys = slope * xs + rng.choice([-1, 0, 1])
            label = f"y = {slope}x"
        elif family == "quadratic":
            ys = (xs - rng.choice([-2, 0, 2])) ** 2
            label = "y = x²"
        elif family == "cubic":
            ys = 0.05 * (xs ** 3) + rng.choice([-1, 0, 1]) * xs
            label = "y = x³ (scaled)"
        elif family == "exp":
            ys = np.exp(0.3 * xs)
            label = "y = e^x"
        elif family == "log":
            xs2 = np.linspace(0.1, 10, 600)
            xs = xs2
            ys = np.log(xs2)
            x_min, x_max = 0, 10
            label = "y = log(x)"
        else:
            return None

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        answer = "Yes" if is_periodic else "No"
        img = self._render(xs, ys, label, x_min, x_max)
        return question, answer, img

    def _render(self, xs, ys, label, x_min, x_max) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.plot(xs, ys, color="#003566", linewidth=2)
        ax.axhline(0, color="#888", linewidth=0.8)
        ax.axvline(0, color="#888", linewidth=0.8)
        ax.grid(True, alpha=0.25, linestyle="--")
        ax.set_xlim(x_min, x_max)
        # Set y limits for a tight view (clip extreme values)
        finite_ys = ys[np.isfinite(ys)]
        if len(finite_ys) > 0:
            y_lo = float(np.percentile(finite_ys, 1))
            y_hi = float(np.percentile(finite_ys, 99))
            pad = 0.5 * max(0.5, y_hi - y_lo)
            ax.set_ylim(y_lo - pad, y_hi + pad)
        ax.set_xlabel("x", fontsize=13)
        ax.set_ylabel("y", fontsize=13)
        ax.set_title("Function plot", fontsize=14, pad=8)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = FunctionPeriodicJudgeQA()
    for level in [0, 3, 6, 9]:
        ans = []
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                ans.append(env._answer)
        print(f"L{level}: yes={ans.count('Yes')} no={ans.count('No')}")
