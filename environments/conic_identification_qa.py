"""
Conic Identification QA (v4 G7b, for Analytic).

Task: given an equation like x² + y² = 25, or plot, identify the conic
type and key parameter (center, vertex, radius).

Reward: MCQ letter match + parameter verification.

Level axes:
  A) Conic types: circle only at L0-3, ellipse at L4-5, hyperbola at L6+
  B) Target: type at L0, center at L3, semi-axes at L6+
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "The figure plots the conic section {eq}. Identify its type. A. circle  B. ellipse  C. hyperbola  D. parabola. Put the letter in <answer>...</answer>.",
    "What type of conic is {eq}? A-D. Put letter in <answer>...</answer>.",
    "Identify the conic: {eq}. A. circle, B. ellipse, C. hyperbola, D. parabola. Put letter in <answer>...</answer>.",
    "Classify the curve {eq}. A-D. Put letter in <answer>...</answer>.",
    "Which conic is {eq}? A-D. Put letter in <answer>...</answer>.",
    "Given {eq}, name the conic type. A-D. Put letter in <answer>...</answer>.",
    "Type of conic for {eq}? A-D. Put letter in <answer>...</answer>.",
    "Classify {eq}. A-D. Put letter in <answer>...</answer>.",
    "Name the conic: {eq}. A-D. Put letter in <answer>...</answer>.",
    "Identify conic {eq}. A-D. Put letter in <answer>...</answer>.",
    "Conic {eq} is which? A-D. Put letter in <answer>...</answer>.",
    "What conic is this: {eq}? A-D. Put letter in <answer>...</answer>.",
    "{eq} is what type of conic? A-D. Put letter in <answer>...</answer>.",
    "Identify {eq}'s conic class. A-D. Put letter in <answer>...</answer>.",
    "Type identification: {eq}. A-D. Put letter in <answer>...</answer>.",
    "Classify the curve {eq} (conic section). A-D. Put letter in <answer>...</answer>.",
]

class ConicIdentificationQA(StandaloneVisualEnv):
    ENV_NAME = "conic_identification"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        if level <= 3:
            types = ["circle", "parabola"]
        elif level <= 5:
            types = ["circle", "ellipse", "parabola"]
        else:
            types = ["circle", "ellipse", "hyperbola", "parabola"]
        return {"types": types}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 181)
        self._primary_complexity_feature = level

        conic = rng.choice(cfg["types"])
        letter_map = {"circle": "A", "ellipse": "B", "hyperbola": "C",
                      "parabola": "D"}
        letter = letter_map[conic]

        if conic == "circle":
            r = rng.randint(2, 6)
            eq = f"x² + y² = {r*r}"
            plot_fn = "circle"
            plot_params = (r,)
        elif conic == "ellipse":
            a = rng.randint(3, 6)
            b = rng.randint(2, min(a, 5))
            if a == b:
                b -= 1
            eq = f"x²/{a*a} + y²/{b*b} = 1"
            plot_fn = "ellipse"
            plot_params = (a, b)
        elif conic == "hyperbola":
            a = rng.randint(2, 5)
            b = rng.randint(2, 5)
            eq = f"x²/{a*a} - y²/{b*b} = 1"
            plot_fn = "hyperbola"
            plot_params = (a, b)
        else:  # parabola
            p = rng.randint(1, 4)
            eq = f"y = {p}x²"
            plot_fn = "parabola"
            plot_params = (p,)

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(eq=eq)
        if level <= 2:
            q += (
                " Hint: identify the conic by its standard form. "
                "x²+y²=r² → CIRCLE (closed loop, equal x and y radius). "
                "x²/a²+y²/b²=1 → ELLIPSE (closed loop, oval). "
                "x²/a²-y²/b²=1 → HYPERBOLA (two open branches). "
                "y=ax² → PARABOLA (single open curve, U-shaped)."
            )

        img = self._render(plot_fn, plot_params, rng)
        return q, letter, img

    def _render(self, plot_fn, params, rng):
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(-8, 8); ax.set_ylim(-8, 8)
        ax.set_aspect("equal")
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.grid(True, linestyle=":", alpha=0.3)

        if plot_fn == "circle":
            r = params[0]
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(r * np.cos(theta), r * np.sin(theta), color="blue", lw=2)
        elif plot_fn == "ellipse":
            a, b = params
            theta = np.linspace(0, 2 * np.pi, 200)
            ax.plot(a * np.cos(theta), b * np.sin(theta), color="blue", lw=2)
        elif plot_fn == "hyperbola":
            a, b = params
            t = np.linspace(-2, 2, 200)
            ax.plot(a * np.cosh(t), b * np.sinh(t), color="blue", lw=2)
            ax.plot(-a * np.cosh(t), b * np.sinh(t), color="blue", lw=2)
        else:  # parabola
            p = params[0]
            x = np.linspace(-3, 3, 200)
            y = p * x * x
            ax.plot(x, y, color="blue", lw=2)

        return self.fig_to_pil(fig)

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_ci"
    os.makedirs(out_dir, exist_ok=True)
    env = ConicIdentificationQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 127
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[ci L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/ci_s{s}_L{level}.png")
            print(f"[ci L{level} s{s}] A={env._answer}")
