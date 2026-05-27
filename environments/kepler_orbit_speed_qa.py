"""
Kepler Orbit Speed QA (D93, P2).

Reference task:
  qid 279 (HS MCQ): "Two planets travel in an elliptical orbit about the
   sun as shown. Which planet have greater orbital speed? choice: (A) Blue
   planet (B) Orange planet (C) Both two planet have the same speed."
   Ans: B.

Renders an elliptical orbit (sun at one focus) with two planets shown at
different positions (closer to or further from sun). By Kepler's 2nd law,
the planet closer to the sun is faster.

Verifier: single MCQ letter A, B, or C.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Two planets travel in an elliptical orbit about the sun as shown. Which planet has greater orbital speed? Choices: (A) Blue planet (B) Orange planet (C) Same speed. Place letter in <answer>...</answer>.",
    "By Kepler's second law, which planet (blue or orange) is moving faster in the orbit shown? Choices: (A) Blue (B) Orange (C) Same. Letter in <answer>...</answer>.",
    "The image shows two planets in an elliptical orbit with the sun at the focus. Which has greater orbital speed? Choices: (A) Blue (B) Orange (C) Same. Place letter in <answer>...</answer>.",
    "Compare the orbital speeds of the two planets in the elliptical orbit. Which is faster? Choices: (A) Blue (B) Orange (C) Same. Letter in <answer>...</answer>.",
    "Determine which planet is moving faster in the depicted elliptical orbit. Choices: (A) Blue (B) Orange (C) Same. Place letter in <answer>...</answer>.",
    "Looking at the orbit and the two planet positions, which planet moves faster? Choices: (A) Blue (B) Orange (C) Same. Place letter in <answer>...</answer>.",
    "Which of the two planets has greater orbital speed at the depicted positions? Choices: (A) Blue (B) Orange (C) Same. Letter in <answer>...</answer>.",
    "Two planets are shown in an elliptical orbit. Which has greater orbital speed? Choices: (A) Blue (B) Orange (C) Same. Place letter in <answer>...</answer>.",
]


class KeplerOrbitSpeedQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "kepler_orbit_speed"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # Levels: at high, distance ratio is closer to 1 (harder)
        if level <= 3:
            return {"min_ratio": 2.0}
        if level <= 6:
            return {"min_ratio": 1.5}
        return {"min_ratio": 1.2}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 7331 + level * 71 + 127)

        # Ellipse parameters (sun at focus)
        a = 5.0  # semi-major
        b_ratio = rng.uniform(0.5, 0.85)
        b = a * b_ratio
        c = math.sqrt(a * a - b * b)  # focal distance

        # Pick two angles theta1, theta2 around the ellipse such that
        # planet positions have distinct distances from sun (focus at (-c, 0))
        # If GT = "Same", put both at quadrature (same distance to sun).
        gt = rng.choices(["A", "B", "C"], weights=[5, 5, 1])[0]
        x1 = y1 = x2 = y2 = None
        for _ in range(200):
            t1 = rng.uniform(0, 2 * math.pi)
            t2 = rng.uniform(0, 2 * math.pi)
            x1 = a * math.cos(t1)
            y1 = b * math.sin(t1)
            x2 = a * math.cos(t2)
            y2 = b * math.sin(t2)
            d1 = math.hypot(x1 - (-c), y1)
            d2 = math.hypot(x2 - (-c), y2)
            if gt == "C":
                if abs(d1 - d2) < 0.3 and abs(t1 - t2) > 1.0:
                    break
            elif gt == "A":  # Blue (planet 1) faster -> closer to sun
                if d2 / max(d1, 1e-9) >= cfg["min_ratio"]:
                    break
            else:  # B: Orange (planet 2) faster -> closer
                if d1 / max(d2, 1e-9) >= cfg["min_ratio"]:
                    break
        else:
            # Fallback: place planets at perihelion/aphelion to satisfy gt
            # Perihelion (closest to focus at (-c,0)) is at (-a, 0); aphelion at (+a, 0)
            if gt == "A":
                # blue at perihelion (closest), orange at aphelion
                x1, y1 = -a, 0
                x2, y2 = a, 0
            elif gt == "B":
                x1, y1 = a, 0
                x2, y2 = -a, 0
            else:
                # both at quadrature (same distance from focus)
                x1, y1 = 0, b
                x2, y2 = 0, -b

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        answer = gt
        img = self._render(a, b, c, x1, y1, x2, y2)
        return question, answer, img

    def _render(self, a, b, c, x1, y1, x2, y2) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Draw ellipse
        ts = np.linspace(0, 2 * math.pi, 200)
        xs = a * np.cos(ts)
        ys = b * np.sin(ts)
        ax.plot(xs, ys, color="#1a1a1a", linewidth=1.8)
        # Sun at focus (-c, 0)
        ax.plot(-c, 0, "o", color="#fbbf24", markersize=20,
                markeredgecolor="#b8860b", markeredgewidth=1.5,
                label="Sun")
        # Planets
        ax.plot(x1, y1, "o", color="#1d4ed8", markersize=14,
                label="Blue planet")
        ax.plot(x2, y2, "o", color="#ea7c1c", markersize=14,
                label="Orange planet")
        # Faint dashed line from sun to each planet
        ax.plot([-c, x1], [0, y1], color="#888", linewidth=1, linestyle=":")
        ax.plot([-c, x2], [0, y2], color="#888", linewidth=1, linestyle=":")

        ax.set_xlim(-a - 1, a + 1)
        ax.set_ylim(-b - 1, b + 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.legend(loc="upper right", fontsize=11)
        ax.set_title("Elliptical orbit (sun at focus)",
                     fontsize=13, pad=8)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = KeplerOrbitSpeedQA()
    for level in [0, 3, 6, 9]:
        ans = []
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                ans.append(env._answer)
        from collections import Counter
        print(f"L{level}: {Counter(ans)}")
