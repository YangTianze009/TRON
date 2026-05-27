"""
Projectile Compare QA (D92, P2).

Reference task:
  qid 275 (HS MCQ): "Ball 1 and ball 2 follow the paths shown, where the
   darkblue path is Ball 1 and the green path is Ball 2. Which ball is in
   the air for a longer time? Assume that you can ignore air resistance for
   this problem. choice: (A) Ball 1 (B) Ball 2 (C) The amount time of balls
   in air is same." Ans: B.

Renders two parabolic projectile paths in different colors. The model picks
the one with longer time aloft. The longer-aloft trajectory is the one with
greater max height (since time aloft = 2 * v_y / g, max height = v_y^2 / 2g
=> max_height ~ time_aloft^2 / 4g).

Verifier: single MCQ letter A, B, or C.
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


# 2026-05-04 R3: added concise hint — time aloft depends on max HEIGHT,
# not horizontal range. v5 mean 0.42 (L0=0.4) suggests model often confuses
# the two; one-sentence hint per template (kept short to respect token budget).
_TEMPLATES = [
    "Ball 1 (dark blue) and Ball 2 (green) follow the projectile paths shown. Ignore air resistance. Which ball is in the air for a longer time? Hint: time aloft depends on peak height, not horizontal range. Choices: (A) Ball 1 (B) Ball 2 (C) Same time. Place letter in <answer>...</answer>.",
    "Two projectiles follow the parabolic paths shown in the image. Which is in the air longer? (Time aloft is set by peak height, not range.) Choices: (A) Ball 1 (B) Ball 2 (C) Same. Place letter in <answer>...</answer>.",
    "The image shows two projectile paths (dark blue = Ball 1, green = Ball 2). Which ball spends more time in the air? Recall: longer time aloft ↔ greater peak height. Choices: (A) Ball 1 (B) Ball 2 (C) Same. Letter in <answer>...</answer>.",
    "Compare the time aloft for the two projectiles in the figure. Which is greater? (Hint: compare peak heights, not horizontal distances.) Choices: (A) Ball 1 (B) Ball 2 (C) Same. Place letter in <answer>...</answer>.",
    "The two trajectories show projectile motion. Which spends a longer time in the air? Time aloft scales with peak height (range is irrelevant). Choices: (A) Ball 1 (B) Ball 2 (C) Same. Letter in <answer>...</answer>.",
    "Looking at the projectile paths, which ball stays in the air longer? Higher peak ⇒ longer time aloft. Choices: (A) Ball 1 (B) Ball 2 (C) Same. Place letter in <answer>...</answer>.",
    "Determine which projectile is in the air longer based on the displayed parabolic paths. Hint: compare peak heights. Choices: (A) Ball 1 (B) Ball 2 (C) Same. Letter in <answer>...</answer>.",
    "Two balls were launched. Their paths are shown. Which ball lands later? (Time aloft depends on peak height.) Choices: (A) Ball 1 (B) Ball 2 (C) Same. Place letter in <answer>...</answer>.",
]


class ProjectileCompareQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "projectile_compare"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # 2026-05-04 R3 retry: hint alone insufficient (v5=0.42, L0=0.4).
        # Widen height-diff factors at every tier so L0 trivially-large gap.
        if level <= 3:
            return {"min_diff_factor": 0.7}  # 70% difference (was 0.5)
        if level <= 6:
            return {"min_diff_factor": 0.4}  # was 0.25
        return {"min_diff_factor": 0.22}  # was 0.15

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 7237 + level * 71 + 113)

        # Allow ground-truth A, B, or C (same).
        gt_choice = rng.choices(["A", "B", "C"], weights=[4, 4, 1])[0]
        # Heights (in arbitrary units) determine time aloft.
        h1 = rng.uniform(2.0, 5.0)
        if gt_choice == "C":
            h2 = h1
        elif gt_choice == "A":
            h2 = h1 * (1 - cfg["min_diff_factor"] - rng.uniform(0.0, 0.2))
            h2 = max(0.5, h2)
        else:  # B
            h2 = h1 * (1 + cfg["min_diff_factor"] + rng.uniform(0.0, 0.2))

        # Horizontal ranges (independent of time aloft, since v_x can vary)
        r1 = rng.uniform(3.0, 6.0)
        r2 = rng.uniform(3.0, 6.0)

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx]
        answer = gt_choice
        img = self._render(h1, r1, h2, r2)
        return question, answer, img

    def _render(self, h1, r1, h2, r2) -> Image.Image:
        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Ground
        x_max = max(r1, r2) + 1
        y_max = max(h1, h2) + 1
        ax.plot([-0.2, x_max + 0.2], [0, 0], color="#000", linewidth=1.5)
        # Ball 1: parabolic path from (0,0) with peak at (r1/2, h1) landing at (r1, 0)
        xs1 = np.linspace(0, r1, 100)
        ys1 = -4 * h1 / (r1 * r1) * (xs1 - r1 / 2) ** 2 + h1
        ax.plot(xs1, ys1, color="#1d4ed8", linewidth=3, label="Ball 1")
        # Ball 2
        xs2 = np.linspace(0, r2, 100)
        ys2 = -4 * h2 / (r2 * r2) * (xs2 - r2 / 2) ** 2 + h2
        ax.plot(xs2, ys2, color="#15803d", linewidth=3, label="Ball 2")
        # Mark start and end with markers
        ax.plot(0, 0, "o", color="#1d4ed8", markersize=10)
        ax.plot(r1, 0, "x", color="#1d4ed8", markersize=10)
        ax.plot(0, 0, "o", color="#15803d", markersize=8)
        ax.plot(r2, 0, "x", color="#15803d", markersize=10)
        ax.legend(loc="upper right", fontsize=12)
        ax.set_xlim(-0.5, x_max)
        ax.set_ylim(-0.5, y_max)
        ax.set_aspect("equal")
        ax.set_xlabel("Horizontal distance", fontsize=12)
        ax.set_ylabel("Height", fontsize=12)
        ax.set_title("Projectile paths (no air resistance)",
                     fontsize=13, pad=8)
        ax.grid(True, alpha=0.2, linestyle=":")
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = ProjectileCompareQA()
    for level in [0, 3, 6, 9]:
        ans = []
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                ans.append(env._answer)
        from collections import Counter
        print(f"L{level}: {Counter(ans)}")
