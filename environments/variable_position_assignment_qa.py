"""
Variable Position Assignment QA (v4 G11b).

Targets: dynamic-math puzzle_test -5.88 (idx=286-style: model gets right multiset
of numbers but wrong position ordering).

Task: figure with labeled positions A, B, C plus algebraic constraints that
uniquely determine each position's value. Ask for the value at each position
**in order** as an ordered tuple.

Reward: ordered tuple exact equality.

Level axes:
  A) Number of positions: 2 at L0, 3 at L3-5, 4 at L6+
  B) Constraint complexity: sum-only at L0, sum+diff at L3, sum+product at L6+
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from .standalone_base import StandaloneVisualEnv

_TEMPLATES = [
    "The figure shows labeled positions {pos_labels} in a triangle/line arrangement. Given the constraints: {constraints}, find the value at each position. Output as ordered tuple in position-label order: {example}. Put in <answer>...</answer>.",
    "Labeled positions {pos_labels} must satisfy {constraints}. Find each value; output as ordered tuple in order of position labels. Put in <answer>...</answer>.",
    "Solve for the values at {pos_labels}. Constraints: {constraints}. Output in order: {example}. Put in <answer>...</answer>.",
    "Given the constraints {constraints}, assign values to positions {pos_labels}. Format as ordered tuple in given position order. Put in <answer>...</answer>.",
    "Positions {pos_labels} with constraints {constraints}. Output as tuple in order shown. Put in <answer>...</answer>.",
    "Find position values for {pos_labels} with constraints {constraints}. Ordered tuple: {example}. Put in <answer>...</answer>.",
    "Solve: positions {pos_labels}, constraints {constraints}. Output values in position-label order: {example}. Put in <answer>...</answer>.",
    "Given positions {pos_labels} and constraints {constraints}, output the ordered tuple of values (in position order). Put in <answer>...</answer>.",
    "Positions {pos_labels} satisfy {constraints}. Find values in order. Put tuple in <answer>...</answer>.",
    "Find values at {pos_labels}: {constraints}. Output as ordered tuple. Put in <answer>...</answer>.",
    "The constraints {constraints} uniquely determine values at {pos_labels}. Output tuple in order. Put in <answer>...</answer>.",
    "Given {constraints}, assign values to {pos_labels}. Ordered tuple (e.g., '{example}'). Put in <answer>...</answer>.",
    "Determine values at positions {pos_labels} from constraints {constraints}. Ordered tuple in <answer>...</answer>.",
    "Constraints: {constraints}. Find values at {pos_labels}. Output in position order. Put in <answer>...</answer>.",
    "Positions {pos_labels} with constraints {constraints}. Ordered-tuple answer. Put in <answer>...</answer>.",
    "Assign values to positions {pos_labels} using constraints {constraints}. Output tuple in order. Put in <answer>...</answer>.",
]

class VariablePositionAssignmentQA(StandaloneVisualEnv):
    ENV_NAME = "variable_position_assignment"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n_pos = 2 + (level + 1) // 3  # 2..5
        return {"n_pos": min(4, n_pos)}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        level = max(0, min(level, 9))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1000 + level * 37 + 163)
        self._primary_complexity_feature = level

        n = cfg["n_pos"]
        pos_labels = ["A", "B", "C", "D"][:n]
        # random values
        values = [rng.randint(5, 30) for _ in range(n)]
        # Build constraints: sum of all, differences between consecutive
        constraints = []
        constraints.append(f"sum of values = {sum(values)}")
        if n >= 2:
            constraints.append(f"{pos_labels[0]} - {pos_labels[1]} = {values[0] - values[1]}")
        if n >= 3:
            constraints.append(f"{pos_labels[1]} + {pos_labels[2]} = {values[1] + values[2]}")
        if n >= 4:
            # BUGFIX 2026-04-24: the old 4th constraint `2C = 2D + k` is linearly
            # dependent on the previous 3 (gives only C-D, but the first 3
            # already determine C-D; system is rank 3 → infinite solutions).
            # Replace with D - A = k (rank 4 verified). Note: review agent
            # suggested "A+D=k or D-A=k"; A+D is dependent (A+D = sum - (B+C)
            # directly), so using D-A=k which is independent.
            constraints.append(f"{pos_labels[3]} - {pos_labels[0]} = {values[3] - values[0]}")
        constraints_str = "; ".join(constraints)

        example = "(" + ", ".join(str(rng.randint(1, 50)) for _ in range(n)) + ")"
        answer = "(" + ", ".join(str(v) for v in values) + ")"

        sidx = (self.seed or 0) % 16
        q = _TEMPLATES[sidx].format(
            pos_labels=", ".join(pos_labels),
            constraints=constraints_str,
            example=example,
        )

        img = self._render(pos_labels, rng)
        return q, answer, img

    def _render(self, pos_labels, rng):
        fig, ax = plt.subplots(figsize=(5, 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        ax.set_aspect("equal")
        ax.axis("off")
        # Place positions on a circle
        import math
        n = len(pos_labels)
        for i, lab in enumerate(pos_labels):
            theta = 2 * math.pi * i / n + math.pi / 2
            x = 3 + 2 * math.cos(theta)
            y = 3 + 2 * math.sin(theta)
            ax.scatter(x, y, s=600, color="#e67e22", zorder=5,
                       edgecolors="black", linewidths=1.5)
            ax.text(x, y, lab, fontsize=18, ha="center", va="center",
                    color="white", fontweight="bold", zorder=6)
            ax.text(x, y - 0.7, "?", fontsize=14, ha="center", va="center",
                    color="darkgreen", fontweight="bold", zorder=6)
        return self.fig_to_pil(fig)

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        pred = predicted.strip().lower().replace(" ", "").rstrip(".")
        gt = ground_truth.strip().lower().replace(" ", "").rstrip(".")
        return pred == gt

if __name__ == "__main__":
    import os
    out_dir = "/tmp/env_check_vpa"
    os.makedirs(out_dir, exist_ok=True)
    env = VariablePositionAssignmentQA()
    for level in (0, 3, 6, 9):
        for seed in range(3):
            s = seed * 1000 + level * 37 + 73
            ok = env.generate(seed=s, parameter={"level": level})
            if not ok:
                print(f"[vpa L{level} s{s}] FAILED")
                continue
            env.render().save(f"{out_dir}/vpa_s{s}_L{level}.png")
            print(f"[vpa L{level} s{s}] A={env._answer}")
