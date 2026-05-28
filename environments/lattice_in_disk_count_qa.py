"""
Lattice in Disk Count QA (D11).

Reference task:
  an external reference (UG float): "Find the number of integer solutions of x²+y² <= 16."
  Ans: 49.

Renders a disk x^2 + y^2 <= R^2 on a coordinate grid with integer lattice
points highlighted. Asks: how many integer (x, y) lie inside the disk
(including boundary).

Verifier: integer answer.
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
    "Count the number of integer solutions (x, y) to x² + y² ≤ {R2}. The disk is shown in the image. Place the integer count in <answer>...</answer>.",
    "How many integer (x, y) pairs satisfy x² + y² ≤ {R2} (the displayed disk)? Place the integer in <answer>...</answer>.",
    "The image shows the disk x² + y² ≤ {R2}. Count the integer lattice points inside (or on its boundary). Place answer in <answer>...</answer>.",
    "Find the number of integer solutions to x² + y² ≤ {R2}. The disk is drawn in the image. Integer in <answer>...</answer>.",
    "How many lattice points (integer coordinates) lie within the disk x² + y² ≤ {R2}? Integer answer in <answer>...</answer>.",
    "Count integer (x, y) inside the displayed disk x² + y² ≤ {R2}. Place the integer in <answer>...</answer>.",
    "Determine how many integer-coordinate points are inside (or on) the disk x² + y² ≤ {R2}. Integer in <answer>...</answer>.",
    "The image shows a disk of squared radius {R2}. Count the integer (x, y) inside. Place the integer in <answer>...</answer>.",
]


class LatticeInDiskCountQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "lattice_in_disk_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 2:
            return {"R_choices": [2, 3]}
        if level <= 4:
            return {"R_choices": [3, 4, 5]}
        if level <= 6:
            return {"R_choices": [4, 5, 6, 7]}
        if level <= 8:
            return {"R_choices": [6, 7, 8]}
        # 2026-05-04: bumped L9 difficulty — bigger radii (more points to count).
        return {"R_choices": [10, 11, 12, 13]}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 5419 + level * 71 + 89)

        R = rng.choice(cfg["R_choices"])
        R2 = R * R
        # Count lattice points
        count = 0
        for x in range(-R, R + 1):
            for y in range(-R, R + 1):
                if x * x + y * y <= R2:
                    count += 1

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(R2=R2)
        answer = str(count)
        img = self._render(R)
        return question, answer, img

    def _render(self, R) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        # Disk
        circle = plt.Circle((0, 0), R, facecolor="#cce5ff",
                            edgecolor="#003566", linewidth=2, alpha=0.4)
        ax.add_patch(circle)
        # Lattice points — uniform color so model must count, not just spot
        # red dots. Showing inside vs outside in different colors leaks the
        # answer (just count the red ones). Aligned with a math benchmark D213 which
        # is largely a math task with the disk shown for orientation.
        pad = max(2, R // 2)
        for x in range(-R - pad, R + pad + 1):
            for y in range(-R - pad, R + pad + 1):
                ax.plot(x, y, "o", markersize=3.5, color="#444444")
        ax.axhline(0, color="#888", linewidth=0.6)
        ax.axvline(0, color="#888", linewidth=0.6)
        pad = max(2, R // 2)
        ax.set_xlim(-R - pad, R + pad)
        ax.set_ylim(-R - pad, R + pad)
        ax.grid(True, alpha=0.2, linestyle=":")
        ax.set_xticks(range(-R - pad, R + pad + 1))
        ax.set_yticks(range(-R - pad, R + pad + 1))
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(f"Disk x² + y² ≤ {R*R}", fontsize=13, pad=6)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = LatticeInDiskCountQA()
    pass_count = 0
    total = 0
    for L in (0, 3, 6):
        for s in (1, 7, 42):
            ok = env.generate(seed=s, parameter={"level": L})
            print(f"L{L} s{s}: ok={ok} A={env._answer if ok else 'X'}")
            if ok:
                wrappers = [
                    f"<answer>{env._answer}</answer>",
                    f"\\boxed{{{env._answer}}}",
                    f"Final answer: {env._answer}",
                ]
                v_pos = env.verify(wrappers[(s or 0) % 3])
                v_neg = env.verify("definitely_wrong_xyz")
                print(f"   positive={v_pos['accuracy']} negative={v_neg['accuracy']}")
                if v_pos['accuracy'] == 1 and v_neg['accuracy'] == 0:
                    pass_count += 1
            total += 1
    print(f"\nPASS: {pass_count}/{total}")
