"""
Cyclic Group Fill QA (D10).

Reference task:
  qid 82 (UG float): "There is a cycle graph of C6 in the image. Fill the
   number in the empty slot so that it becomes the integers modulo 6 under
   addition (Z6)." Ans: 4.

Renders a directed cycle graph C_n with n vertices labeled 0..n-1 in order
around the circle, but one vertex's label is replaced with '?'. Asks for
the integer that fills the slot to make the cycle consistent with
Z/n under +.

Verifier: integer answer.
"""
import math
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "The image shows a cycle graph C_{n} representing the integers modulo {n} under addition (Z_{n}). One slot is marked '?'. Fill in the missing integer so that the labels around the cycle, in order, form 0, 1, 2, ..., {nm1}. Place the integer in <answer>...</answer>.",
    "A C_{n} cycle graph is shown in the image. The labels around the cycle go in order 0, 1, ..., {nm1} (forming Z_{n} under +). Find the integer in the slot marked '?'. Place answer in <answer>...</answer>.",
    "The image displays the integers modulo {n} arranged on a cycle (Z_{n} under addition). One label is replaced with '?'. What integer belongs there? Place the integer in <answer>...</answer>.",
    "The cycle in the image represents Z_{n}. Labels around the cycle should be 0, 1, ..., {nm1}. Find the missing label '?'. Integer in <answer>...</answer>.",
    "Look at the cycle graph C_{n}. The labels are the integers 0..{nm1} (Z_{n}). One is hidden as '?'. What number is it? Integer answer in <answer>...</answer>.",
    "Determine the missing integer marked '?' in the C_{n} cycle (representing Z_{n} under addition). Place answer in <answer>...</answer>.",
    "In the cycle graph showing Z_{n} under +, find the integer hidden at '?'. Place the integer in <answer>...</answer>.",
    "Fill the number in the empty slot '?' on the C_{n} cycle so it becomes Z_{n} under addition. Integer in <answer>...</answer>.",
]


class CyclicGroupFillQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "cyclic_group_fill"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        if level <= 2:
            return {"n": 4}
        if level <= 4:
            return {"n": 6}
        if level <= 6:
            return {"n": 7}
        if level <= 8:
            return {"n": 8}
        return {"n": 10}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        n = cfg["n"]
        rng = random.Random((self.seed or 0) * 8807 + level * 71 + 79)

        labels = list(range(n))
        # Sometimes start with a non-zero rotation
        rotation = rng.randint(0, n - 1)
        labels = labels[rotation:] + labels[:rotation]
        # Pick which vertex to hide
        hide_idx = rng.randint(0, n - 1)
        answer_int = labels[hide_idx]

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(n=n, nm1=n - 1)
        answer = str(answer_int)
        img = self._render(n, labels, hide_idx)
        return question, answer, img

    def _render(self, n, labels, hide_idx) -> Image.Image:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.set_aspect("equal")
        ax.axis("off")

        positions = []
        for i in range(n):
            ang = 2 * math.pi * i / n + math.pi / 2
            positions.append((math.cos(ang), math.sin(ang)))

        # Draw cycle edges (directed)
        for i in range(n):
            x1, y1 = positions[i]
            x2, y2 = positions[(i + 1) % n]
            ax.annotate(
                "",
                xy=(x2 * 0.9, y2 * 0.9),
                xytext=(x1 * 0.9, y1 * 0.9),
                arrowprops=dict(arrowstyle="->", color="#1a1a1a", lw=1.5),
            )

        # Draw vertices and labels
        for i, (x, y) in enumerate(positions):
            is_hidden = (i == hide_idx)
            fc = "#fde2e4" if is_hidden else "#dbe9f7"
            ec = "#b00020" if is_hidden else "#1a1a1a"
            circ = plt.Circle((x, y), 0.18, facecolor=fc, edgecolor=ec,
                              linewidth=2, zorder=3)
            ax.add_patch(circ)
            txt = "?" if is_hidden else str(labels[i])
            color = "#b00020" if is_hidden else "#1a1a1a"
            ax.text(x, y, txt, ha="center", va="center",
                    fontsize=15, fontweight="bold", color=color, zorder=5)

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_title(f"Cycle graph C_{n}", fontsize=14, pad=8)
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = CyclicGroupFillQA()
    for level in [0, 3, 6, 9]:
        ans = set()
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                ans.add(env._answer)
        print(f"L{level}: distinct={len(ans)} sample={sorted(ans)[:5]}")
