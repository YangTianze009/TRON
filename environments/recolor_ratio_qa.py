"""
Recolor Ratio QA (D40, P3).

Reference task:
  an external reference (ES float): "How many black squares need to be coloured in white,
   so that there are exactly fourth as many black squares as there are
   white squares?" Ans: 9.

Renders a grid of B black + W white squares. Asks: how many black squares
need to be recolored white so that black:white = 1:k (e.g., 1:4).

Verifier: integer answer.
"""
import random
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "How many black squares need to be coloured white so that there are exactly {k} times as many white squares as black squares (i.e., black:white = 1:{k})? Place the integer in <answer>...</answer>.",
    "Recolor some black squares to white. How many recolorings are needed so the ratio of black to white becomes 1:{k}? Integer in <answer>...</answer>.",
    "The grid shows black and white squares. How many black squares must be recolored white to achieve a 1:{k} ratio of black:white? Place integer in <answer>...</answer>.",
    "Find the number of black squares to repaint white so that white squares are {k}× as many as black. Integer in <answer>...</answer>.",
    "How many black squares should be recolored white so that black:white = 1:{k}? Place the integer count in <answer>...</answer>.",
    "Count the black squares that need to become white to make the white-to-black ratio equal to {k}:1. Integer in <answer>...</answer>.",
    "Determine the number of black squares to recolor white so the resulting ratio of black to white equals 1:{k}. Place integer in <answer>...</answer>.",
    "Recolor black to white. How many recolorings yield black:white = 1:{k} in the displayed grid? Integer in <answer>...</answer>.",
]


class RecolorRatioQA(StandaloneVisualEnv):
    BENCHMARK_NUM_TOLERANCE_ABS = 0.001
    ENV_NAME = "recolor_ratio"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(9, level))
        # Need k*B - W = (k+1)*x with x in [0, B] => N divisible by k+1 for
        # a solution to exist (else any (B,W,N) with x = (k*B - W)/(k+1)
        # constrained to integer). We pick (k, N) so the equation is solvable.
        if level <= 2:
            return {"k": 2, "grid_dim": (3, 3)}    # N=9, 9/(2+1)=3 OK
        if level <= 4:
            return {"k": 3, "grid_dim": (4, 4)}    # N=16, 16/(3+1)=4 OK
        if level <= 6:
            return {"k": 4, "grid_dim": (5, 5)}    # N=25, 25/5=5 OK
        if level <= 8:
            return {"k": 4, "grid_dim": (5, 6)}    # N=30, 30/5=6 OK
        return {"k": 5, "grid_dim": (6, 6)}        # N=36, 36/6=6 OK

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = max(0, min(9, int(parameter.get("level", 0))))
        cfg = self._level_config(level)
        k = cfg["k"]
        rows, cols = cfg["grid_dim"]
        N = rows * cols
        rng = random.Random((self.seed or 0) * 5563 + level * 71 + 149)

        # Choose B (black count) such that there exists integer x where
        # (B - x) : (W + x) = 1 : k, i.e., k*(B-x) = W+x
        # => k*B - k*x = W + x => k*B - W = (k+1)*x
        # => x = (k*B - W) / (k+1)
        # We want x >= 0, x <= B
        for _ in range(60):
            B = rng.randint(2, N - 2)
            W = N - B
            num = k * B - W
            if num >= 0 and num % (k + 1) == 0:
                x = num // (k + 1)
                if 0 <= x <= B:
                    break
        else:
            return None

        # Build grid: B black squares + W white squares, randomly placed
        cells = ["B"] * B + ["W"] * W
        rng.shuffle(cells)
        grid = [cells[r * cols:(r + 1) * cols] for r in range(rows)]

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(k=k)
        answer = str(x)
        img = self._render(grid, rows, cols)
        return question, answer, img

    def _render(self, grid, rows, cols) -> Image.Image:
        fig, ax = plt.subplots(figsize=(cols, rows), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        for r in range(rows):
            for c in range(cols):
                color = "#1a1a1a" if grid[r][c] == "B" else "#ffffff"
                ax.add_patch(plt.Rectangle((c, rows - 1 - r), 1, 1,
                                           facecolor=color,
                                           edgecolor="#444",
                                           linewidth=1.2))
        ax.set_xlim(-0.1, cols + 0.1)
        ax.set_ylim(-0.1, rows + 0.1)
        ax.set_aspect("equal")
        ax.axis("off")
        return self.fig_to_pil(fig, dpi=120)


if __name__ == "__main__":
    env = RecolorRatioQA()
    for level in [0, 3, 6, 9]:
        ans = []
        for seed in range(20):
            if env.generate(seed, {"level": level}):
                ans.append(env._answer)
        from collections import Counter
        print(f"L{level}: {Counter(ans).most_common(5)}")
