"""
Maximum container area: given a sequence of bar heights drawn as a bar chart,
output the maximum value of (j - i) * min(h[i], h[j]) over all pairs i < j.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to compute the maximum container area for the bar heights below.\n\n"
    "### Game Rules:\n"
    "1. Pick two indices i < j; the container holds water with width (j - i) and height min(heights[i], heights[j]).\n"
    "2. Maximum area = max over all pairs (i, j).\n\n"
    "### Coordinate System:\n"
    "- Heights indexed 0..len(heights)-1.\n\n"
    "### Current Puzzle State:\n"
    "- heights = {heights}\n\n"
    "### Output Format:\n"
    "Output the maximum area as an integer inside <answer>...</answer>.\n"
    "Example: <answer>72</answer>",

    "Compute the max container water area for the bars below.\n\n"
    "### Game Rules:\n"
    "- Two-pointer max area: max (j-i) * min(h[i], h[j]).\n\n"
    "### Coordinate System:\n"
    "- 0-indexed heights array.\n\n"
    "### Current Puzzle State:\n"
    "heights = {heights}\n\n"
    "### Output Format:\n"
    "Output the integer area inside <answer>...</answer>.",

    "Your task is to find max water area for the heights below.\n\n"
    "### Game Rules:\n"
    "Standard ContainerWithMostWater.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "heights = {heights}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",
]


def _max_area(heights: List[int]) -> int:
    n = len(heights)
    best = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = (j - i) * min(heights[i], heights[j])
            if a > best:
                best = a
    return best


class ContainerMaxWaterQA(StandaloneVisualEnv):
    ENV_NAME = "container_max_water"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0: 4 bars, max value already obvious
        # increase n and height range with level
        n = 4 + level
        n = min(n, 12)
        max_h = 5 + level * 2
        return {"level": level, "n": n, "max_h": max_h}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 991 + level * 41 + 17)

        n = cfg["n"]
        if level == 0:
            # 4 bars where the optimum is obvious: two tall bars at the ends.
            # e.g. [4, 1, 2, 5] — area = 3 * min(4,5) = 12.
            left = rng.randint(3, 5)
            right = rng.randint(3, 5)
            mid_lo = rng.randint(1, 2)
            mid_hi = rng.randint(1, 2)
            heights = [left, mid_lo, mid_hi, right]
        else:
            heights = [rng.randint(1, cfg["max_h"]) for _ in range(n)]
            # ensure not all equal (would still be valid but boring)
            if len(set(heights)) == 1:
                heights[0] += 1

        ans = _max_area(heights)
        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(heights=heights)
        img = self._render(heights)
        return question, str(ans), img

    def _render(self, heights: List[int]) -> Image.Image:
        n = len(heights)
        fig, ax = plt.subplots(figsize=(max(5, n * 0.5), 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.bar(range(n), heights, color="#5fa5d9", edgecolor="#1d3a55",
               linewidth=1.0, width=0.55)
        ymax = max(heights)
        for i, v in enumerate(heights):
            ax.text(i, v + ymax * 0.01 + 0.18, str(v),
                    ha="center", va="bottom", fontsize=10, color="#222")
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(i) for i in range(n)])
        ax.set_xlabel("index")
        ax.set_ylabel("height")
        ax.set_ylim(0, ymax * 1.18 + 2)
        ax.grid(True, axis="y", alpha=0.3)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
