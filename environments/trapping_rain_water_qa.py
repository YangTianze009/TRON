"""
Trapping Rain Water: given an elevation map (heights array) drawn as bars,
output the total amount of rain water that can be trapped between bars.

Difficulty axes:
  - array length (4 → 12)
  - height range (0-4 → 0-9)
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to compute the total trapped rain water for the elevation map below.\n\n"
    "### Game Rules:\n"
    "1. Each bar has unit width and integer height.\n"
    "2. Water trapped at index i = max(0, min(left_max[i], right_max[i]) - height[i]).\n"
    "3. Output the total water trapped over all indices.\n\n"
    "### Coordinate System:\n"
    "- Heights indexed 0..len(height)-1.\n\n"
    "### Current Puzzle State:\n"
    "- height = {height}\n\n"
    "### Output Format:\n"
    "Output the integer total trapped water inside <answer>...</answer>.\n"
    "Example: <answer>50</answer>",

    "Compute the total trapped rain water for the heights below.\n\n"
    "### Game Rules:\n"
    "- Standard trapping-rain-water.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "height = {height}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",

    "Your task is to compute trapped water for the heights below.\n\n"
    "### Game Rules:\n"
    "Total water trapped between bars after rain.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "height = {height}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",
]


def _trapped_water(heights: List[int]) -> int:
    n = len(heights)
    if n < 3:
        return 0
    left_max = [0] * n
    right_max = [0] * n
    left_max[0] = heights[0]
    for i in range(1, n):
        left_max[i] = max(left_max[i - 1], heights[i])
    right_max[-1] = heights[-1]
    for i in range(n - 2, -1, -1):
        right_max[i] = max(right_max[i + 1], heights[i])
    return sum(min(left_max[i], right_max[i]) - heights[i] for i in range(n))


class TrappingRainWaterQA(StandaloneVisualEnv):
    ENV_NAME = "trapping_rain_water"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n = 4 + level
        h_max = min(4 + level // 2, 9)
        return {"level": level, "n": n, "h_max": h_max}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1129 + level * 127 + 11)

        for _ in range(40):
            heights = [rng.randint(0, cfg["h_max"]) for _ in range(cfg["n"])]
            total = _trapped_water(heights)
            # Prefer non-zero trapped water (more interesting)
            if total > 0 or rng.random() < 0.2:
                break

        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(height=heights)
        img = self._render(heights, cfg["h_max"])
        return question, str(total), img

    def _render(self, heights, h_max) -> Image.Image:
        n = len(heights)
        fig, ax = plt.subplots(figsize=(max(5, n * 0.5), 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.bar(range(n), heights, color="#4C72B0", edgecolor="#1a3050",
               linewidth=1.0, width=0.85)
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(i) for i in range(n)])
        ax.set_yticks(range(h_max + 1))
        ax.set_ylim(0, h_max + 0.5)
        ax.set_xlabel("index")
        ax.set_ylabel("height")
        ax.grid(True, axis="y", alpha=0.3)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
