"""
Largest Rectangle in Histogram — structured-puzzle style (algorithm problem H40).

Rule: Given an array of bar heights (unit-width bars), find the largest
rectangular area that fits inside the histogram.

Studied reference qids (design notes lines 255-265):
  - idx=631 (D=1): heights `[20, 7, 6, 6, 18, 17, 4, 1]` -> 36
  - idx=655 (D=3): array of ~12
  - idx=670 (D=5): length 15

Answer format: single integer (the maximum rectangular area).

Difficulty axis: array length 5 -> 16; max bar height 5 -> 25.
"""
import random
import re
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to compute the largest rectangle area in the histogram below.\n\n"
    "### Game Rules:\n"
    "1. Each bar has unit width and an integer height.\n"
    "2. The largest inscribed rectangle has area max over (i, j) of (j - i + 1) * min(heights[i..j]).\n\n"
    "### Coordinate System:\n"
    "- Bars indexed 0..len(heights)-1.\n\n"
    "### Current Puzzle State:\n"
    "- heights = {heights_str}\n\n"
    "### Output Format:\n"
    "Output the largest area as an integer inside <answer>...</answer>.\n"
    "Example: <answer>36</answer>",

    "Compute the largest rectangle in the histogram below.\n\n"
    "### Game Rules:\n"
    "- Largest inscribed rectangle.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "heights = {heights_str}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",

    "Your task is to find the LargestRectangleInHistogram value below.\n\n"
    "### Game Rules:\n"
    "Standard monotonic-stack problem.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "heights = {heights_str}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",
]


def _largest_rectangle_in_histogram(heights: List[int]) -> int:
    """Standard monotonic-stack O(n) algorithm."""
    stack: List[int] = []
    max_area = 0
    n = len(heights)
    for i in range(n + 1):
        cur = 0 if i == n else heights[i]
        while stack and heights[stack[-1]] > cur:
            top = stack.pop()
            left = stack[-1] if stack else -1
            width = i - left - 1
            area = heights[top] * width
            if area > max_area:
                max_area = area
        stack.append(i)
    return max_area


class LargestRectangleInHistogramQA(StandaloneVisualEnv):
    ENV_NAME = "largest_rectangle_in_histogram"
    # Use min/max ratio shaping for numeric answers.
    shape_strategy = "min_over_max"
    shape_beta = 2.0

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # 2026-05-04 R3 retry: softened bump — v5=0.20 (worst). Existing
        # R3 only trimmed L0 (5→4) and L9 (16→13); whole gradient still
        # too steep. Lower L0 to 3 bars (toy case), shrink hmax across
        # the board, cap L9 at 11/16 — keeps monotonic L0<L9 gradient.
        if level == 0:
            length = 3
            hmax = 5
        elif level <= 2:
            length = 5
            hmax = 8
        elif level <= 4:
            length = 6
            hmax = 10
        elif level <= 6:
            length = 8
            hmax = 12
        elif level <= 8:
            length = 10
            hmax = 14
        else:
            length = 11
            hmax = 16
        return {"level": level, "length": length, "hmax": hmax}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        length = cfg["length"]
        hmax = cfg["hmax"]
        rng = random.Random((self.seed or 0) * 1471 + level * 41 + 7)

        for _attempt in range(8):
            heights = [rng.randint(1, hmax) for _ in range(length)]
            # Avoid all-equal histograms (trivial)
            if len(set(heights)) <= 1:
                continue
            ans_value = _largest_rectangle_in_histogram(heights)
            if ans_value <= 0:
                continue
            self._heights = heights
            heights_str = "[" + ", ".join(str(h) for h in heights) + "]"
            sidx = (self.seed or 0) % len(_TEMPLATES)
            question = _TEMPLATES[sidx].format(heights_str=heights_str)
            img = self._render(heights, ans_value)
            return question, str(ans_value), img
        return None

    def _render(self, heights: List[int], ans_value: int) -> Image.Image:
        n = len(heights)
        hmax = max(heights)
        fig_w = max(4.0, 0.45 * n + 1.0)
        fig_h = max(3.0, 0.18 * hmax + 1.5)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=130)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        # Bars
        xs = list(range(n))
        bar_color = "#3498db"
        ax.bar(xs, heights, width=0.96, color=bar_color,
               edgecolor="#1a3a6e", linewidth=1.2, align="edge")
        for i, h in enumerate(heights):
            ax.text(i + 0.5, h + 0.3, str(h),
                    ha="center", va="bottom", fontsize=9,
                    fontweight="bold", color="#1a3a6e")
        ax.set_xticks([i + 0.5 for i in range(n)])
        ax.set_xticklabels([str(i) for i in range(n)], fontsize=9)
        ax.set_ylim(0, hmax + max(2, hmax * 0.15))
        ax.set_xlim(-0.05, n + 0.05)
        ax.set_xlabel("bar index", fontsize=10)
        ax.set_ylabel("height", fontsize=10)
        ax.grid(True, axis="y", alpha=0.25, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()

    def _check_answer(self, predicted: str, ground_truth: str) -> bool:
        # Defer to base numeric matching by short-circuiting
        return super()._check_answer(predicted, ground_truth)
