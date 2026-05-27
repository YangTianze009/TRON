"""
H-Index: given a list of citation counts (drawn as bars), compute the
researcher's h-index = max h such that at least h papers have ≥ h citations.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to compute the h-index for the citations array below.\n\n"
    "### Game Rules:\n"
    "1. The h-index is the maximum integer h such that at least h papers have at least h citations each.\n\n"
    "### Coordinate System:\n"
    "- Citations array indexed 0..len(citations)-1.\n\n"
    "### Current Puzzle State:\n"
    "- citations = {citations}\n\n"
    "### Output Format:\n"
    "Output the h-index as an integer inside <answer>...</answer>.\n"
    "Example: <answer>6</answer>",

    "Compute the h-index for the citation array below.\n\n"
    "### Game Rules:\n"
    "- Largest h such that >=h papers have >=h citations.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed array.\n\n"
    "### Current Puzzle State:\n"
    "citations = {citations}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",

    "Your task is to compute h-index given citations.\n\n"
    "### Game Rules:\n"
    "Standard HIndex.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "citations = {citations}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",
]


def _h_index(citations: List[int]) -> int:
    sorted_c = sorted(citations, reverse=True)
    h = 0
    for i, c in enumerate(sorted_c):
        if c >= i + 1:
            h = i + 1
        else:
            break
    return h


class HIndexQA(StandaloneVisualEnv):
    ENV_NAME = "h_index"

    def _level_config(self, level: int) -> Dict:
        # 2026-05-04 R3: softened — was n=4+level (so L9 had 13 papers) and
        # max_cite=5+level*2 (so L9 had 0..23 cites). The bar chart became
        # cluttered with 13 bars, hard to read each value (L0=0.7 L9=0.7
        # — both low). Cap at 10 papers and 20 max cites at L9.
        level = max(0, min(level, 9))
        n = min(4 + level, 10)
        max_cite = min(5 + level * 2, 20)
        return {"level": level, "n": n, "max_cite": max_cite}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1163 + level * 139 + 7)

        citations = [rng.randint(0, cfg["max_cite"]) for _ in range(cfg["n"])]
        ans = _h_index(citations)
        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(citations=citations)
        img = self._render(citations)
        return question, str(ans), img

    def _render(self, citations) -> Image.Image:
        n = len(citations)
        fig, ax = plt.subplots(figsize=(max(5, n * 0.5), 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.bar(range(n), citations, color="#C44E52", edgecolor="#5a1a1c",
               linewidth=1.0, width=0.85)
        for i, v in enumerate(citations):
            ax.text(i, v + max(citations) * 0.01 + 0.15, str(v),
                    ha="center", va="bottom", fontsize=9, color="#222")
        ax.set_xticks(range(n))
        ax.set_xticklabels([f"P{i+1}" for i in range(n)])
        ax.set_xlabel("paper")
        ax.set_ylabel("citations")
        ax.set_ylim(0, max(citations) * 1.15 + 2)
        ax.grid(True, axis="y", alpha=0.3)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
