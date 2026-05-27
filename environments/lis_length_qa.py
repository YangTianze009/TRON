"""
Longest Increasing Subsequence: given a sequence drawn as bars, output the
length of the longest strictly increasing subsequence.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to compute the length of the longest strictly increasing subsequence below.\n\n"
    "### Game Rules:\n"
    "1. A subsequence is obtained by deleting zero or more elements without reordering.\n"
    "2. Strictly increasing means each successive element is greater than the previous.\n"
    "3. Output the maximum length of any such subsequence.\n\n"
    "### Coordinate System:\n"
    "- Sequence indexed 0..len(sequence)-1.\n\n"
    "### Current Puzzle State:\n"
    "- sequence = {sequence}\n\n"
    "### Output Format:\n"
    "Output the LIS length as an integer inside <answer>...</answer>.\n"
    "Example: <answer>5</answer>",

    "Compute the LIS length for the sequence below.\n\n"
    "### Game Rules:\n"
    "- Longest strictly increasing subsequence.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "sequence = {sequence}\n\n"
    "### Output Format:\n"
    "Output the integer length inside <answer>...</answer>.",

    "Your task is to find LIS length below.\n\n"
    "### Game Rules:\n"
    "Standard longest-increasing-subsequence DP.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "sequence = {sequence}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",
]


def _lis_length(seq: List[int]) -> int:
    if not seq:
        return 0
    from bisect import bisect_left
    tails = []
    for x in seq:
        idx = bisect_left(tails, x)
        if idx == len(tails):
            tails.append(x)
        else:
            tails[idx] = x
    return len(tails)


class LISLengthQA(StandaloneVisualEnv):
    ENV_NAME = "lis_length"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        n = 5 + level
        max_val = 8 + level
        return {"level": level, "n": n, "max_val": max_val}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1151 + level * 131 + 23)

        seq = [rng.randint(1, cfg["max_val"]) for _ in range(cfg["n"])]
        ans = _lis_length(seq)
        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(sequence=seq)
        img = self._render(seq, cfg["max_val"])
        return question, str(ans), img

    def _render(self, seq, max_val) -> Image.Image:
        n = len(seq)
        fig, ax = plt.subplots(figsize=(max(5, n * 0.5), 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.bar(range(n), seq, color="#55A868", edgecolor="#234d35",
               linewidth=1.0, width=0.85)
        for i, v in enumerate(seq):
            ax.text(i, v + 0.15, str(v), ha="center", va="bottom",
                    fontsize=10, color="#222")
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(i) for i in range(n)])
        ax.set_xlabel("index")
        ax.set_ylabel("value")
        ax.set_ylim(0, max_val + 1.5)
        ax.grid(True, axis="y", alpha=0.3)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
