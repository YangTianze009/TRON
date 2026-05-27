"""
Count strict hills + valleys in a numeric sequence.

Definition (matches the standard variant): collapse runs of equal consecutive
values into a single representative, then count indices that are strictly
greater than both neighbors (hill) or strictly less than both (valley).
Endpoints never count. Output the total.
"""
import random
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .standalone_base import StandaloneVisualEnv


_TEMPLATES = [
    "Your task is to count hills and valleys in the terrain sequence below.\n\n"
    "### Game Rules:\n"
    "1. Collapse consecutive equal values into a single value.\n"
    "2. A hill is an index strictly greater than both neighbors; a valley is strictly less.\n"
    "3. Endpoints do not count.\n"
    "4. Output total hills + valleys.\n\n"
    "### Coordinate System:\n"
    "- Sequence indexed 0..len(terrain)-1.\n\n"
    "### Current Puzzle State:\n"
    "- terrain = {terrain}\n\n"
    "### Output Format:\n"
    "Output the integer count inside <answer>...</answer>.\n"
    "Example: <answer>0</answer>",

    "Count hills and valleys in the terrain below.\n\n"
    "### Game Rules:\n"
    "- Strict local extrema after collapsing equals.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed.\n\n"
    "### Current Puzzle State:\n"
    "terrain = {terrain}\n\n"
    "### Output Format:\n"
    "Output integer inside <answer>...</answer>.",

    "Your task is to count strict turning points in the sequence below.\n\n"
    "### Game Rules:\n"
    "Standard CountHillsAndValleys.\n\n"
    "### Coordinate System:\n"
    "- 0-indexed sequence.\n\n"
    "### Current Puzzle State:\n"
    "terrain = {terrain}\n\n"
    "### Output Format:\n"
    "Output the integer inside <answer>...</answer>.",
]


def _count_hills_valleys(seq: List[int]) -> int:
    # collapse consecutive duplicates
    collapsed = []
    for v in seq:
        if not collapsed or collapsed[-1] != v:
            collapsed.append(v)
    count = 0
    for i in range(1, len(collapsed) - 1):
        a, b, c = collapsed[i - 1], collapsed[i], collapsed[i + 1]
        if (b > a and b > c) or (b < a and b < c):
            count += 1
    return count


class HillsValleysCountQA(StandaloneVisualEnv):
    ENV_NAME = "hills_valleys_count"

    def _level_config(self, level: int) -> Dict:
        level = max(0, min(level, 9))
        # L0: 4 elements with one obvious peak
        # gradually grow length and value range
        n = 4 + level
        n = min(n, 12)
        max_v = 5 + level * 2
        return {"level": level, "n": n, "max_v": max_v}

    def _generate_problem(
        self, seed: int, parameter: Dict
    ) -> Optional[Tuple[str, str, Image.Image]]:
        level = int(parameter.get("level", 0))
        cfg = self._level_config(level)
        rng = random.Random((self.seed or 0) * 1373 + level * 47 + 23)

        n = cfg["n"]
        if level == 0:
            # 4-element sequence with exactly one obvious peak: e.g. [1, 5, 3, 1].
            # answer is 1.
            base = rng.randint(1, 3)
            peak = base + rng.randint(3, 5)
            tail1 = base + rng.randint(0, 2)
            tail2 = max(1, base - rng.randint(0, 1))
            seq = [base, peak, tail1, tail2]
        else:
            seq = [rng.randint(1, cfg["max_v"]) for _ in range(n)]
            # avoid all-equal
            if len(set(seq)) == 1:
                seq[len(seq) // 2] += 1

        ans = _count_hills_valleys(seq)
        sidx = (self.seed or 0) % len(_TEMPLATES)
        question = _TEMPLATES[sidx].format(terrain=seq)
        img = self._render(seq)
        return question, str(ans), img

    def _render(self, seq: List[int]) -> Image.Image:
        n = len(seq)
        fig, ax = plt.subplots(figsize=(max(5, n * 0.55), 4), dpi=120)
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")
        ax.bar(range(n), seq, color="#6fb86f", edgecolor="#1f4d1f",
               linewidth=1.0, width=0.78)
        ymax = max(seq)
        for i, v in enumerate(seq):
            ax.text(i, v + ymax * 0.01 + 0.18, str(v),
                    ha="center", va="bottom", fontsize=10, color="#222")
        ax.set_xticks(range(n))
        ax.set_xticklabels([str(i) for i in range(n)])
        ax.set_xlabel("index")
        ax.set_ylabel("value")
        ax.set_ylim(0, ymax * 1.18 + 2)
        ax.grid(True, axis="y", alpha=0.3)

        from io import BytesIO
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).copy()
